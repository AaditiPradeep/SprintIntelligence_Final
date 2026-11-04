"""Train and export the v6.1 Resolution neural model in Google Colab.

Run from /content after uploading the v6.1 dataset, feature list and fixed
source-aware split. The exported bundle is not automatically installed in the
dashboard. Only load model/joblib files you trust.
"""
import hashlib
import json
import random
import sys
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import torch
import sentence_transformers
from sentence_transformers import SentenceTransformer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path("training_dataset_v6_1.csv")
FEATURES = Path("model_feature_columns_v6.json")
SPLITS = Path("source_aware_splits_v6_1_fixed.csv")
OUT = Path("resolution_neural_bundle_v6_1")
ZIP = Path("resolution_neural_bundle_v6_1.zip")
TARGET = "issue_resolution_risk_label"
CATEGORICAL = ["source_collection", "priority_at_cutoff", "issue_type_at_cutoff", "status_at_cutoff"]
SEED = 42
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_LIMIT = 3000


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def best_f1_threshold(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if not len(thresholds):
        return .5
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(np.clip(thresholds[int(np.nanargmax(f1))], .01, .99))


def metrics(labels, scores, threshold):
    pred = (scores >= threshold).astype(int)
    return {
        "threshold": threshold,
        "records": int(len(labels)),
        "positive_records": int(labels.sum()),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision_pr_auc": float(average_precision_score(labels, scores)),
        "confusion_matrix_labels_0_1": confusion_matrix(labels, pred, labels=[0, 1]).tolist(),
    }


def build_model(structured_dim, text_dim, vocabulary):
    structured_input = tf.keras.Input(shape=(structured_dim,), name="structured_input")
    structured = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(structured_input)
    structured = tf.keras.layers.BatchNormalization()(structured)
    structured = tf.keras.layers.Dropout(.30)(structured)
    text_input = tf.keras.Input(shape=(text_dim,), name="text_input")
    text = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(text_input)
    text = tf.keras.layers.Dropout(.25)(text)
    sequence_input = tf.keras.Input(shape=(1,), dtype=tf.string, name="sequence_input")
    vectorizer = tf.keras.layers.TextVectorization(max_tokens=200, output_mode="int", output_sequence_length=80,
                                                   standardize=None, split="whitespace")
    vectorizer.set_vocabulary(vocabulary)
    sequence = vectorizer(sequence_input)
    sequence = tf.keras.layers.Embedding(len(vocabulary), 16, mask_zero=True)(sequence)
    # Some issues have empty/masked workflow sequences. The cuDNN GRU can
    # reject these masks, so use the compatible TensorFlow GRU implementation.
    sequence = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(16, use_cudnn=False))(sequence)
    sequence = tf.keras.layers.Dropout(.25)(sequence)
    merged = tf.keras.layers.Concatenate()([structured, text, sequence])
    merged = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(merged)
    merged = tf.keras.layers.BatchNormalization()(merged)
    merged = tf.keras.layers.Dropout(.35)(merged)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="risk")(merged)
    model = tf.keras.Model(inputs={"structured_input": structured_input, "text_input": text_input,
                                   "sequence_input": sequence_input}, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc")])
    return model


def main():
    for path in (DATA, FEATURES, SPLITS):
        if not path.is_file():
            raise FileNotFoundError(f"Upload the required file: {path}")
    # Do not overwrite a previous trained model without explicit user action.
    if OUT.exists() or ZIP.exists():
        raise FileExistsError(f"{OUT} or {ZIP} already exists. Save it elsewhere before rerunning.")
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    safe = json.loads(FEATURES.read_text(encoding="utf-8"))
    if not isinstance(safe, list) or not set(CATEGORICAL).issubset(safe):
        raise ValueError("The safe feature list is missing the expected categorical columns.")
    numeric = [c for c in safe if c not in CATEGORICAL]
    df = pd.read_csv(DATA, low_memory=False)
    required = set(safe + ["source_collection", "issue_key", "created_at", TARGET,
                           "issue_text_at_cutoff", "early_event_sequence"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Dataset is missing: " + ", ".join(sorted(missing)))
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce", format="mixed")
    df = df.dropna(subset=["created_at", "source_collection", "issue_key", TARGET]).copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="raise").astype(int)
    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Resolution labels must be binary (0 or 1).")
    df["issue_text_at_cutoff"] = df["issue_text_at_cutoff"].fillna("").astype(str).str.slice(0, TEXT_LIMIT)
    df["early_event_sequence"] = df["early_event_sequence"].fillna("").astype(str)
    split = pd.read_csv(SPLITS)
    assignments = split.loc[split["target"] == TARGET, ["source_collection", "issue_key", "split"]]
    task = df.merge(assignments, on=["source_collection", "issue_key"], how="inner", validate="one_to_one")
    if len(task) != len(assignments):
        raise ValueError("The fixed split file does not match the dataset.")
    if set(task["split"].unique()) != {"train", "validation", "test"}:
        raise ValueError("Expected train, validation, and test splits.")
    train, valid, test = [task.loc[task["split"] == name].copy() for name in ("train", "validation", "test")]
    if any(frame[TARGET].nunique() != 2 for frame in (train, valid, test)):
        raise ValueError("Every split needs examples of both resolution classes.")
    print("Split sizes and positive counts:")
    for name, frame in (("train", train), ("validation", valid), ("test", test)):
        print(f"  {name}: {len(frame):,} issues, {frame[TARGET].sum():,} positives")

    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                              ("scale", StandardScaler())]), numeric),
        ("category", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                               ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL),
    ])
    columns = numeric + CATEGORICAL
    train_x = preprocessor.fit_transform(train[columns]).astype("float32")
    valid_x = preprocessor.transform(valid[columns]).astype("float32")
    test_x = preprocessor.transform(test[columns]).astype("float32")
    selector = SelectKBest(mutual_info_classif, k=min(30, train_x.shape[1]))
    train_x = selector.fit_transform(train_x, train[TARGET].to_numpy()).astype("float32")
    valid_x = selector.transform(valid_x).astype("float32")
    test_x = selector.transform(test_x).astype("float32")
    names = preprocessor.get_feature_names_out()[selector.get_support()].tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {TEXT_MODEL} on {device}; internet is needed on the first run.")
    encoder = SentenceTransformer(TEXT_MODEL, device=device)
    def embed(frame):
        return encoder.encode(frame["issue_text_at_cutoff"].tolist(), batch_size=64,
                              show_progress_bar=True, normalize_embeddings=True,
                              convert_to_numpy=True).astype("float32")
    train_text, valid_text, test_text = embed(train), embed(valid), embed(test)
    # The text encoder is no longer needed on the GPU once embeddings exist.
    if device == "cuda":
        encoder.to("cpu")
        torch.cuda.empty_cache()
    vocab_layer = tf.keras.layers.TextVectorization(max_tokens=200, output_mode="int", output_sequence_length=80,
                                                    standardize=None, split="whitespace")
    vocab_layer.adapt(tf.data.Dataset.from_tensor_slices(train["early_event_sequence"].to_numpy()).batch(256))
    vocabulary = vocab_layer.get_vocabulary()

    def pack(frame, structured, text):
        return {"structured_input": structured, "text_input": text,
                "sequence_input": tf.reshape(tf.constant(frame["early_event_sequence"].to_numpy(), dtype=tf.string), (-1, 1))}
    train_inputs = pack(train, train_x, train_text)
    valid_inputs = pack(valid, valid_x, valid_text)
    test_inputs = pack(test, test_x, test_text)
    model = build_model(train_x.shape[1], train_text.shape[1], vocabulary)
    model.summary()
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=4, restore_best_weights=True),
                 tf.keras.callbacks.ReduceLROnPlateau(monitor="val_pr_auc", mode="max", patience=2, factor=.5, min_lr=1e-5)]
    history = model.fit(train_inputs, train[TARGET].to_numpy(), validation_data=(valid_inputs, valid[TARGET].to_numpy()),
                        epochs=25, batch_size=128, callbacks=callbacks, verbose=1)
    valid_scores = model.predict(valid_inputs, batch_size=256, verbose=0).reshape(-1)
    threshold = best_f1_threshold(valid[TARGET].to_numpy(), valid_scores)
    test_scores = model.predict(test_inputs, batch_size=256, verbose=0).reshape(-1)
    results = {"validation": metrics(valid[TARGET].to_numpy(), valid_scores, threshold),
               "test": metrics(test[TARGET].to_numpy(), test_scores, threshold)}
    print("\nEvaluation (threshold selected on validation):")
    print(json.dumps(results, indent=2))

    OUT.mkdir()
    model.save(OUT / "resolution_model.keras")
    joblib.dump({"processor": preprocessor, "selector": selector, "numeric_features": numeric,
                 "categorical_features": CATEGORICAL, "selected_feature_names": names}, OUT / "preprocessing.joblib")
    encoder.save(str(OUT / "text_encoder"))
    np.save(OUT / "structured_train_mean.npy", train_x.mean(axis=0))
    np.save(OUT / "text_embedding_train_mean.npy", train_text.mean(axis=0))
    metadata = {
        "task": TARGET, "model": "multimodal TensorFlow neural model", "loss": "binary_crossentropy",
        "threshold_validation_f1": threshold, "observation_days": 30,
        "outcome_definition": "resolution after 180 days from creation, or unresolved at deadline",
        "text_encoder": TEXT_MODEL, "text_character_limit": TEXT_LIMIT,
        "sequence_max_tokens": 200, "sequence_length": 80, "random_seed": SEED,
        "selected_feature_names": names, "source_files_sha256": {p.name: sha256(p) for p in (DATA, FEATURES, SPLITS)},
        "versions": {"python": sys.version, "tensorflow": tf.__version__, "scikit_learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__,
                     "torch": torch.__version__, "sentence_transformers": sentence_transformers.__version__},
        "explanation_note": "Training means are reference data for future attribution; no Integrated Gradients is computed here.",
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.DataFrame(history.history).to_csv(OUT / "training_history.csv", index_label="epoch")
    pd.DataFrame({"source_collection": test["source_collection"].to_numpy(), "issue_key": test["issue_key"].to_numpy(),
                  "true_label": test[TARGET].to_numpy(), "risk_score": test_scores,
                  "prediction": (test_scores >= threshold).astype(int)}).to_csv(OUT / "test_predictions.csv", index=False)

    # Reload every saved inference component and check it on the same examples.
    restored = tf.keras.models.load_model(OUT / "resolution_model.keras", compile=False)
    prep = joblib.load(OUT / "preprocessing.joblib")
    encoder_copy = SentenceTransformer(str(OUT / "text_encoder"), device="cpu")
    sample = test.head(min(5, len(test)))
    sample_x = prep["selector"].transform(prep["processor"].transform(sample[columns])).astype("float32")
    sample_text = encoder_copy.encode(sample["issue_text_at_cutoff"].tolist(), batch_size=5,
                                      normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    restored_scores = restored.predict(pack(sample, sample_x, sample_text), verbose=0).reshape(-1)
    if not np.allclose(restored_scores, test_scores[:len(sample)], atol=1e-5, rtol=1e-5):
        raise RuntimeError("Saved inference bundle did not reproduce model predictions. Do not deploy it.")
    with zipfile.ZipFile(ZIP, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in OUT.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(OUT.parent))
    print(f"\nReload check passed. Download: {ZIP.resolve()}")
    print("The bundle must be integrated into the dashboard separately; it does not yet compute neural explanations.")


if __name__ == "__main__":
    main()
