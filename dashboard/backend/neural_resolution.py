"""Inference adapter for the trusted v6.1 Resolution Risk neural bundle.

The bundle contains a TensorFlow multimodal model, its fitted structured-data
preprocessor and its local MiniLM text encoder.  It is intentionally loaded
lazily: users who only upload an already-scored risk register do not need the
neural runtime installed.
"""
from functools import lru_cache
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "models" / "resolution_neural_bundle_v6_1"
REQUIRED_INPUTS = {"issue_text_at_cutoff", "early_event_sequence"}


def _require_bundle_files():
    required = [
        BUNDLE / "resolution_model.keras",
        BUNDLE / "preprocessing.joblib",
        BUNDLE / "text_encoder",
        BUNDLE / "metadata.json",
        BUNDLE / "structured_train_mean.npy",
        BUNDLE / "text_embedding_train_mean.npy",
    ]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise ValueError("Resolution neural bundle is incomplete. Missing: " + ", ".join(missing))


@lru_cache(maxsize=1)
def load_neural_runtime():
    """Load only the locally generated, trusted neural-model artifacts."""
    _require_bundle_files()
    try:
        import tensorflow as tf
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Neural Resolution Risk scoring needs TensorFlow and sentence-transformers. "
            "Install the dashboard requirements, then restart the server."
        ) from exc

    metadata = json.loads((BUNDLE / "metadata.json").read_text(encoding="utf-8"))
    preprocessing = joblib.load(BUNDLE / "preprocessing.joblib")
    model = tf.keras.models.load_model(BUNDLE / "resolution_model.keras", compile=False)
    encoder = SentenceTransformer(str(BUNDLE / "text_encoder"), device="cpu")
    structured_reference = np.load(BUNDLE / "structured_train_mean.npy").astype("float32")
    text_reference = np.load(BUNDLE / "text_embedding_train_mean.npy").astype("float32")
    return tf, metadata, preprocessing, model, encoder, structured_reference, text_reference


def score_resolution_neural(frame, include_explanations=True):
    """Return Resolution Risk scores and branch-level local explanations.

    The text and workflow sequence are first-30-day, cutoff-safe fields.  They
    are required rather than defaulted to empty values, to avoid misleading
    dashboard scores for incomplete candidate files.
    """
    missing_inputs = sorted(REQUIRED_INPUTS - set(frame.columns))
    if missing_inputs:
        raise ValueError(
            "Candidate CSV is not neural-ready. Rebuild it with "
            "work/build_candidate_sprint_items.py; missing: " + ", ".join(missing_inputs)
        )
    tf, metadata, preprocessing, model, encoder, structured_reference, text_reference = load_neural_runtime()
    numeric = preprocessing["numeric_features"]
    categorical = preprocessing["categorical_features"]
    required_structured = numeric + categorical
    missing_structured = [column for column in required_structured if column not in frame.columns]
    if missing_structured:
        raise ValueError("Candidate input is missing neural model features: " + ", ".join(missing_structured))

    values = frame[required_structured].copy()
    for column in numeric:
        values[column] = pd.to_numeric(values[column], errors="coerce")
    for column in categorical:
        values[column] = values[column].replace("", np.nan)
    structured = preprocessing["selector"].transform(
        preprocessing["processor"].transform(values)
    ).astype("float32")
    text = frame["issue_text_at_cutoff"].fillna("").astype(str).str.slice(
        0, int(metadata.get("text_character_limit", 3000))
    ).tolist()
    text_embeddings = encoder.encode(
        text, batch_size=32, normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")
    sequence = frame["early_event_sequence"].fillna("").astype(str).to_numpy()
    if structured_reference.shape[0] != structured.shape[1]:
        raise ValueError("The neural structured reference does not match the saved preprocessor.")
    if text_reference.shape[0] != text_embeddings.shape[1]:
        raise ValueError("The neural text reference does not match the saved MiniLM encoder.")

    def inputs(structured_values, text_values, sequence_values):
        return {
            "structured_input": structured_values,
            "text_input": text_values,
            "sequence_input": tf.reshape(tf.constant(sequence_values, dtype=tf.string), (-1, 1)),
        }

    def predict(model_inputs):
        return model.predict(model_inputs, batch_size=128, verbose=0).reshape(-1)

    # Branch-level ablation: keep the other two inputs fixed and replace one
    # branch with the exact training-set reference saved with the model.
    # This measures local score sensitivity; branch deltas need not sum to the
    # full score because the neural branches can interact.
    full_scores = predict(inputs(structured, text_embeddings, sequence))
    if not include_explanations:
        return full_scores, None
    structured_baseline = np.broadcast_to(structured_reference, structured.shape).copy()
    text_baseline = np.broadcast_to(text_reference, text_embeddings.shape).copy()
    empty_sequence = np.full(len(frame), "", dtype=str)
    ablations = {
        "Structured early-development signals": predict(inputs(structured_baseline, text_embeddings, sequence)),
        "Issue text at the 30-day cutoff": predict(inputs(structured, text_baseline, sequence)),
        "Workflow-event sequence in the first 30 days": predict(inputs(structured, text_embeddings, empty_sequence)),
    }
    explanations = []
    for row_index, full_score in enumerate(full_scores):
        factors = []
        for label, ablated_scores in ablations.items():
            delta = float(full_score - ablated_scores[row_index])
            factors.append({
                "feature": label,
                "label": label,
                "observed": "available",
                "direction": "increases" if delta >= 0 else "decreases",
                "contribution": delta,
            })
        factors.sort(key=lambda factor: abs(factor["contribution"]), reverse=True)
        explanations.append({
            "method": (
                "Neural branch-level ablation: each branch is replaced with its saved training-set "
                "reference while the other two remain fixed. Contribution is the probability change."
            ),
            "factors": factors,
            "attribution_available": True,
            "limitation": "Branch contributions can interact and therefore do not necessarily sum to the final score.",
        })
    return full_scores, explanations
