"""Calibrate Resolution Risk probabilities for expected-utility evaluation.

Fits Platt scaling only on the fixed validation partition and measures it on
the fixed test partition. The saved selected Random Forest is used because it
is the reproducible exported model available for retrospective evaluation.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODEL = DATA / "issue_resolution_risk_selected_model_v6_1.joblib"
DATASET = DATA / "training_dataset_v6_2_effort.csv"
SPLITS = DATA / "source_aware_splits_v6_1_fixed.csv"
TARGET = "issue_resolution_risk_label"


def ece(y_true, probability, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    index = np.digitize(probability, edges[1:-1], right=True)
    total = len(y_true)
    return float(sum(
        abs(y_true[index == bucket].mean() - probability[index == bucket].mean()) * (index == bucket).sum() / total
        for bucket in range(bins) if (index == bucket).any()
    ))


def score(artifact, frame):
    columns = artifact["numeric_features"] + artifact["categorical_features"]
    values = frame[columns].copy()
    for column in artifact["numeric_features"]:
        values[column] = pd.to_numeric(values[column], errors="coerce")
    matrix = artifact["selector"].transform(artifact["processor"].transform(values))
    positive_index = list(artifact["model"].classes_).index(1)
    return artifact["model"].predict_proba(matrix)[:, positive_index]


def metrics(labels, probabilities):
    return {
        "records": int(len(labels)),
        "positives": int(labels.sum()),
        "brier_score": round(float(brier_score_loss(labels, probabilities)), 6),
        "expected_calibration_error_10_bins": round(ece(labels, probabilities), 6),
        "roc_auc": round(float(roc_auc_score(labels, probabilities)), 6),
    }


def main():
    for path in (MODEL, DATASET, SPLITS):
        if not path.exists():
            raise FileNotFoundError(path)
    artifact = joblib.load(MODEL)
    dataset = pd.read_csv(DATASET, low_memory=False)
    dataset[TARGET] = pd.to_numeric(dataset[TARGET], errors="coerce")
    split = pd.read_csv(SPLITS)
    assignments = split.loc[split["target"] == TARGET, ["source_collection", "issue_key", "split"]]
    task = dataset.merge(assignments, on=["source_collection", "issue_key"], how="inner", validate="one_to_one")
    task = task.dropna(subset=[TARGET]).copy()
    task[TARGET] = task[TARGET].astype(int)
    validation = task.loc[task["split"] == "validation"].copy()
    test = task.loc[task["split"] == "test"].copy()
    raw_validation = score(artifact, validation)
    raw_test = score(artifact, test)
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
    calibrator.fit(raw_validation.reshape(-1, 1), validation[TARGET].to_numpy())
    calibrated_test = calibrator.predict_proba(raw_test.reshape(-1, 1))[:, 1]
    joblib.dump(calibrator, DATA / "resolution_probability_calibrator_v6_2.joblib")
    predictions = test[["source_collection", "issue_key", TARGET, "story_points_at_cutoff", "story_points_available_at_cutoff"]].copy()
    predictions["raw_resolution_probability"] = raw_test
    predictions["calibrated_resolution_probability"] = calibrated_test
    predictions.to_csv(DATA / "resolution_test_predictions_calibrated_v6_2.csv", index=False)
    effort_test = predictions.loc[predictions["story_points_available_at_cutoff"] == 1]
    report = {
        "model": "issue_resolution_risk_selected_model_v6_1 Random Forest + validation-only Platt scaling",
        "calibration_split": "source-aware validation partition",
        "evaluation_split": "source-aware test partition",
        "test_raw": metrics(test[TARGET].to_numpy(), raw_test),
        "test_calibrated": metrics(test[TARGET].to_numpy(), calibrated_test),
        "effort_qualified_test_records": int(len(effort_test)),
        "calibrator_file": "resolution_probability_calibrator_v6_2.joblib",
        "prediction_file": "resolution_test_predictions_calibrated_v6_2.csv",
        "limitation": "This calibrates the exported Random Forest evaluation model. The neural Resolution scorer requires its own validation-based calibration artifact before it can be used in expected-utility optimisation.",
    }
    (DATA / "resolution_calibration_report_v6_2.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
