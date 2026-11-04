"""Fit a validation-only calibration model for the exported Resolution neural model."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from dashboard.backend.neural_resolution import score_resolution_neural
DATA = ROOT / "data" / "processed"
TARGET = "issue_resolution_risk_label"


def ece(labels, probabilities, bins=10):
    index = np.digitize(probabilities, np.linspace(0, 1, bins + 1)[1:-1], right=True)
    return float(sum(abs(labels[index == bucket].mean() - probabilities[index == bucket].mean()) * (index == bucket).mean()
                     for bucket in range(bins) if (index == bucket).any()))


def describe(labels, probabilities):
    return {"records": int(len(labels)), "positives": int(labels.sum()),
            "brier_score": round(float(brier_score_loss(labels, probabilities)), 6),
            "expected_calibration_error_10_bins": round(ece(labels, probabilities), 6),
            "roc_auc": round(float(roc_auc_score(labels, probabilities)), 6)}


def main():
    dataset = pd.read_csv(DATA / "training_dataset_v6_2_effort.csv", low_memory=False)
    dataset[TARGET] = pd.to_numeric(dataset[TARGET], errors="coerce")
    splits = pd.read_csv(DATA / "source_aware_splits_v6_1_fixed.csv")
    assignment = splits.loc[splits.target == TARGET, ["source_collection", "issue_key", "split"]]
    task = dataset.merge(assignment, on=["source_collection", "issue_key"], validate="one_to_one").dropna(subset=[TARGET]).copy()
    task[TARGET] = task[TARGET].astype(int)
    valid = task.loc[task.split == "validation"].copy()
    test = task.loc[task.split == "test"].copy()
    raw_valid, _ = score_resolution_neural(valid, include_explanations=False)
    raw_test, _ = score_resolution_neural(test, include_explanations=False)
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
    calibrator.fit(raw_valid.reshape(-1, 1), valid[TARGET].to_numpy())
    calibrated = calibrator.predict_proba(raw_test.reshape(-1, 1))[:, 1]
    joblib.dump(calibrator, DATA / "resolution_neural_probability_calibrator_v6_2.joblib")
    output = test[["source_collection", "issue_key", TARGET, "story_points_at_cutoff", "story_points_available_at_cutoff"]].copy()
    output["raw_neural_resolution_probability"] = raw_test
    output["calibrated_neural_resolution_probability"] = calibrated
    output.to_csv(DATA / "resolution_neural_test_predictions_calibrated_v6_2.csv", index=False)
    report = {"model": "multimodal Resolution neural model + validation-only Platt scaling",
              "test_raw": describe(test[TARGET].to_numpy(), raw_test),
              "test_calibrated": describe(test[TARGET].to_numpy(), calibrated),
              "effort_qualified_test_records": int(output.story_points_available_at_cutoff.sum()),
              "methodology_note": "Calibration was fit on the source-aware validation split only; test labels were not used during fitting."}
    (DATA / "resolution_neural_calibration_report_v6_2.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
