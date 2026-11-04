"""Score candidate issues and explain the exact saved classical models.

Only load locally trusted joblib artifacts: pickle-based formats can execute code.
The logistic baseline is the transformed mean of the task's training rows.
Tree SHAP uses the Random Forest's training-path distribution as its reference.
"""
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
MODELS = {
    "volatility_risk": ("requirement_volatility_selected_model_v6_1.joblib", "requirement_volatility_label"),
    "reopen_risk": ("issue_reopen_selected_model_v6_1.joblib", "issue_reopen_label"),
}
RESOLUTION_RF = ("issue_resolution_risk_selected_model_v6_1.joblib", "issue_resolution_risk_label")
RESOLUTION_CALIBRATOR = "resolution_probability_calibrator_v6_2.joblib"


def normalise_priority(value):
    """Map source-specific Jira priority strings to one dashboard vocabulary."""
    raw = str(value or "").strip()
    label = raw.lower()
    if "p1" in label or "urgent" in label or "critical" in label or "blocker" in label or "highest" in label:
        return "Highest"
    if "p2" in label or "high" in label:
        return "High"
    if "p3" in label or "medium" in label or "normal" in label:
        return "Medium"
    if "p4" in label or "low" in label:
        return "Low"
    if "p5" in label or "lowest" in label or "trivial" in label:
        return "Lowest"
    return "Medium"


@lru_cache(maxsize=3)
def load_artifact(filename):
    path = DATA / filename
    if not path.exists():
        raise ValueError(f"Model file is missing: {path.name}")
    return joblib.load(path)


@lru_cache(maxsize=3)
def logistic_reference(filename, target):
    """Use only the original train split, never validation/test rows."""
    dataset_path = DATA / "training_dataset_v6_1.csv"
    split_path = DATA / "source_aware_splits_v6_1_fixed.csv"
    if not dataset_path.exists() or not split_path.exists():
        raise ValueError("Logistic explanations need training_dataset_v6_1.csv and source_aware_splits_v6_1_fixed.csv.")
    artifact = load_artifact(filename)
    cols = artifact["numeric_features"] + artifact["categorical_features"]
    dataset = pd.read_csv(dataset_path, usecols=list(dict.fromkeys(["source_collection", "issue_key"] + cols)), low_memory=False)
    assignments = pd.read_csv(split_path)
    assignments = assignments.loc[(assignments["target"] == target) & (assignments["split"] == "train"), ["source_collection", "issue_key"]]
    train = dataset.merge(assignments, on=["source_collection", "issue_key"], validate="one_to_one")
    if train.empty or len(train) != len(assignments):
        raise ValueError("The original training data does not match the saved split file.")
    matrix = artifact["selector"].transform(artifact["processor"].transform(train[cols]))
    return np.asarray(matrix).mean(axis=0)


def display_name(name):
    prefix, _, field = str(name).partition("__")
    return (field or prefix).replace("_", " ")


def explain_task(frame, filename, target):
    artifact = load_artifact(filename)
    cols = artifact["numeric_features"] + artifact["categorical_features"]
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError("Candidate input is missing model features: " + ", ".join(missing))
    values = frame[cols].copy()
    for col in artifact["numeric_features"]:
        values[col] = pd.to_numeric(values[col], errors="coerce")
    for col in artifact["categorical_features"]:
        values[col] = values[col].replace("", np.nan)
    transformed = artifact["processor"].transform(values)
    selected = np.asarray(artifact["selector"].transform(transformed))
    model = artifact["model"]
    classes = list(model.classes_)
    positive_index = classes.index(1)
    probabilities = model.predict_proba(selected)[:, positive_index]
    names = artifact["selected_feature_names"]
    if hasattr(model, "coef_"):
        baseline = logistic_reference(filename, target)
        contributions = (selected - baseline) * model.coef_[0]
        baseline_logit = float(model.intercept_[0] + baseline @ model.coef_[0])
        if not np.allclose(baseline_logit + contributions.sum(axis=1), model.decision_function(selected), atol=1e-8):
            raise ValueError("Linear contributions do not reconcile with model predictions.")
        method = "Linear contributions versus training-set average (log-odds)"
    else:
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent", model_output="raw")
        shap_values = explainer.shap_values(selected)
        if isinstance(shap_values, list):
            contributions = np.asarray(shap_values[positive_index])
        else:
            contributions = np.asarray(shap_values)
            if contributions.ndim == 3:
                contributions = contributions[:, :, positive_index]
        if contributions.shape != selected.shape:
            raise ValueError(f"Unexpected SHAP output shape: {contributions.shape}")
        expected = np.asarray(explainer.expected_value).reshape(-1)
        base = expected[positive_index] if len(expected) > 1 else expected[0]
        if not np.allclose(base + contributions.sum(axis=1), probabilities, atol=1e-5):
            raise ValueError("SHAP contributions do not reconcile with model probabilities.")
        method = "Tree SHAP versus the forest training-path reference (probability)"
    explanation_rows = []
    for row_index, contributions_for_row in enumerate(contributions):
        order = np.argsort(np.abs(contributions_for_row))[::-1][:5]
        factors = []
        for i in order:
            if abs(contributions_for_row[i]) <= 1e-12:
                continue
            feature = str(names[i])
            field = feature.split("__", 1)[-1]
            is_indicator = field.startswith("missingindicator_")
            if is_indicator:
                field = field[len("missingindicator_"):]
            if field not in values.columns:
                field = next((c for c in artifact["categorical_features"] if field.startswith(c + "_")), "")
            value = values.iloc[row_index][field] if field else None
            observed = "missing" if value is None or pd.isna(value) else str(value)
            factors.append({"feature": feature, "label": display_name(feature), "observed": observed,
                            "direction": "increases" if contributions_for_row[i] > 0 else "decreases",
                            "contribution": float(contributions_for_row[i])})
        explanation_rows.append({"method": method, "factors": factors})
    return probabilities, explanation_rows


def score_candidates(items):
    if not isinstance(items, list) or not 0 < len(items) <= 100:
        raise ValueError("Upload between 1 and 100 candidate issues.")
    frame = pd.DataFrame(items)
    if not {"issue_key", "source_collection"}.issubset(frame.columns):
        raise ValueError("Candidate CSV must contain issue_key and source_collection.")
    if frame["issue_key"].isna().any() or (frame["issue_key"].astype(str).str.strip() == "").any():
        raise ValueError("Every candidate must have an issue key.")
    if frame.duplicated(["source_collection", "issue_key"]).any():
        raise ValueError("Candidate source and issue-key pairs must be unique.")
    output = []
    for _, row in frame.iterrows():
        output.append({k: v for k, v in row.to_dict().items() if pd.notna(v)})
    for row in output:
        raw_priority = row.get("priority", row.get("priority_at_cutoff", "Medium"))
        row["priority_raw"] = str(raw_priority)
        row["priority"] = normalise_priority(raw_priority)
    for key, (filename, target) in MODELS.items():
        probabilities, explanations = explain_task(frame, filename, target)
        artifact = load_artifact(filename)
        source_index = artifact["categorical_features"].index("source_collection")
        encoder = artifact["processor"].named_transformers_["category"].named_steps["onehot"]
        known_sources = set(map(str, encoder.categories_[source_index]))
        for i, row in enumerate(output):
            row[key] = float(probabilities[i])
            row.setdefault("explanations", {})[key] = explanations[i]
            if str(row.get("source_collection")) not in known_sources:
                row["source_warning"] = "Source was unseen during training; predictions are exploratory and may not generalize."
    # Resolution Risk is deliberately scored by the saved multimodal neural
    # model, replacing the prior Random Forest dashboard scorer.
    from dashboard.backend.neural_resolution import (
        score_resolution_neural,
    )
    neural_scores, neural_explanations = score_resolution_neural(frame)
    for i, row in enumerate(output):
        row["resolution_risk"] = float(neural_scores[i])
        row.setdefault("explanations", {})["resolution_risk"] = neural_explanations[i]
        row["resolution_model"] = "Multimodal neural model (v6.1)"
    # The dashboard presents the neural model's Resolution Risk score and
    # branch-ablation explanation.  Optimisation deliberately uses a separate
    # calibrated Random Forest probability: it was the stronger model in the
    # retrospective portfolio-selection study.
    rf_probability, _ = explain_task(frame, *RESOLUTION_RF)
    calibrator = load_artifact(RESOLUTION_CALIBRATOR)
    calibrated_probability = calibrator.predict_proba(np.asarray(rf_probability).reshape(-1, 1))[:, 1]
    for i, row in enumerate(output):
        row["resolution_optimization_probability"] = float(calibrated_probability[i])
        row["resolution_optimization_model"] = "Validation-calibrated Random Forest (v6.2)"
    for row in output:
        combined = .25 * row["volatility_risk"] + .50 * row["resolution_risk"] + .25 * row["reopen_risk"]
        row["combined_risk"] = combined
        row["risk_band"] = "High" if combined > .70 else "Medium" if combined > .40 else "Low"
        if row["resolution_risk"] >= .70 or combined >= .70:
            row["planner_action"], row["action_reason"] = "Defer", "Resolution risk or combined risk meets the defer rule."
        elif row["volatility_risk"] >= .60:
            row["planner_action"], row["action_reason"] = "Split", "Volatility risk meets the split rule."
        elif row["reopen_risk"] >= .45:
            row["planner_action"], row["action_reason"] = "Monitor", "Reopen risk meets the monitoring rule."
        else:
            row["planner_action"], row["action_reason"] = "Include", "None of the current action-rule thresholds was met."
    return output
