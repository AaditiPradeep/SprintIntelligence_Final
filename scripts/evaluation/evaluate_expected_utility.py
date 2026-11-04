"""Retrospective pseudo-sprint evaluation of calibrated expected-utility ILP.

This is selection-quality replay on the untouched source-aware test partition;
it is not evidence that an intervention caused future outcomes.
"""
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pulp
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
GAMMA = 1.0  # Failure costs the same value as successful delivery.
WINDOW_DAYS = 60
PRIORITY_VALUE = {"lowest": .10, "low": .25, "medium": .50, "high": .75, "highest": 1.0, "critical": 1.0}


def solve_ilp(frame, utility, capacity):
    problem = pulp.LpProblem("expected_utility_sprint", pulp.LpMaximize)
    variables = [pulp.LpVariable(f"x_{index}", cat="Binary") for index in range(len(frame))]
    problem += pulp.lpSum(float(utility[index]) * variables[index] for index in range(len(frame)))
    problem += pulp.lpSum(float(frame.iloc[index].story_points_at_cutoff) * variables[index] for index in range(len(frame))) <= capacity
    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"ILP status: {pulp.LpStatus[status]}")
    return np.array([pulp.value(variable) > .5 for variable in variables])


def greedy(frame, score, capacity, seed=None):
    order = np.arange(len(frame)) if seed is None else np.random.default_rng(seed).permutation(len(frame))
    if seed is None:
        order = sorted(order, key=lambda index: (-float(score[index]), float(frame.iloc[index].story_points_at_cutoff)))
    selected, used = np.zeros(len(frame), dtype=bool), 0.0
    for index in order:
        effort = float(frame.iloc[index].story_points_at_cutoff)
        if used + effort <= capacity:
            selected[index] = True
            used += effort
    return selected


def realised(frame, selected):
    committed = frame.loc[selected]
    effort = float(committed.story_points_at_cutoff.sum())
    delivered = float((committed.business_value * (1 - committed.issue_resolution_risk_label)).sum())
    failed = int(committed.issue_resolution_risk_label.sum())
    wasted = float((committed.story_points_at_cutoff * committed.issue_resolution_risk_label).sum())
    return {"committed_items": int(selected.sum()), "committed_effort": effort, "realized_value": delivered,
            "failed_commitments": failed, "failed_commitment_share": float(failed / selected.sum()) if selected.any() else 0.0,
            "wasted_effort": wasted, "value_per_effort": float(delivered / effort) if effort else 0.0}


def bootstrap_difference(per_sprint, method, baseline, seed=42, draws=2000):
    values = per_sprint[method].to_numpy() - per_sprint[baseline].to_numpy()
    rng = np.random.default_rng(seed)
    samples = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(draws)]
    return {"mean_difference": round(float(values.mean()), 5), "ci_95": [round(float(np.quantile(samples, .025)), 5), round(float(np.quantile(samples, .975)), 5)]}


def main(gamma=GAMMA, prediction_file="resolution_test_predictions_calibrated_v6_2.csv"):
    base = pd.read_csv(DATA / "training_dataset_v6_2_effort.csv", low_memory=False)
    prediction_path = Path(prediction_file)
    if not prediction_path.is_absolute():
        prediction_path = DATA / prediction_path
    prediction = pd.read_csv(prediction_path)
    probability_columns = ["calibrated_resolution_probability", "calibrated_neural_resolution_probability"]
    probability_column = next((column for column in probability_columns if column in prediction.columns), None)
    if probability_column is not None and probability_column != "calibrated_resolution_probability":
        prediction = prediction.rename(columns={probability_column: "calibrated_resolution_probability"})
    required_prediction_columns = {"source_collection", "issue_key", "issue_resolution_risk_label", "calibrated_resolution_probability"}
    missing_prediction_columns = required_prediction_columns.difference(prediction.columns)
    if missing_prediction_columns:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing_prediction_columns)}")
    columns = ["source_collection", "issue_key", "created_at", "priority_at_cutoff", "story_points_at_cutoff", "issue_resolution_risk_label"]
    frame = prediction.merge(base[columns], on=["source_collection", "issue_key", "issue_resolution_risk_label", "story_points_at_cutoff"], validate="one_to_one")
    frame = frame.dropna(subset=["story_points_at_cutoff", "calibrated_resolution_probability"]).copy()
    frame = frame.loc[pd.to_numeric(frame["story_points_at_cutoff"], errors="coerce") > 0].copy()
    frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True, format="mixed")
    frame["business_value"] = frame.priority_at_cutoff.astype(str).str.lower().map(PRIORITY_VALUE).fillna(PRIORITY_VALUE["medium"])
    frame["expected_utility"] = frame.business_value * (1 - frame.calibrated_resolution_probability) - gamma * frame.business_value * frame.calibrated_resolution_probability
    groups = []
    for source, source_frame in frame.groupby("source_collection"):
        source_frame = source_frame.copy()
        origin = source_frame.created_at.min()
        source_frame["pseudo_sprint"] = ((source_frame.created_at - origin).dt.days // WINDOW_DAYS).astype(int)
        totals = source_frame.groupby("pseudo_sprint").story_points_at_cutoff.sum()
        capacity = float(totals.median())
        for bucket, sprint in source_frame.groupby("pseudo_sprint"):
            if len(sprint) >= 3 and sprint.story_points_at_cutoff.sum() >= capacity:
                groups.append((f"{source}-{bucket}", sprint.reset_index(drop=True), capacity))
    if len(groups) < 5:
        raise RuntimeError(f"Only {len(groups)} pseudo-sprints were eligible; need at least 5.")

    records = []
    for sprint_id, sprint, capacity in groups:
        utility = sprint.expected_utility.to_numpy()
        value = sprint.business_value.to_numpy()
        effort = sprint.story_points_at_cutoff.to_numpy()
        strategies = {
            "expected_utility_ilp": solve_ilp(sprint, utility, capacity),
            "priority_only_greedy": greedy(sprint, value, capacity),
            "value_per_effort_greedy": greedy(sprint, value / effort, capacity),
            "rules_plus_greedy": greedy(sprint, np.where(sprint.calibrated_resolution_probability.to_numpy() < .5, value / effort, -1), capacity),
            "no_risk_ilp_ablation": solve_ilp(sprint, value, capacity),
            "oracle_ilp": solve_ilp(sprint, value * (1 - sprint.issue_resolution_risk_label.to_numpy()) - gamma * value * sprint.issue_resolution_risk_label.to_numpy(), capacity),
        }
        random_metrics = [realised(sprint, greedy(sprint, np.zeros(len(sprint)), capacity, seed=1000 + iteration)) for iteration in range(100)]
        for name, selected in strategies.items():
            records.append({"pseudo_sprint": sprint_id, "method": name, "capacity": capacity, **realised(sprint, selected)})
        records.append({"pseudo_sprint": sprint_id, "method": "random_selection_mean", "capacity": capacity,
                        **{key: float(np.mean([metric[key] for metric in random_metrics])) for key in random_metrics[0]}})
    results = pd.DataFrame(records)
    gamma_tag = str(gamma).replace(".", "_")
    model_tag = prediction_path.stem.replace("resolution_test_predictions_calibrated_v6_2", "random_forest").replace("resolution_neural_test_predictions_calibrated_v6_2", "neural")
    results.to_csv(DATA / f"expected_utility_pseudosprint_results_v6_2_{model_tag}_gamma_{gamma_tag}.csv", index=False)
    value_table = results.pivot(index="pseudo_sprint", columns="method", values="realized_value")
    report = {
        "evaluation_population": {"effort_qualified_test_records": int(len(frame)), "pseudo_sprints": int(len(groups)), "window_days": WINDOW_DAYS},
        "prediction_file": prediction_path.name,
        "expected_utility": f"V_i(1-r_i) - gamma*V_i*r_i, with gamma={gamma} and calibrated Resolution Risk probability r_i.",
        "failure_cost_ratio_gamma": gamma,
        "priority_proxy": PRIORITY_VALUE,
        "aggregate_results": results.groupby("method")[["realized_value", "failed_commitment_share", "wasted_effort", "value_per_effort"]].mean().round(5).to_dict(orient="index"),
        "expected_utility_vs_priority_only": bootstrap_difference(value_table, "expected_utility_ilp", "priority_only_greedy"),
        "expected_utility_vs_no_risk_ablation": bootstrap_difference(value_table, "expected_utility_ilp", "no_risk_ilp_ablation"),
        "wilcoxon_expected_utility_vs_priority_only": float(wilcoxon(value_table["expected_utility_ilp"], value_table["priority_only_greedy"], zero_method="wilcox", alternative="two-sided").pvalue),
        "limitations": ["Retrospective replay measures selection quality, not intervention effects.", "Priority is a business-value proxy.", "The test population has no cutoff-safe dependency graph, so dependency cost is not evaluated here."],
    }
    (DATA / f"expected_utility_evaluation_report_v6_2_{model_tag}_gamma_{gamma_tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=GAMMA, help="Failure-cost/value ratio; must be non-negative.")
    parser.add_argument("--prediction-file", default="resolution_test_predictions_calibrated_v6_2.csv", help="Calibrated test-prediction CSV, relative to data/processed or an absolute path.")
    args = parser.parse_args()
    if args.gamma < 0:
        raise ValueError("gamma must be non-negative")
    main(args.gamma, args.prediction_file)
