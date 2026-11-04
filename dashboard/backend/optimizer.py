"""Capacity-constrained sprint portfolio optimisation using calibrated expected utility.

The optimiser uses the calibrated Random Forest Resolution-Risk probability,
not the dashboard's neural display score. That choice follows the held-out
portfolio replay, where the Random Forest gave the stronger selection signal.
"""
from __future__ import annotations

import math


# The public Jira data has no true business-value field. This is the explicit,
# monotonic priority proxy used in the retrospective evaluation study.
PRIORITY_VALUE = {"lowest": .10, "low": .25, "medium": .50, "high": .75, "highest": 1.0, "critical": 1.0}
DEFAULT_FAILURE_COST_RATIO = 1.0


def _normalise_priority(value):
    label = str(value or "").strip().lower()
    if "p1" in label or "urgent" in label or "critical" in label or "blocker" in label or "highest" in label:
        return "highest"
    if "p2" in label or "high" in label:
        return "high"
    if "p3" in label or "medium" in label or "normal" in label:
        return "medium"
    if "p4" in label or "low" in label:
        return "low"
    if "p5" in label or "lowest" in label or "trivial" in label:
        return "lowest"
    return "medium"


def _number(value, name, issue_key):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{issue_key}: {name} must be numeric for sprint optimisation.") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{issue_key}: {name} must be a non-negative finite number.")
    return result


def _dependency_keys(row):
    value = row.get("depends_on", row.get("dependency_keys", ""))
    return [key.strip() for key in str(value).replace(";", ",").split(",") if key.strip()]


def optimise_sprint(items, capacity, failure_cost_ratio=DEFAULT_FAILURE_COST_RATIO):
    """Select a portfolio maximising expected delivery utility.

    U_i = V_i(1-r_i) - gamma*V_i*r_i, where V_i is a documented
    priority-as-value proxy and r_i is a validation-calibrated Resolution Risk
    probability. Capacity and named dependencies are hard constraints.
    """
    try:
        import pulp
    except ImportError as exc:
        raise RuntimeError("Sprint optimisation needs PuLP. Install dashboard requirements and restart the server.") from exc
    if not isinstance(items, list) or not items:
        raise ValueError("Optimisation needs at least one scored backlog item.")
    capacity = _number(capacity, "capacity", "Sprint")
    gamma = _number(failure_cost_ratio, "failure_cost_ratio", "Sprint")
    if capacity <= 0:
        raise ValueError("Sprint capacity must be greater than zero.")

    prepared, key_to_index = [], {}
    for index, raw in enumerate(items):
        row = dict(raw)
        key = str(row.get("issue_key", "")).strip()
        if not key or key in key_to_index:
            raise ValueError("Every candidate must have a unique issue_key for optimisation.")
        effort = _number(row.get("story_points", row.get("effort_estimate")), "story_points/effort_estimate", key)
        if effort <= 0:
            raise ValueError(f"{key}: effort must be greater than zero for optimisation.")
        if "resolution_optimization_probability" not in row:
            raise ValueError(
                "This register has no validation-calibrated Random Forest probability. "
                "Upload a candidate CSV so Sprint Intelligence can score it first, then optimise that scored result."
            )
        probability = _number(row["resolution_optimization_probability"], "calibrated Resolution Risk probability", key)
        if probability > 1:
            raise ValueError(f"{key}: calibrated Resolution Risk probability must be between 0 and 1.")
        priority = _normalise_priority(row.get("priority", row.get("priority_at_cutoff", "Medium")))
        value = PRIORITY_VALUE.get(priority, PRIORITY_VALUE["medium"])
        utility = value * (1 - probability) - gamma * value * probability
        prepared.append({
            "row": row, "key": key, "effort": effort, "probability": probability,
            "value": value, "utility": utility, "dependencies": _dependency_keys(row),
        })
        key_to_index[key] = index

    problem = pulp.LpProblem("expected_utility_sprint_portfolio", pulp.LpMaximize)
    decision = [pulp.LpVariable(f"commit_{index}", cat="Binary") for index in range(len(prepared))]
    problem += pulp.lpSum(item["utility"] * decision[index] for index, item in enumerate(prepared))
    problem += pulp.lpSum(item["effort"] * decision[index] for index, item in enumerate(prepared)) <= capacity
    for index, item in enumerate(prepared):
        for dependency in item["dependencies"]:
            problem += decision[index] == 0 if dependency not in key_to_index else decision[index] <= decision[key_to_index[dependency]]

    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"Sprint optimisation did not find an optimal portfolio: {pulp.LpStatus[status]}.")
    selected = [item for index, item in enumerate(prepared) if pulp.value(decision[index]) > .5]
    selected_keys = {item["key"] for item in selected}
    output = []
    for item in prepared:
        row, included = item["row"], item["key"] in selected_keys
        probability, value, utility = item["probability"], item["value"], item["utility"]
        row["optimizer_decision"] = "Commit" if included else "Reserve / defer"
        row["optimizer_reason"] = (
            f"Selected: value proxy {value:.2f}, calibrated Resolution Risk {probability:.0%}, "
            f"and expected utility {utility:.3f} fit the {capacity:g}-point portfolio."
            if included else
            f"Reserved: calibrated Resolution Risk {probability:.0%} gives expected utility {utility:.3f}; "
            "other feasible items provide more expected delivery value within capacity."
        )
        output.append(row)
    committed_effort = sum(item["effort"] for item in selected)
    return {"items": output, "optimization": {
        "method": "0-1 integer linear programming with calibrated expected utility (PuLP/CBC)",
        "capacity": capacity,
        "committed_effort": round(committed_effort, 2),
        "remaining_capacity": round(capacity - committed_effort, 2),
        "selected_issue_keys": [item["key"] for item in selected],
        "deferred_issue_keys": [item["key"] for item in prepared if item["key"] not in selected_keys],
        "objective_value": round(float(pulp.value(problem.objective)), 3),
        "failure_cost_ratio_gamma": gamma,
        "risk_probability_source": "Validation-calibrated Random Forest Resolution Risk (v6.2)",
        "assumptions": "U = V(1-r) - gamma*V*r. Priority is a documented proxy for business value; gamma is the failure-cost/value ratio. Named dependencies are hard constraints.",
        "limitation": "This is retrospective decision support. It does not prove that selecting an item changes its outcome.",
    }}
