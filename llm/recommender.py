"""Sprint Intelligence — grounded local Ollama recommendation layer.

The predictive models calculate risk scores. Qwen selects approved strategy
IDs and can draft issue-specific implementation guidance around them. The
deterministic planner remains responsible for Include/Split/Defer/Monitor.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .approved_strategies import get_default_strategy, get_strategies, risk_level_for_probability

DEFAULT_MODEL = "qwen3:1.7b"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
VALID_TASKS = {"requirement_volatility", "resolution_risk", "reopen_risk"}
VALID_ACTIONS = {"Include", "Split", "Defer", "Monitor"}

SYSTEM_PROMPT = """You are the recommendation layer of an Agile Sprint Planning tool.
Risk probabilities and top factors were computed by a separate predictive model.
Do not predict, change, or challenge them. Do not change the planner action.
Select one or two IDs only from the approved list. Then write 2-3 concise,
issue-specific steps which operationalise those selected strategies using only
the supplied issue summary and model-factor context. Do not claim an action was
already performed. Do not invent facts, people, dates, tools, or dependencies.
Return exactly one JSON object."""

USER_TEMPLATE = """Issue: {issue_key}
Issue summary: {issue_summary}
Risk type: {risk_type}
Risk probability: {risk_probability:.0%}
Risk level: {risk_level}
Planner action: {planner_action}
Model explanation context (an association, not a proven cause): {factors_block}

Approved strategies:
{strategies_block}

Return JSON only."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "selected_strategy_ids": {
            "type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 2,
        },
        "rationale": {"type": "string"},
        "tailored_guidance": {"type": "string"},
    },
    "required": ["selected_strategy_ids", "rationale", "tailored_guidance"],
    "additionalProperties": False,
}


@dataclass
class RiskInput:
    issue_key: str
    risk_type: str
    risk_probability: float
    top_factors: list[str]
    planner_action: str
    issue_summary: str = ""
    risk_level: str = field(default="")

    def __post_init__(self):
        if self.risk_type not in VALID_TASKS:
            raise ValueError(f"Unsupported risk type: {self.risk_type}")
        if not 0 <= self.risk_probability <= 1:
            raise ValueError("risk_probability must be between 0 and 1.")
        if self.planner_action not in VALID_ACTIONS:
            raise ValueError(f"Unsupported planner action: {self.planner_action}")
        if not self.risk_level:
            self.risk_level = risk_level_for_probability(self.risk_probability)


class MitigationRecommender:
    """Call a locally running Ollama Qwen model without API keys or SDKs."""

    def __init__(self, model_name: str | None = None, base_url: str | None = None):
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)).rstrip("/")

    @staticmethod
    def _safe_factor(factor: str) -> str:
        return re.sub(r"\s+", " ", str(factor)).strip().replace("{", "(").replace("}", ")")[:160]

    def _build_user_message(self, risk: RiskInput) -> tuple[str, set[str]]:
        approved = [s for s in get_strategies(risk.risk_type)
                    if (s["id"] != "increase_monitoring" or risk.planner_action in {"Include", "Monitor"})
                    and (s["id"] != "defer_for_refinement" or risk.planner_action == "Defer")]
        allowed_ids = {strategy["id"] for strategy in approved}
        factors = "; ".join(self._safe_factor(factor) for factor in risk.top_factors[:5]) or "No factor explanation available"
        strategies = "\n".join(f'- {item["id"]}: {item["text"]}' for item in approved)
        return USER_TEMPLATE.format(
            issue_key=risk.issue_key, risk_type=risk.risk_type,
            issue_summary=self._safe_factor(risk.issue_summary) or "No summary supplied",
            risk_probability=risk.risk_probability, risk_level=risk.risk_level,
            planner_action=risk.planner_action, factors_block=factors, strategies_block=strategies,
        ), allowed_ids

    def _chat(self, user_message: str) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "format": RESPONSE_SCHEMA,
            "options": {"temperature": 0},
        }
        request = Request(
            f"{self.base_url}/api/chat", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"Ollama could not run '{self.model_name}': {detail or exc.reason}") from exc
        except URLError as exc:
            raise RuntimeError(
                f"Ollama is not reachable at {self.base_url}. Start Ollama, then pull '{self.model_name}'."
            ) from exc
        try:
            return str(body["message"]["content"]).strip()
        except (KeyError, TypeError) as exc:
            raise RuntimeError("Ollama returned an unexpected chat response.") from exc

    @staticmethod
    def _parse_json(text: str) -> dict:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    def generate(self, risk: RiskInput) -> dict:
        user_message, allowed_ids = self._build_user_message(risk)
        failure = ""
        try:
            parsed = self._parse_json(self._chat(user_message))
        except (RuntimeError, OSError, TimeoutError) as exc:
            parsed = {}
            failure = f"LLM unavailable: {exc}"
        requested_ids = parsed.get("selected_strategy_ids", [])
        if not isinstance(requested_ids, list) or not 1 <= len(requested_ids) <= 2 or any(not isinstance(item, str) or item not in allowed_ids for item in requested_ids):
            requested_ids = []
        selected_ids = list(dict.fromkeys(requested_ids))
        fallback_used = not selected_ids
        if fallback_used:
            selected_ids = [get_default_strategy(risk.risk_type)["id"]]
            failure = failure or "The model response did not contain valid approved strategy IDs."

        approved_by_id = {strategy["id"]: strategy for strategy in get_strategies(risk.risk_type)}
        # Strategy IDs are validated. Free-form guidance is visibly labelled as
        # generated guidance because ID validation cannot fact-check its wording.
        rationale = ("Default approved strategy; model selection was unavailable."
                     if fallback_used else "The model selected these approved strategies; review their suitability before acting.")
        # This is constructed from model-provided facts, not invented by the
        # LLM. It gives the dashboard a per-issue explanation of why the
        # selected mitigation is being surfaced.
        primary_signal = self._safe_factor(risk.top_factors[0]) if risk.top_factors else "no local explanation was available"
        contextual_rationale = (
            f"{risk.risk_type.replace('_', ' ').title()} is {risk.risk_level.lower()} "
            f"({risk.risk_probability:.0%}) for {risk.issue_key}. "
            f"Strongest available model signal: {primary_signal}."
        )
        tailored_guidance = str(parsed.get("tailored_guidance", "")).strip()
        if fallback_used or not tailored_guidance:
            tailored_guidance = (
                "No tailored LLM guidance is available. Review the approved strategy "
                "and the model factors with the delivery team before acting."
            )
        tailored_guidance = re.sub(r"\s+", " ", tailored_guidance)[:700]
        return {
            "issue_key": risk.issue_key, "risk_type": risk.risk_type,
            "risk_probability": risk.risk_probability, "risk_level": risk.risk_level,
            "planner_action": risk.planner_action, "selected_strategy_ids": selected_ids,
            "approved_recommendations": [approved_by_id[strategy_id]["text"] for strategy_id in selected_ids],
            "rationale": rationale,
            "contextual_rationale": contextual_rationale,
            "tailored_guidance": tailored_guidance,
            "tailored_guidance_status": "LLM-generated guidance; review before acting" if not fallback_used else "No tailored guidance generated",
            "grounding_status": "fallback_approved_strategy" if fallback_used else "validated_approved_strategy_ids",
            "fallback_used": fallback_used,
            "fallback_reason": failure,
        }
