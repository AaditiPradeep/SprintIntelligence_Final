"""Score the explanation demo once, so its action mix can be audited."""
from pathlib import Path

import pandas as pd

from dashboard.backend.explain import score_candidates


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"


def main():
    candidates = pd.read_csv(DATA / "explanation_dependency_demo_candidates_v6_2.csv").fillna("")
    scored = pd.DataFrame(score_candidates(candidates.to_dict("records")))
    scored.to_csv(DATA / "explanation_dependency_demo_scored_v6_2.csv", index=False)
    print(scored["planner_action"].value_counts().to_dict())


if __name__ == "__main__":
    main()
