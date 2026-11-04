"""Integration check against the locally saved trusted models and real sample."""
import io
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from dashboard.backend.explain import score_candidates
from dashboard.backend.app import Handler


class CandidateScoringTests(unittest.TestCase):
    def test_sample_has_scores_and_local_explanations(self):
        path = Path(__file__).resolve().parents[1] / "data/processed/candidate_sprint_items.csv"
        rows = pd.read_csv(path).head(2).fillna("").to_dict(orient="records")
        results = score_candidates(rows)
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIn(result["planner_action"], {"Include", "Monitor", "Split", "Defer"})
            for key in ("volatility_risk", "resolution_risk", "reopen_risk"):
                self.assertGreaterEqual(result[key], 0)
                self.assertLessEqual(result[key], 1)
                self.assertTrue(result["explanations"][key]["factors"])
                self.assertIn("method", result["explanations"][key])
            if result["source_collection"] == "Apache":
                self.assertIn("unseen", result["source_warning"])

    def test_duplicate_issue_is_rejected(self):
        row = {"source_collection": "Apache", "issue_key": "A-1"}
        with self.assertRaisesRegex(ValueError, "unique"):
            score_candidates([row, row])

    def test_candidate_api_routes_real_model_output(self):
        path = Path(__file__).resolve().parents[1] / "data/processed/candidate_sprint_items.csv"
        rows = pd.read_csv(path).head(1).fillna("").to_dict(orient="records")
        body = json.dumps({"items": rows}).encode()
        fake = SimpleNamespace(path="/api/score-candidates", headers={"Content-Length": str(len(body))},
                               rfile=io.BytesIO(body), reply=lambda status, payload: (status, payload))
        status, payload = Handler.do_POST(fake)
        self.assertEqual(status, 200)
        self.assertTrue(payload["items"][0]["explanations"]["resolution_risk"]["factors"])


if __name__ == "__main__":
    unittest.main()
