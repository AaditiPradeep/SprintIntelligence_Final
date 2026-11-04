import unittest
from unittest.mock import patch

from llm.recommender import MitigationRecommender, RiskInput


class MitigationTests(unittest.TestCase):
    def setUp(self):
        self.recommender = MitigationRecommender()
        self.risk = RiskInput("DEMO-1", "resolution_risk", .8, ["Inactivity increases the model score"], "Defer")

    def test_valid_selection_displays_only_canonical_strategy(self):
        response = '{"selected_strategy_ids":["review_dependencies"],"rationale":"Invented unsupported facts"}'
        with patch.object(self.recommender, "_chat", return_value=response):
            result = self.recommender.generate(self.risk)
        self.assertFalse(result["fallback_used"])
        self.assertEqual(result["selected_strategy_ids"], ["review_dependencies"])
        self.assertNotIn("Invented", result["rationale"])
        self.assertEqual(result["grounding_status"], "validated_approved_strategy_ids")
        self.assertIn("DEMO-1", result["contextual_rationale"])
        self.assertIn("80%", result["contextual_rationale"])

    def test_invalid_or_contradictory_selection_falls_back(self):
        for response in ('not json', '{"selected_strategy_ids":["not_approved"]}',
                         '{"selected_strategy_ids":["increase_monitoring"]}'):
            with self.subTest(response=response), patch.object(self.recommender, "_chat", return_value=response):
                result = self.recommender.generate(self.risk)
            self.assertTrue(result["fallback_used"])
            self.assertEqual(result["selected_strategy_ids"], ["review_dependencies"])
            self.assertEqual(result["grounding_status"], "fallback_approved_strategy")

    def test_unavailable_ollama_falls_back(self):
        with patch.object(self.recommender, "_chat", side_effect=RuntimeError("unavailable")):
            result = self.recommender.generate(self.risk)
        self.assertTrue(result["fallback_used"])
        self.assertIn("unavailable", result["fallback_reason"])


if __name__ == "__main__":
    unittest.main()
