import unittest

from dashboard.backend.optimizer import optimise_sprint


class OptimizerTests(unittest.TestCase):
    def test_capacity_and_dependency_constraint(self):
        rows = [
            {"issue_key": "A", "story_points": 5, "priority": "Highest", "resolution_optimization_probability": .2, "has_dependencies": 0},
            {"issue_key": "B", "story_points": 5, "priority": "High", "resolution_optimization_probability": .2, "has_dependencies": 0, "depends_on": "A"},
            {"issue_key": "C", "story_points": 8, "priority": "Low", "resolution_optimization_probability": .8, "has_dependencies": 1},
        ]
        output = optimise_sprint(rows, 10)
        self.assertLessEqual(output["optimization"]["committed_effort"], 10)
        selected = set(output["optimization"]["selected_issue_keys"])
        self.assertFalse("B" in selected and "A" not in selected)

    def test_missing_effort_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "story_points"):
            optimise_sprint([{"issue_key": "A", "combined_risk": .2}], 10)


if __name__ == "__main__":
    unittest.main()
