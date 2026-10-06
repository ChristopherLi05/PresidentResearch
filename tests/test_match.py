import unittest

from src.agent import BasicAgent, BasicAgentPlus
from test import TEAM_CONFIGURATIONS, format_report, run_match


class MatchTests(unittest.TestCase):
    def test_pairings_reuse_deal_seeds_and_report_metrics(self):
        progress = _Progress()
        result = run_match(BasicAgentPlus, BasicAgent, trials_per_configuration=4, seed=7,
                           progress=progress)

        self.assertEqual(result["games"], 24)
        self.assertEqual(progress.completed, 24)
        self.assertEqual({tuple(record["configuration"]) for record in result["records"]},
                         set(TEAM_CONFIGURATIONS))
        for trial in range(1, 5):
            seeds = {record["deck_seed"] for record in result["records"] if record["trial"] == trial}
            self.assertEqual(len(seeds), 1)
        for summary in result["summary"].values():
            self.assertEqual(sum(summary["placement_counts"].values()), 48)
            self.assertEqual(len(summary["mean_placement_95_ci"]), 2)
            self.assertEqual(set(summary["placement_by_initial_rank"]),
                             {"President", "Vice President", "Vice Scum", "Scum"})
        self.assertIn("Final placement by initial rank", format_report(result))

    def test_invalid_trial_limit_is_rejected(self):
        with self.assertRaises(ValueError):
            run_match(BasicAgent, BasicAgentPlus, trials_per_configuration=0)


class _Progress:
    def __init__(self):
        self.completed = 0

    def update(self, amount):
        self.completed += amount
