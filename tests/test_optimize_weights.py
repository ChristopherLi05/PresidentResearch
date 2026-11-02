import contextlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import optimize_weights as optimizer
from src.agent import ExperiencedAgent2, ExperiencedAgent4


class OptimizeWeightsTests(unittest.TestCase):
    def test_agent4_exposes_turn_order_weights(self):
        bounds = optimizer.bounds_for(ExperiencedAgent4)
        for name in ("NEXT_RESPONSE_PENALTY", "NEXT_FINISH_PENALTY",
                     "NEXT_COMPLETION_PENALTY", "NEXT_BOMB_PENALTY"):
            self.assertIn(name, bounds)
        weights = {name: getattr(ExperiencedAgent4, name) for name in bounds}
        candidate = optimizer.make_agent(ExperiencedAgent4, weights)
        self.assertEqual(candidate.NEXT_BOMB_PENALTY, ExperiencedAgent4.NEXT_BOMB_PENALTY)

    def test_candidates_do_not_mutate_defaults_or_share_cached_costs(self):
        defaults = {name: getattr(ExperiencedAgent2, name) for name in optimizer.BOUNDS}
        changed = dict(defaults, GROUP_WEIGHT=defaults["GROUP_WEIGHT"] + 1)
        first = optimizer.make_agent(ExperiencedAgent2, defaults)()
        second = optimizer.make_agent(ExperiencedAgent2, changed)()
        inputs = ((("3", 1),), (0,) * 13, ())
        self.assertAlmostEqual(second._hand_cost_cached(*inputs) - first._hand_cost_cached(*inputs), 1)
        self.assertEqual(ExperiencedAgent2.GROUP_WEIGHT, defaults["GROUP_WEIGHT"])
        first._hand_cost_cached.cache_clear()

    def test_proposals_are_reproducible_and_bounded(self):
        weights = {name: getattr(ExperiencedAgent2, name) for name in optimizer.BOUNDS}
        self.assertEqual(optimizer.propose(weights, random.Random(4), .2),
                         optimizer.propose(weights, random.Random(4), .2))
        rng = random.Random(7)
        for _ in range(100):
            weights = optimizer.propose(weights, rng, .2)
            optimizer.make_agent(ExperiencedAgent2, weights)

    def test_search_selection_validation_and_export(self):
        calls = []
        scores = iter([2.5, 2.4, 2.6, 2.5, 2.45])

        def evaluate(base, weights, opponents, trials, seed, max_actions, progress=None):
            calls.append((dict(weights), trials, seed))
            return {"mean_placement": next(scores), "by_opponent": {}}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "weights.json"
            with patch.object(optimizer, "evaluate", side_effect=evaluate), contextlib.redirect_stdout(io.StringIO()):
                optimizer.main(["--method", "hill", "--agent", "ExperiencedAgent2", "--iterations", "2",
                                "--trials", "1", "--validation-trials", "2", "--output", str(output)])
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["best_weights"], calls[1][0])
            self.assertEqual(calls[4][0], calls[1][0])
            self.assertEqual(len({call[2] for call in calls[:3]}), 1)
            self.assertEqual(calls[3][2], calls[4][2])
            self.assertNotEqual(calls[0][2], calls[3][2])
            namespace = {}
            exec(output.with_suffix(".py").read_text(), namespace)
            self.assertEqual(namespace["OptimizedAgent"].GROUP_WEIGHT, report["best_weights"]["GROUP_WEIGHT"])

    def test_es_moves_toward_lower_placement_and_cancels_ties(self):
        scores = [({"mean_placement": 2.0}, {"mean_placement": 3.0})]
        updated = optimizer.es_update([0.5, 0.5], [[1.0, -1.0]], scores, .1, .02)
        self.assertAlmostEqual(updated[0], .6)
        self.assertAlmostEqual(updated[1], .4)
        tied = [({"mean_placement": 2.5}, {"mean_placement": 2.5})]
        self.assertEqual(optimizer.es_update([.5], [[1.0]], tied, .1, .02), [.5])

    def test_es_uses_fresh_paired_deals_and_selects_only_on_benchmark(self):
        calls = []
        # initial benchmark, pair, benchmark, pair, benchmark, final validation
        scores = iter([2.5, 2.0, 3.0, 2.4, 1.0, 4.0, 2.6, 2.5, 2.45])

        def evaluate(base, weights, opponents, trials, seed, max_actions, progress=None):
            calls.append((dict(weights), trials, seed))
            return {"mean_placement": next(scores), "by_opponent": {}}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "es.json"
            with patch.object(optimizer, "evaluate", side_effect=evaluate), contextlib.redirect_stdout(io.StringIO()):
                optimizer.main(["--iterations", "2", "--pairs", "1", "--trials", "1",
                                "--benchmark-trials", "2", "--validation-trials", "3", "--output", str(output)])
            report = json.loads(output.read_text())
            self.assertEqual(report["best_weights"], calls[3][0])
            self.assertEqual(calls[8][0], calls[3][0])
            self.assertEqual(calls[1][2], calls[2][2])
            self.assertEqual(calls[4][2], calls[5][2])
            self.assertEqual(calls[0][2], calls[3][2])
            self.assertEqual(calls[3][2], calls[6][2])
            self.assertEqual(calls[7][2], calls[8][2])
            self.assertEqual(len({calls[i][2] for i in (0, 1, 4, 7)}), 4)
            # Mirrored candidates, allowing clipping at the configured bounds.
            center = optimizer.normalized_weights(calls[0][0])
            direction = report["history"][1]["directions"][0]
            for index, sign in ((1, 1), (2, -1)):
                expected = optimizer.decode_weights([x + sign * .1 * e for x, e in zip(center, direction)])
                self.assertEqual(calls[index][0], expected)

    def test_es_resume_continues_generation_and_random_state(self):
        calls = []

        def evaluate(base, weights, opponents, trials, seed, max_actions, progress=None):
            calls.append((dict(weights), trials, seed))
            return {"mean_placement": 2.5, "by_opponent": {}}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "checkpoint.json"
            arguments = ["--agent", "ExperiencedAgent2", "--iterations", "1", "--pairs", "1",
                         "--trials", "1", "--benchmark-trials", "1", "--validation-trials", "1",
                         "--seed", "19", "--output", str(output), "--no-progress"]
            with patch.object(optimizer, "evaluate", side_effect=evaluate), contextlib.redirect_stdout(io.StringIO()):
                optimizer.main(arguments)
                first = json.loads(output.read_text())
                optimizer.main(["--resume", str(output), "--iterations", "1", "--output", str(output),
                                "--no-progress"])
            resumed = json.loads(output.read_text())
            self.assertEqual(first["next_generation"], 2)
            self.assertEqual(resumed["next_generation"], 3)
            self.assertEqual([item["iteration"] for item in resumed["history"]], [0, 1, 2])
            self.assertEqual(resumed["agent"], "ExperiencedAgent2")
            self.assertEqual(resumed["configuration"]["pairs"], 1)
            self.assertNotEqual(resumed["history"][1]["training_seed"], resumed["history"][2]["training_seed"])


if __name__ == "__main__":
    unittest.main()
