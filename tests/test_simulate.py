from collections import Counter
import random
import unittest
from unittest.mock import patch

from src.president import Round
from src.simulate import PLAYERS, format_report, placement_counts, role_assignments, simulate


class SimulationTests(unittest.TestCase):
    def test_1000_assignments_balance_every_role_across_seats(self):
        assignments = list(role_assignments(1000, random.Random(42)))
        self.assertEqual(len(assignments), 1000)
        self.assertGreater(len({tuple(a.values()) for a in assignments}), 4)
        for assignment in assignments:
            self.assertEqual(set(assignment.values()), {1, 2, 3, 4})
        for player in PLAYERS:
            self.assertEqual(Counter(a[player] for a in assignments),
                             {1: 250, 2: 250, 3: 250, 4: 250})
        self.assertEqual(len(list(role_assignments(7, random.Random(42)))), 7)

    def test_counts_compare_starting_role_with_result_not_player_seat(self):
        records = [{"starting_roles": {"a": 4, "b": 1, "c": 3, "d": 2},
                    "placements": {"a": 1, "b": 2, "c": 3, "d": 4}}]
        self.assertEqual(placement_counts(records),
                         [[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]])

    def test_trials_are_reproducible_and_start_with_assigned_later_round_roles(self):
        with patch("src.simulate.Round", wraps=Round) as constructor:
            result = simulate(games=4, seed=7)
        self.assertEqual(result, simulate(games=4, seed=7))
        self.assertEqual(constructor.call_count, 4)
        for call, record in zip(constructor.call_args_list, result["records"]):
            self.assertFalse(call.kwargs["first_round"])
            self.assertEqual(call.kwargs["roles"], record["starting_roles"])
            self.assertEqual(set(record["placements"].values()), {1, 2, 3, 4})
        counts = placement_counts(result["records"])
        self.assertEqual([sum(row) for row in counts], [4] * 4)
        self.assertEqual([sum(row[i] for row in counts) for i in range(4)], [4] * 4)
        self.assertIn("Mean place", format_report(result))

    def test_invalid_sizes_and_unfinished_trials_fail_explicitly(self):
        for options in ({"games": 0}, {"games": -1}, {"max_actions": 0}):
            with self.assertRaises(ValueError):
                simulate(**options)
        with self.assertRaisesRegex(RuntimeError, "trial 1 reached the action limit"):
            simulate(games=1, max_actions=1)
