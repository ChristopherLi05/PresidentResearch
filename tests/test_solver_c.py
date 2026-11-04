"""Native/Python exact-solver parity; build with python -m src.solver_c --build."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import random
import unittest
from unittest.mock import patch

from src.president import Card, Round
from src.solver import solve_round
from src import solver_c
from tests.test_president import ready_round


class NativeSolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if solver_c.library() is None:
            raise unittest.SkipTest("Build native solver with python -m src.solver_c --build")

    def compare(self, position):
        before = deepcopy(position.__dict__)
        expected = solve_round(position, backend="python")
        self.assertEqual(solve_round(position, backend="c"), expected)
        self.assertEqual(position.__dict__, before)
        return expected

    def test_real_rounds_all_decisions_and_table_growth(self):
        checked, largest = 0, 0
        phases, pairs = set(), set()
        for seed in range(50):
            rng = random.Random(seed)
            state = Round(list("abcd"), first_round=True, rng=rng)
            while not state.done:
                if len(state.active) == 2 and sum(len(state.hands[p]) for p in state.active) <= 14:
                    result = self.compare(state)
                    largest = max(largest, result.states_evaluated)
                    phases.add(state.phase); pairs.add(tuple(state._active_in_order())); checked += 1
                actor = state._actor()
                state.apply_action(actor, rng.choice(state.legal_actions(actor)))
        self.assertGreater(checked, 100)
        self.assertGreater(largest, 700)  # exercises memo-table reallocation
        self.assertEqual(phases, {"play", "completion"})
        self.assertEqual(len(pairs), 6)

    def test_inherited_boards_completion_and_bombs(self):
        for last in "abcd":
            state = ready_round({"a": [Card("7", s) for s in "CDH"] + [Card("2", "C"), Card("3", "C")],
                                 "b": [Card("A", "C"), Card("4", "C")]}, top=("7", 1), last=last)
            self.compare(state)
            state._offer_completions("a")
            self.compare(state)
            state.apply_action("a", {"type": "decline_completion"})
            self.compare(state)

    def test_parallel_calls_are_isolated(self):
        state = ready_round({"a": [Card("3", "C"), Card("7", "C"), Card("2", "C")],
                             "b": [Card("4", "C"), Card("8", "C")]})
        expected = self.compare(state)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: solve_round(state, backend="c"), range(40)))
        self.assertTrue(all(result == expected for result in results))

    def test_missing_library_fallback_and_explicit_error(self):
        state = ready_round({"a": [Card("3", "C")], "b": [Card("4", "C")]})
        with patch("src.solver_c.library", return_value=None):
            self.assertEqual(solve_round(state), solve_round(state, backend="python"))
            with self.assertRaises(RuntimeError):
                solve_round(state, backend="c")
        with self.assertRaises(ValueError):
            solve_round(state, backend="invalid")


if __name__ == "__main__":
    unittest.main()
