from concurrent.futures import ProcessPoolExecutor
import unittest

from src import agent_search_c
from src.agent import ExperiencedAgent4
import tune_search as ts


class ParseTests(unittest.TestCase):
    def test_values_and_defaults(self):
        v = ts.parse_variant("a:search_worlds=8,search_depth=none,search_through_own_turns=true,"
                             "search_leaf_scale=0.5,search_backend=python,NEXT_FINISH_PENALTY=0")
        self.assertEqual(v["kwargs"], {"search_worlds": 8, "search_depth": None,
                                       "search_through_own_turns": True, "search_leaf_scale": 0.5,
                                       "search_backend": "python", "search_time_ms": None})
        self.assertEqual(v["weights"], {"NEXT_FINISH_PENALTY": 0})

    def test_explicit_time_and_danger(self):
        v = ts.parse_variant("x:search_time_ms=50,danger=0")
        self.assertEqual(v["kwargs"]["search_time_ms"], 50)
        self.assertEqual(v["weights"], {w: 0 for w in ts.DANGER_WEIGHTS})

    def test_bad_specs(self):
        for bad in (":a=1", "x:oops", "x:=1"):
            with self.assertRaises(ValueError):
                ts.parse_variant(bad)


class ClassTests(unittest.TestCase):
    def test_subclass(self):
        cls = ts.build_agent_class(ts.parse_variant("a:search_worlds=3,danger=0"))
        agent = cls()
        self.assertIsInstance(agent, ExperiencedAgent4)
        self.assertEqual(agent.search_worlds, 3)
        self.assertIsNone(agent.search_time_ms)
        self.assertEqual(agent.NEXT_FINISH_PENALTY, 0)
        self.assertEqual(ExperiencedAgent4.NEXT_FINISH_PENALTY, 6.0)
        self.assertTrue(agent_search_c.supported(agent))

    def test_rejections(self):
        with self.assertRaises(ValueError):
            ts.check_variant(ts.parse_variant("a:no_such_option=1"))
        with self.assertRaises(ValueError):
            ts.build_agent_class(ts.parse_variant("a:NO_SUCH_WEIGHT=1"))


class EndToEndTests(unittest.TestCase):
    def test_tiny_run(self):
        variant = ts.parse_variant("tiny:search_worlds=2")
        with ProcessPoolExecutor(max_workers=2) as pool:
            result = ts.run_variant(variant, "src.agent:ExperiencedAgent3Tuned", ts.make_deals(1, 1), pool)
        self.assertEqual(result["summary"]["games"], 6)
        self.assertEqual(sum(result["summary"]["placement_counts"].values()), 12)
        self.assertGreater(result["search"]["search_decisions"], 0)
        self.assertIn("tiny", ts.format_table([result]))


if __name__ == "__main__":
    unittest.main()
