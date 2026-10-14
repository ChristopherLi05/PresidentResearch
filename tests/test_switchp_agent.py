"""Source-derived evaluation fixtures and President protocol regressions."""
from collections import Counter
from copy import deepcopy
import random
import unittest
from unittest.mock import patch

from arena import client_factory
from src.agent import ExperiencedAgent, ExperiencedAgent2, SwitchPAgent
from src.president import Game
from tests.test_president import PLAYERS, card, ready_round


class SwitchPAgentTests(unittest.TestCase):
    def setUp(self):
        self.agent = SwitchPAgent()

    def test_independent_scoring_and_original_trades(self):
        self.assertIs(client_factory("src.agent:SwitchPAgent"), SwitchPAgent)
        self.assertNotIsInstance(self.agent, ExperiencedAgent2)
        self.assertIs(SwitchPAgent._trade_rank, ExperiencedAgent._trade_rank)
        self.assertIs(SwitchPAgent._trade_return, ExperiencedAgent._trade_return)
        self.assertFalse(hasattr(self.agent, "_hand_cost"))
        self.assertFalse(hasattr(self.agent, "_control"))

    def test_source_single_strength(self):
        self.assertEqual(self.agent._strength("J", 1, Counter(Q=1, K=2), [8, 8, 8]), 40)
        self.assertEqual(self.agent._strength("J", 1, Counter(Q=1, K=2, A=1), [8, 8, 8]), 10)
        self.assertEqual(self.agent._strength("A", 1, Counter(), [8, 8, 8]), 100)

    def test_source_group_penalty_table_and_heads_up(self):
        # p=4, width=2: copies 2/3/4 give k=1/2/3 => penalties 9/15/24.
        for copies, expected in ((2, 91), (3, 85), (4, 76)):
            self.assertEqual(self.agent._strength("J", 2, Counter(Q=copies), [8, 8, 8]), expected)
        self.assertEqual(self.agent._strength("J", 2, Counter(Q=2), [8]), 70)

    def test_president_interruptions_and_clear(self):
        self.assertEqual(self.agent._strength("A", 1, Counter(A=1), [8]), 70)
        self.assertEqual(self.agent._strength("A", 3, Counter(A=1), [8]), 70)
        self.assertEqual(self.agent._strength("A", 1, Counter({"2": 1}), [8]), 70)
        self.assertEqual(self.agent._strength("3", 4, Counter({"2": 4}), [8]), 200)
        self.assertEqual(self.agent._strength("A", 1, Counter(A=1), [8], stack=4), 200)

    def test_source_hand_buckets_and_second_minimum(self):
        pattern = self.agent._pattern(Counter({"3": 1, "J": 1, "A": 1}), Counter(Q=1, K=1), [8, 8, 8])
        self.assertEqual(pattern["z"], 1)
        self.assertEqual(pattern["second"], 40)
        self.assertEqual(self.agent._pattern(Counter(A=1), Counter(), [8])["second"], 201)

    def test_source_near_finish_leads_strong_group_first(self):
        round_ = ready_round({"a": [card("3"), card("A")], "b": [card("9")]})
        with patch.object(SwitchPAgent, "_public_info", return_value=(Counter(K=1), [1])):
            action = self.agent.respond(round_.message_for("a"))
        self.assertEqual(action["cards"][0]["rank"], "A")

    def test_follow_preserves_only_control_with_many_weak_groups(self):
        before = {"count": 5, "z": 7}
        after = {"count": 4, "z": 8, "minimum": 1, "second": 1}
        self.assertFalse(self.agent._accept_follow(before, after, 100, 1)[0])
        self.assertTrue(self.agent._accept_follow(before, after, 200, 4)[0])

    def test_finish_and_bomb_cover(self):
        round_ = ready_round({"a": [card("4"), card("4", "D")], "b": [card("9")]})
        message = round_.message_for("a")
        before = deepcopy(message)
        self.assertEqual(len(self.agent.respond(message)["cards"]), 2)
        self.assertEqual(message, before)
        round_ = ready_round({"a": [card("3"), card("2")], "b": [card("9")]}, top=("A", 1), last="b")
        self.assertEqual(self.agent.respond(round_.message_for("a"))["type"], "bomb")

    def test_completion_does_not_strand_twos(self):
        message = {"hand": [card(r).json() for r in ("7", "2")], "events": [],
                   "request": {"type": "completion", "legal_actions": [
                       {"type": "complete", "cards": [card("7").json()]}, {"type": "decline_completion"}]}}
        self.assertEqual(self.agent.respond(message)["type"], "decline_completion")
        message["hand"] = [card("7").json()]
        self.assertEqual(self.agent.respond(message)["type"], "complete")

    def test_prediction_accuracy_updates_once(self):
        message = {"players": PLAYERS, "events": [
            {"type": "board_cleared", "player": "b"} for _ in range(3)],
            "rankings": {"b": 1, "a": 2, "c": 3, "d": 4}}
        self.assertEqual(self.agent._predict_leader(message), "b")
        self.agent.observe(message)
        self.agent.observe(message)
        self.assertEqual(self.agent._prediction_results["b"], [1, 1])
        self.agent.observe({"rankings": None})
        self.agent.observe(message)
        self.assertEqual(self.agent._prediction_results["b"], [2, 2])

    def test_keep_uses_original_initial_threshold_and_strong_pair_exception(self):
        message = {"player": "a", "players": PLAYERS, "events": [],
                   "hand": [card(r).json() for r in ("3", "4", "5", "6", "7", "8", "9", "K")]}
        unseen, opponents = self.agent._public_info(message)
        action = {"type": "play", "cards": [card("K").json()]}
        self.assertTrue(self.agent._keep(message, action, unseen, opponents))
        # Source threshold is strictly greater than 12.
        with patch.object(SwitchPAgent, "_pattern", return_value={"z": 12}):
            self.assertFalse(self.agent._keep(message, action, unseen, opponents))
        pair = {"type": "play", "cards": [card("A").json(), card("A", "D").json()]}
        self.assertFalse(self.agent._keep(message, pair, unseen, opponents))

    def test_keep_stops_after_a_player_finishes(self):
        message = {"player": "a", "players": PLAYERS,
                   "events": [{"type": "player_done", "player": "b"}],
                   "hand": [card(r).json() for r in ("3", "4", "5", "6", "7", "8", "9", "K")]}
        unseen, opponents = self.agent._public_info(message)
        self.assertFalse(self.agent._keep(message, {"type": "play", "cards": [card("K").json()]},
                                         unseen, opponents))

    def test_mixed_and_self_play_finish_without_e2_scoring(self):
        with patch.object(ExperiencedAgent2, "_choose_play", side_effect=AssertionError("E2 scoring used")):
            for seed in range(6):
                game = Game(PLAYERS, rng=random.Random(seed))
                agents = {p: SwitchPAgent() if seed % 2 == 0 or i % 2 == 0 else ExperiencedAgent()
                          for i, p in enumerate(PLAYERS)}
                for _ in range(3):
                    game.start_round()
                    result = game.run(agents, max_actions=3000)
                    self.assertEqual(set(result.rankings.values()), {1, 2, 3, 4})
