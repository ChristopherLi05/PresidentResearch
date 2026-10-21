"""Tactical and protocol regressions for the second experienced policy."""

from copy import deepcopy
from collections import Counter
import random
import unittest
from unittest.mock import patch

from src.agent import BasicAgent, ExperiencedAgent2
from src.president import Game
from tests.test_president import PLAYERS, card, ready_round


class ExperiencedAgent2Tests(unittest.TestCase):
    def setUp(self):
        self.agent = ExperiencedAgent2()

    def test_probability_cache_reuses_inputs(self):
        self.agent._holding.cache_clear()
        expected = self.agent._holding.__wrapped__(39, 3, 13, 2)
        self.assertEqual(self.agent._holding(39, 3, 13, 2), expected)
        self.assertEqual(self.agent._holding(39, 3, 13, 2), expected)
        self.assertEqual(self.agent._holding.cache_info().hits, 1)

    def test_control_cache_snapshots_mutable_inputs(self):
        self.agent._control_cached.cache_clear()
        unseen = self.agent._unseen({"hand": [], "events": []})
        opponents = [("b", 5)]
        original = self.agent._control("A", 1, unseen, opponents)
        self.assertEqual(self.agent._control("A", 1, unseen.copy(), [("c", 5)]), original)
        self.assertEqual(self.agent._control_cached.cache_info().hits, 1)
        unseen["A"] = unseen["2"] = 0
        self.assertEqual(self.agent._control("A", 1, unseen, opponents), 1.0)
        self.assertLess(original, 1.0)
        opponents.append(("d", 3))
        self.agent._control("A", 1, unseen, opponents)
        self.assertEqual(self.agent._control_cached.cache_info().misses, 3)

    def test_cached_and_uncached_decisions_match(self):
        class UncachedAgent(ExperiencedAgent2):
            _holding = staticmethod(ExperiencedAgent2._holding.__wrapped__)
            _control_cached = classmethod(ExperiencedAgent2._control_cached.__wrapped__)
            _hand_cost_cached = ExperiencedAgent2._hand_cost_cached.__wrapped__
            _lead_cost_cached = ExperiencedAgent2._lead_cost_cached.__wrapped__

        for seed in range(2):
            game = Game(PLAYERS, rng=random.Random(seed))
            cached, uncached = self.agent, UncachedAgent()
            case = self

            class ComparingAgent(ExperiencedAgent2):
                def respond(self, message):
                    action = cached.respond(message)
                    case.assertEqual(action, uncached.respond(message))
                    return action

            agents = {p: ComparingAgent() for p in PLAYERS}
            for _ in range(2):
                game.start_round()
                game.run(agents, max_actions=3000)

    def test_hand_and_lead_caches_snapshot_all_inputs(self):
        for name in ("_hand_cost", "_lead_cost"):
            evaluate = getattr(self.agent, name)
            cached = getattr(self.agent, name + "_cached")
            cached.cache_clear()
            hand = Counter({"3": 1, "A": 2, "2": 1})
            unseen = self.agent._unseen({"hand": [], "events": []})
            opponents = [("b", 5)]
            original = evaluate(hand, unseen, opponents)
            hits = cached.cache_info().hits
            self.assertEqual(evaluate(hand.copy(), unseen.copy(), [("c", 5)]), original)
            self.assertEqual(cached.cache_info().hits, hits + 1)
            for change in (lambda: hand.update({"3": 1}),
                           lambda: unseen.update({"A": -1}),
                           lambda: opponents.append(("d", 3))):
                change()
                misses = cached.cache_info().misses
                actual = evaluate(hand, unseen, opponents)
                self.assertGreater(cached.cache_info().misses, misses)
                expected = cached.__wrapped__(self.agent, tuple(hand.items()),
                    tuple(unseen[r] for r in unseen), tuple(n for _, n in opponents))
                self.assertEqual(actual, expected)

    def test_finish_skips_scoring_but_bomb_does_not(self):
        for kind in ("play", "complete"):
            winning = {"type": kind, "cards": [card("4").json()]}
            message = {"hand": winning["cards"]}
            with patch.object(self.agent, "_unseen", side_effect=AssertionError("scored finish")):
                self.assertIs(self.agent._choose_play(message, [{"type": "pass"}, winning]), winning)
        bomb = {"type": "bomb", "card": card("2").json()}
        passed = {"type": "pass"}
        self.assertIs(self.agent._choose_play({"hand": [bomb["card"]]}, [bomb, passed]), passed)

    def test_trade_returns_score_once_per_rank_and_keep_first_tie(self):
        hand = [card("3").json(), card("3", "D").json(), card("4").json()]
        actions = [{"type": "trade_return", "card": c} for c in hand]
        with patch.object(self.agent, "_trade_value", return_value=1) as score:
            result = self.agent._trade_return({"hand": hand}, actions)
        self.assertIs(result, actions[0])
        self.assertEqual(score.call_count, 2)

    def test_bomb_can_improve_a_hand_without_fully_repairing_cover(self):
        round_ = ready_round({
            "a": [card("3"), card("2", "C"), card("2", "D"), card("2", "H")],
            "b": [card("9")], "c": [card("10")], "d": [card("J")],
        }, top=("A", 1), last="d")
        self.assertEqual(self.agent.respond(round_.message_for("a"))["type"], "bomb")

    def test_public_cards_change_control_estimates(self):
        message = {"hand": [card("10").json()], "events": []}
        unknown = self.agent._unseen(message)
        message["events"] = [{"type": "played", "cards": [card(r, s).json()
                               for r in ("10", "J", "Q", "K", "A", "2")
                               for s in ("C", "D", "H", "S")
                               if (r, s) != ("10", "C")]}]
        exposed = self.agent._unseen(message)
        self.assertLess(self.agent._control("10", 1, unknown, [("b", 5)]), 1)
        self.assertEqual(self.agent._control("10", 1, exposed, [("b", 5)]), 1)

    def test_immediate_finish_and_input_immutability(self):
        round_ = ready_round({"a": [card("4"), card("4", "D")], "b": [card("9")]})
        message = round_.message_for("a")
        before = deepcopy(message)
        action = self.agent.respond(message)
        self.assertEqual(len(action["cards"]), 2)
        self.assertEqual(message, before)

    def test_trade_can_return_a_two_when_it_improves_the_hand(self):
        message = {"hand": [card("3").json(), card("2").json(), card("2", "D").json()],
                   "players": [], "events": []}
        actions = [{"type": "trade_return", "card": c} for c in message["hand"]]
        self.assertEqual(self.agent._trade_return(message, actions)["card"]["rank"], "2")

    def test_failed_and_already_owned_ranks_are_not_requested(self):
        message = {"player": "a", "hand": [card("A", s).json() for s in ("C", "D", "H", "S")],
                   "events": [{"type": "trade_requested", "initiator": "a", "rank": "2", "success": False}],
                   "players": []}
        actions = [{"type": "trade_request", "rank": r} for r in ("A", "2", "K")]
        self.assertEqual(self.agent._trade_rank(message, actions)["rank"], "K")

    def test_mixed_agents_finish_successive_rounds(self):
        for seed in range(3):
            game = Game(PLAYERS, rng=random.Random(seed))
            agents = {p: ExperiencedAgent2() if i % 2 else BasicAgent()
                      for i, p in enumerate(PLAYERS)}
            for _ in range(3):
                game.start_round()
                result = game.run(agents, max_actions=3000)
                self.assertEqual(set(result.rankings.values()), {1, 2, 3, 4})
