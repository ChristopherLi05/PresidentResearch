"""Tactical and protocol regressions for the second experienced policy."""

from copy import deepcopy
from collections import Counter
import random
import unittest
from unittest.mock import patch

from src.agent import BasicAgent, ExperiencedAgent2, ExperiencedAgent3, ExperiencedAgent4
from src.president import Card, Game, RANK_VALUE, standard_deck
from src.solver import solve_round
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


class ExperiencedAgent3Tests(unittest.TestCase):
    def test_weak_hand_does_not_force_conservation_of_an_ace(self):
        round_ = ready_round({
            "a": [card(r) for r in ("3", "4", "5", "6", "7", "A")],
            "b": [card("9")], "c": [card("10")], "d": [card("J")],
        }, top=("Q", 1), last="d")
        message = round_.message_for("a")
        before = deepcopy(message)
        self.assertEqual(ExperiencedAgent2().respond(message)["type"], "play")
        self.assertEqual(ExperiencedAgent3().respond(message)["type"], "play")
        self.assertEqual(message, before)
        # Public history now indicates an opponent has only three cards left.
        message["events"].append({"type": "played", "player": "b",
                                  "cards": [c.json() for c in
                                            [card(r, s) for r in ("8", "9", "10")
                                             for s in "CDHS"][:10]]})
        self.assertEqual(ExperiencedAgent3().respond(message)["type"], "play")

    def test_conservation_does_not_prevent_clear_or_forced_lead(self):
        round_ = ready_round({
            "a": [card(r) for r in ("3", "4", "5", "6", "7", "A")],
            "b": [card("9")], "c": [card("10")], "d": [card("J")],
        }, top=("A", 1), last="d")
        round_.top_stack_count = 3
        message = round_.message_for("a")
        self.assertEqual(ExperiencedAgent3().respond(message), ExperiencedAgent2().respond(message))
        round_.top_rank = None
        message = round_.message_for("a")
        self.assertEqual(ExperiencedAgent3().respond(message), ExperiencedAgent2().respond(message))

    def test_control_reserve_values_support_for_weak_groups(self):
        base, agent = ExperiencedAgent2(), ExperiencedAgent3()
        unseen, opponents = Counter(), [("b", 6), ("c", 6)]
        # Isolate hand shape from probability estimation: one reliable group
        # supporting weak groups deserves more reserve value than all-weak
        # groups, while a lone group has nothing left to support.
        def control(rank, *args):
            return 1.0 if rank == "A" else 0.0
        with patch.object(base, "_control", side_effect=control), patch.object(agent, "_control", side_effect=control):
            for hand in (Counter({"3": 1, "4": 1}), Counter({"A": 1})):
                self.assertEqual(agent._hand_cost(hand, unseen, opponents),
                                 base._hand_cost(hand, unseen, opponents))
            supported = Counter({"3": 1, "4": 1, "A": 1})
            self.assertLess(agent._hand_cost(supported, unseen, opponents),
                            base._hand_cost(supported, unseen, opponents))

    def test_reserve_preference_fades_with_progress_and_finish_pressure(self):
        base, agent = ExperiencedAgent2(), ExperiencedAgent3()
        hand, unseen = Counter({"3": 1, "4": 1, "A": 1}), Counter()
        def control(rank, *args):
            return 1.0 if rank == "A" else 0.0
        with patch.object(base, "_control", side_effect=control), patch.object(agent, "_control", side_effect=control):
            def bonus(sizes):
                opponents = [(str(i), n) for i, n in enumerate(sizes)]
                return (base._hand_cost(hand, unseen, opponents)
                        - agent._hand_cost(hand, unseen, opponents))
            self.assertGreater(bonus([13, 13, 13]), bonus([6, 6, 6]))
            self.assertGreater(bonus([6, 6, 6]), bonus([2, 2, 2]))
            # Same total cards: one close opponent reduces patience further.
            self.assertGreater(bonus([6, 6, 6]), bonus([1, 8, 9]))
            self.assertEqual(bonus([]), 0)
            self.assertEqual(bonus([13, 13]), bonus([14, 13]))

    def test_uses_exact_solver_when_two_players_remain(self):
        # Build a protocol-realistic endgame: every card outside the two
        # active hands appears in public play history, so the opponent hand is
        # exactly recoverable without private information.
        round_ = ready_round({
            "a": [card("3"), card("4")],
            "b": [card("5"), card("2")],
        })
        remaining = set(standard_deck()) - set(round_.hands["a"] + round_.hands["b"])
        round_.events = [{"type": "opening_play", "player": "c",
                          "cards": [next(iter(remaining)).json()]}]
        used = Card.from_json(round_.events[0]["cards"][0])
        round_.events.append({"type": "played", "player": "d",
                              "cards": [c.json() for c in sorted(remaining - {used})]})
        message = round_.message_for("a")
        rebuilt = ExperiencedAgent3._endgame_round(message)
        self.assertIsNotNone(rebuilt)
        expected = solve_round(rebuilt).best_action
        self.assertEqual(ExperiencedAgent3().respond(message), expected)

    def test_uses_solver_with_compact_message_without_turn(self):
        round_ = ready_round({"a": [card("3"), card("4")], "b": [card("5"), card("2")]})
        remaining = set(standard_deck()) - set(round_.hands["a"] + round_.hands["b"])
        round_.events = [{"type": "opening_play", "player": "c",
                          "cards": [next(iter(remaining)).json()]}]
        used = Card.from_json(round_.events[0]["cards"][0])
        round_.events.append({"type": "played", "player": "d",
                              "cards": [c.json() for c in sorted(remaining - {used})]})
        message = round_.message_for("a")
        expected = ExperiencedAgent3().respond(message)
        message.pop("turn")
        self.assertEqual(ExperiencedAgent3().respond(message), expected)

    def test_malformed_completion_history_falls_back_to_parent_policy(self):
        round_ = ready_round({"a": [card("3"), card("4")], "b": [card("5"), card("2")]})
        round_._offer_completions("a")
        message = round_.message_for("a")
        message["events"].append({"type": "played", "player": "b", "cards": []})
        self.assertEqual(ExperiencedAgent3().respond(message), ExperiencedAgent2().respond(message))

    def test_falls_back_to_experienced_agent2_outside_a_duel(self):
        round_ = ready_round({"a": [card("3")], "b": [card("4")], "c": [card("5")]})
        message = round_.message_for("a")
        self.assertEqual(ExperiencedAgent3().respond(message), ExperiencedAgent2().respond(message))


class ExperiencedAgent4Tests(unittest.TestCase):
    def setUp(self):
        self.agent = ExperiencedAgent4()

    @staticmethod
    def _unseen(**counts):
        return Counter({rank: counts.get(rank, 0) for rank in RANK_VALUE})

    def test_next_opponent_bomb_is_a_turn_order_danger(self):
        unseen = self._unseen(**{"2": 1})
        danger = self.agent._turn_order_danger(
            {"type": "pass"}, {"rank": "9", "count": 1, "stack_count": 1},
            unseen, [("b", 2), ("c", 10)])
        self.assertEqual(danger, self.agent.NEXT_BOMB_PENALTY)

    def test_matching_play_skips_next_normal_response(self):
        unseen = self._unseen(**{"10": 1})
        danger = self.agent._turn_order_danger(
            {"type": "play", "cards": [card("10").json()]},
            {"rank": "10", "count": 1, "stack_count": 1}, unseen, [("b", 1)])
        self.assertEqual(danger, 0.0)

    def test_immediate_clear_has_no_next_opponent_danger(self):
        unseen = self._unseen(**{"2": 1, "A": 4})
        danger = self.agent._turn_order_danger(
            {"type": "play", "cards": [card("A").json() for _ in range(4)]}, {},
            unseen, [("b", 5)])
        self.assertEqual(danger, 0.0)
