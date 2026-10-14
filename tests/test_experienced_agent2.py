"""Tactical and protocol regressions for the second experienced policy."""

from copy import deepcopy
import random
import unittest

from src.agent import BasicAgent, ExperiencedAgent2
from src.president import Game
from tests.test_president import PLAYERS, card, ready_round


class ExperiencedAgent2Tests(unittest.TestCase):
    def setUp(self):
        self.agent = ExperiencedAgent2()

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
