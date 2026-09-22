import random
import unittest
from copy import deepcopy

from src.client import BasicClient
from src.president import Game, Round, standard_deck
from tests.test_president import PLAYERS, action_cards, card, ready_round


class BasicClientTests(unittest.TestCase):
    def setUp(self):
        self.client = BasicClient()

    def test_plays_lowest_rank_even_when_pass_and_bomb_are_offered_first(self):
        round_ = ready_round({"a": [card("10"), card("J"), card("2")],
                              "b": [card("Q")], "c": [card("K")], "d": [card("A")]},
                             top=("9", 1), last="d")
        message = round_.message_for("a")
        before = deepcopy(message)
        self.assertEqual(self.client.respond(message), action_cards(card("10")))
        self.assertEqual(message, before)

    def test_lowest_rank_takes_priority_over_group_size(self):
        round_ = ready_round({"a": [card("7"), card("8"), card("8", "D")],
                              "b": [card("9")], "c": [card("10")], "d": [card("J")]})
        self.assertEqual(self.client.respond(round_.message_for("a")), action_cards(card("7")))

    def test_equal_rank_lead_prefers_largest_group(self):
        round_ = ready_round({"a": [card("7"), card("7", "D"), card("K")],
                              "b": [card("9")], "c": [card("10")], "d": [card("J")]})
        self.assertEqual(self.client.respond(round_.message_for("a")),
                         action_cards(card("7"), card("7", "D")))

    def test_bombs_instead_of_passing_when_no_ordinary_play_exists(self):
        round_ = ready_round({"a": [card("3"), card("2")],
                              "b": [card("9")], "c": [card("10")], "d": [card("J")]},
                             top=("A", 2), last="d")
        action = self.client.respond(round_.message_for("a"))
        self.assertEqual(action, {"type": "bomb", "card": card("2").json()})
        round_.apply_action("a", action)

    def test_passes_only_when_unable_to_play(self):
        round_ = ready_round({"a": [card("3")], "b": [card("9")],
                              "c": [card("10")], "d": [card("J")]},
                             top=("A", 1), last="d")
        self.assertEqual(self.client.respond(round_.message_for("a")), {"type": "pass"})

    def test_always_completes_for_every_completion_size(self):
        for needed in (1, 2, 3):
            with self.subTest(needed=needed):
                completion = {"type": "complete", "cards": [card("7", s).json()
                              for s in ("D", "H", "S")[:needed]]}
                message = {"request": {"type": "completion", "legal_actions": [
                    {"type": "decline_completion"}, completion]}}
                self.assertEqual(self.client.respond(message), completion)
        message["request"]["legal_actions"] = [{"type": "decline_completion"}]
        self.assertEqual(self.client.respond(message), {"type": "decline_completion"})

    def test_drafts_highest_available_face_card_in_game_rank_order(self):
        round_ = Round(PLAYERS, first_round=False,
                       roles=dict(zip(PLAYERS, range(1, 5))), deck=standard_deck())
        # The sorted deck exposes 6, 9, Q, 2. Previously chosen piles are excluded.
        for player, expected_pile in zip(PLAYERS, (3, 2, 1, 0)):
            action = self.client.respond(round_.message_for(player))
            self.assertEqual(action, {"type": "choose_pile", "pile": expected_pile})
            round_.apply_action(player, action)

    def test_trade_requests_descend_after_failure_and_restart_after_success(self):
        round_ = Round(PLAYERS, first_round=False,
                       roles=dict(zip(PLAYERS, range(1, 5))), deck=standard_deck())
        for player, pile in zip(PLAYERS, (0, 1, 3, 2)):
            round_.apply_action(player, {"type": "choose_pile", "pile": pile})
        # Scum holds Q as its highest rank, with enough queens for both trades.
        for _ in range(2):
            for rank in ("2", "A", "K", "Q"):
                action = self.client.respond(round_.message_for("a"))
                self.assertEqual(action, {"type": "trade_request", "rank": rank})
                round_.apply_action("a", action)
            self.assertEqual(round_.phase, "trade_return")
            action = self.client.respond(round_.message_for("a"))
            self.assertEqual(action["card"]["rank"], "3")
            round_.apply_action("a", action)
        self.assertEqual(round_._actor(), "b")
        self.assertEqual(self.client.respond(round_.message_for("b")),
                         {"type": "trade_request", "rank": "2"})

    def test_basic_clients_finish_multiple_rounds_using_only_json_messages(self):
        for seed in range(3):
            game = Game(PLAYERS, rng=random.Random(seed))
            clients = {p: BasicClient() for p in PLAYERS}
            for _ in range(3):
                round_ = game.start_round()
                self.assertIs(game.run(clients, max_actions=2000), round_)
                self.assertEqual(set(round_.rankings.values()), {1, 2, 3, 4})

    def test_rejects_messages_without_a_request(self):
        with self.assertRaisesRegex(ValueError, "no legal action"):
            self.client.respond({"request": None})
