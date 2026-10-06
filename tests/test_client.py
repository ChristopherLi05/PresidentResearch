import random
import unittest
from copy import deepcopy

from src.agent import BasicAgent, BasicAgentPlus, Agent, ExperiencedAgent
from src.president import Game, Round, standard_deck
from src.tk_client import HumanClient
from src.tk_four_human_game import PLAYERS as HUMAN_PLAYERS
from src.tk_omniscient_game import omniscient_message
from tests.test_president import PLAYERS, action_cards, card, ready_round


class BasicClientTests(unittest.TestCase):
    def setUp(self):
        self.client = BasicAgent()

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

    def test_plays_jack_before_bomb(self):
        round_ = ready_round({"a": [card("J"), card("2")],
                              "b": [card("Q")], "c": [card("K")], "d": [card("A")]},
                             top=("10", 1), last="d")
        self.assertEqual(self.client.respond(round_.message_for("a")),
                         action_cards(card("J")))

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
            clients = {p: BasicAgent() for p in PLAYERS}
            for _ in range(3):
                round_ = game.start_round()
                self.assertIs(game.run(clients, max_actions=2000), round_)
                self.assertEqual(set(round_.rankings.values()), {1, 2, 3, 4})

    def test_rejects_messages_without_a_request(self):
        with self.assertRaisesRegex(ValueError, "no legal action"):
            self.client.respond({"request": None})


class BasicClientPlusTests(unittest.TestCase):
    def test_bombs_replace_jack_or_higher_ordinary_plays(self):
        client = BasicAgentPlus()
        round_ = ready_round({"a": [card("J"), card("2")],
                              "b": [card("Q")], "c": [card("K")], "d": [card("A")]},
                             top=("10", 1), last="d")
        self.assertEqual(client.respond(round_.message_for("a")),
                         {"type": "bomb", "card": card("2").json()})


class ExperiencedAgentTests(unittest.TestCase):
    def setUp(self):
        self.client = ExperiencedAgent()

    @staticmethod
    def _trade_message(cards, *, events=None):
        return {
            "player": "a", "hand": [c.json() for c in cards],
            "events": events or [{"type": "draft_complete"}],
            "request": {"type": "trade_request", "legal_actions": [
                {"type": "trade_request", "rank": rank}
                for rank in ("3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2")
            ]},
        }

    def test_trade_priority_creates_quad_before_matching_ace_or_requesting_two(self):
        message = self._trade_message([card("7", suit) for suit in ("C", "D", "H")] +
                                      [card("A"), card("2")])
        self.assertEqual(self.client.respond(message), {"type": "trade_request", "rank": "7"})

    def test_trade_request_is_recalculated_after_a_failure(self):
        events = [{"type": "draft_complete"},
                  {"type": "trade_requested", "initiator": "a", "receiver": "d",
                   "rank": "A", "success": False}]
        message = self._trade_message([card("A"), card("3"), card("4")], events=events)
        self.assertEqual(self.client.respond(message), {"type": "trade_request", "rank": "2"})

    def test_trade_requests_third_two_only_after_ace_match_is_no_longer_available(self):
        message = self._trade_message([card("A", "C"), card("A", "D"), card("2", "C"), card("2", "D"),
                                       card("3"), card("4"), card("5")])
        self.assertEqual(self.client.respond(message), {"type": "trade_request", "rank": "2"})

    def test_trade_matches_k_q_j_in_descending_rank_order_after_two_failures(self):
        events = [{"type": "draft_complete"},
                  {"type": "trade_requested", "initiator": "a", "receiver": "d", "rank": "2", "success": False}]
        message = self._trade_message([card("J"), card("Q"), card("K"), card("3"), card("4")], events=events)
        self.assertEqual(self.client.respond(message), {"type": "trade_request", "rank": "K"})

    def test_trade_requests_high_singleton_before_lower_rank_work(self):
        events = [{"type": "draft_complete"},
                  {"type": "trade_requested", "initiator": "a", "receiver": "d", "rank": "2", "success": False}]
        message = self._trade_message([card("3"), card("4")], events=events)
        self.assertEqual(self.client.respond(message), {"type": "trade_request", "rank": "A"})

    def test_fourth_two_requires_completed_high_rank_work_and_four_group_cover(self):
        safe_hand = [card(rank, suit) for rank in ("J", "Q", "K", "A") for suit in ("C", "D", "H", "S")]
        safe_hand += [card("2", "C"), card("2", "D"), card("2", "H")]
        self.assertEqual(self.client.respond(self._trade_message(safe_hand))["rank"], "2")

    def test_trade_does_not_request_an_unsafe_extra_two(self):
        message = self._trade_message([card("3"), card("2")])
        self.assertNotEqual(self.client.respond(message)["rank"], "2")

    def test_trade_return_prefers_a_low_triple_that_becomes_a_pair(self):
        round_ = ready_round({"a": [card("4", suit) for suit in ("C", "D", "H")] +
                                    [card("9"), card("2")]})
        round_.phase = "trade_return"
        round_._pending_trade = ("a", "d", "4")
        action = self.client.respond(round_.message_for("a"))
        self.assertEqual(action["card"]["rank"], "4")

    def test_trade_return_preserves_bomb_safety_and_does_not_return_a_two(self):
        round_ = ready_round({"a": [card("3"), card("4", "C"), card("4", "D"), card("2", "C"), card("2", "D")]})
        round_.phase = "trade_return"
        round_._pending_trade = ("a", "d", "4")
        action = self.client.respond(round_.message_for("a"))
        self.assertEqual(action["card"]["rank"], "4")

    def test_trade_return_uses_low_singleton_before_a_pair_or_high_card(self):
        round_ = ready_round({"a": [card("5"), card("9", "C"), card("9", "D"), card("K"), card("2")]})
        round_.phase = "trade_return"
        round_._pending_trade = ("a", "d", "5")
        self.assertEqual(self.client.respond(round_.message_for("a"))["card"]["rank"], "5")

    def test_completion_declines_only_when_it_would_break_bomb_safety(self):
        unsafe = {
            "hand": [c.json() for c in (card("7"), card("3"), card("2", "C"), card("2", "D"))],
            "request": {"type": "completion", "legal_actions": [
                {"type": "complete", "cards": [card("7").json()]}, {"type": "decline_completion"},
            ]},
        }
        self.assertEqual(self.client.respond(unsafe), {"type": "decline_completion"})
        unsafe["hand"] = [card("7").json()]
        self.assertEqual(self.client.respond(unsafe)["type"], "complete")

    def test_play_spends_a_two_when_ordinary_groups_equal_twos(self):
        round_ = ready_round({"a": [card("3"), card("4"), card("2", "C"), card("2", "D")],
                              "b": [card("9")], "c": [card("10")], "d": [card("J")]},
                             top=("3", 1), last="d")
        self.assertEqual(self.client.respond(round_.message_for("a"))["type"], "bomb")

    def test_play_finishes_normally_but_never_uses_a_finishing_bomb(self):
        normal_finish = ready_round({"a": [card("4")], "b": [card("9")], "c": [card("10")], "d": [card("J")]},
                                   top=("3", 1), last="d")
        self.assertEqual(self.client.respond(normal_finish.message_for("a")), action_cards(card("4")))

        bomb_escape = ready_round({"a": [card("3"), card("2")], "b": [card("9")], "c": [card("10")], "d": [card("J")]},
                                  top=("A", 1), last="d")
        action = self.client.respond(bomb_escape.message_for("a"))
        self.assertEqual(action["type"], "bomb")
        self.assertEqual(action["card"]["rank"], "2")

    def test_play_removes_a_singleton_instead_of_splitting_a_pair(self):
        round_ = ready_round({"a": [card("4", "C"), card("4", "D"), card("5")],
                              "b": [card("9")], "c": [card("10")], "d": [card("J")]},
                             top=("3", 1), last="d")
        self.assertEqual(self.client.respond(round_.message_for("a")), action_cards(card("5")))

    def test_play_uses_low_quad_and_preserves_high_quad_when_other_groups_exist(self):
        low_quad = ready_round({"a": [card("3", suit) for suit in ("C", "D", "H", "S")] +
                                      [card("4"), card("2")]})
        self.assertEqual(self.client.respond(low_quad.message_for("a"))["cards"][0]["rank"], "3")

        high_quad = ready_round({"a": [card("A", suit) for suit in ("C", "D", "H", "S")] +
                                       [card("3"), card("4"), card("2")]})
        self.assertEqual(self.client.respond(high_quad.message_for("a"))["cards"][0]["rank"], "3")

    def test_passes_a_low_board_when_every_play_is_a_structure_sacrifice(self):
        round_ = ready_round({"a": [card("4", "C"), card("4", "D")],
                              "b": [card("9")], "c": [card("10")], "d": [card("J")]},
                             top=("3", 1), last="d")
        self.assertEqual(self.client.respond(round_.message_for("a")), {"type": "pass"})

    def test_dangerous_opponent_prevents_a_sacrifice_pass(self):
        round_ = ready_round({"a": [card("4", "C"), card("4", "D")],
                              "b": [card("9")], "c": [card("10")], "d": [card("J")]},
                             top=("3", 1), last="d")
        round_.events.extend({"type": "played", "player": "b", "cards": [card("9").json()]}
                             for _ in range(11))
        self.assertEqual(self.client.respond(round_.message_for("a"))["type"], "play")


class HumanClientTests(unittest.TestCase):
    def test_is_a_client_without_creating_a_window(self):
        client = HumanClient()
        self.assertIsInstance(client, Agent)
        self.assertIsNone(client._root)

    def test_omniscient_client_is_a_human_protocol_client(self):
        from src.tk_client import OmniscientHumanClient
        self.assertIsInstance(OmniscientHumanClient(), Agent)

    def test_auto_declines_a_completion_when_there_is_no_possible_completion(self):
        client = HumanClient()
        message = {"request": {"type": "completion", "legal_actions": [
            {"type": "decline_completion"},
        ]}}
        self.assertEqual(client.respond(message), {"type": "decline_completion"})
        self.assertIsNone(client._root)

    def test_card_history_uses_only_public_card_events(self):
        history = HumanClient.card_history({"events": [
            {"type": "trade_completed", "initiator": "a", "receiver": "b"},
            {"type": "opening_play", "player": "a", "cards": [card("3").json()]},
            {"type": "bombed", "player": "b", "card": card("2").json()},
            {"type": "passed", "player": "d"},
            {"type": "played", "player": "c", "cards": [card("7").json()]},
        ]})
        self.assertEqual(history, [
            {"player": "a", "label": "opened", "cards": [card("3").json()]},
            {"player": "b", "label": "bombed", "cards": [card("2").json()]},
            {"player": "d", "label": "passed", "cards": []},
            {"player": "c", "label": "played", "cards": [card("7").json()]},
        ])

    def test_card_history_keeps_prior_turns_after_a_public_board_clear(self):
        history = HumanClient.card_history({"events": [
            {"type": "played", "player": "a", "cards": [card("7").json()]},
            {"type": "board_cleared", "player": "a"},
            {"type": "played", "player": "a", "cards": [card("8").json()]},
        ]})
        self.assertEqual(history, [
            {"player": "a", "label": "played", "cards": [card("7").json()]},
            {"player": "a", "label": "played", "cards": [card("8").json()]},
        ])

    def test_public_activity_includes_trade_outcome_and_board_clear(self):
        activity = HumanClient.public_activity({"events": [
            {"type": "trade_requested", "initiator": "a", "receiver": "d", "rank": "A", "success": False},
            {"type": "trade_completed", "initiator": "a", "receiver": "d"},
            {"type": "board_cleared", "player": "b"},
        ]})
        self.assertEqual(activity, [
            "Trade: a requested A from d — not available",
            "Trade complete: a exchanged with d",
            "Board cleared — b leads",
        ])

    def test_turn_history_keeps_public_events_across_a_board_clear(self):
        history = HumanClient.turn_history({"events": [
            {"type": "played", "player": "a", "cards": [card("7").json()]},
            {"type": "board_cleared", "player": "a"},
            {"type": "trade_requested", "initiator": "a", "receiver": "d", "rank": "A", "success": True},
        ]})
        self.assertEqual(history, [
            "1. a played: 7♣",
            "2. Board cleared; a leads",
            "3. a requested A from d (available)",
        ])

    def test_multi_card_action_is_available_regardless_of_click_order(self):
        client = HumanClient()
        first, second = card("7", "C").json(), card("7", "H").json()
        client._selected = [second, first]
        action = {"type": "play", "cards": [first, second]}
        self.assertEqual(client._selected_action([action]), action)

    def test_four_human_host_defines_four_distinct_seats(self):
        self.assertEqual(len(HUMAN_PLAYERS), 4)
        self.assertEqual(len(set(HUMAN_PLAYERS)), 4)


class ObservingAgent(BasicAgent):
    def __init__(self):
        super().__init__()
        self.observations = []

    def observe(self, message):
        self.observations.append(deepcopy(message))


class GameNotificationTests(unittest.TestCase):
    def test_run_broadcasts_initial_and_post_action_state_to_connected_clients(self):
        game = Game(PLAYERS)
        game.round = ready_round({"a": [card("7")], "b": [card("8")]})
        clients = {player: ObservingAgent() for player in ("a", "b")}
        game.run(clients, max_actions=1)
        for client in clients.values():
            self.assertEqual(len(client.observations), 2)
            self.assertIsNone(client.observations[0]["top"])
            self.assertTrue(client.observations[-1]["rankings"])


class OmniscientGameTests(unittest.TestCase):
    def test_state_includes_public_seating_and_later_round_roles(self):
        roles = dict(zip(PLAYERS, range(1, 5)))
        round_ = Round(PLAYERS, first_round=False, roles=roles, deck=standard_deck())
        message = round_.message_for("a")
        self.assertEqual(message["players"], list(PLAYERS))
        self.assertEqual(message["roles"], roles)

    def test_omniscient_message_contains_every_hand(self):
        round_ = Round(PLAYERS, first_round=True, deck=standard_deck())
        message = omniscient_message(round_, "a")
        self.assertEqual(set(message["all_hands"]), set(PLAYERS))
        self.assertEqual(sum(len(hand) for hand in message["all_hands"].values()), 52)

    def test_omniscient_hands_use_game_rank_order_not_string_order(self):
        round_ = Round(PLAYERS, first_round=True, deck=standard_deck())
        round_.hands["b"] = [card("A"), card("10"), card("2"), card("J")]
        ranks = [card_["rank"] for card_ in omniscient_message(round_, "a")["all_hands"]["b"]]
        self.assertEqual(ranks, ["10", "J", "A", "2"])
