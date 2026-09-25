"""Executable examples for every local rule supported by the President engine."""

import random
import re
import unittest
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from itertools import combinations
import json

from src.client import RandomClient
from src.president import Card, Game, IllegalAction, Round, standard_deck


PLAYERS = ["a", "b", "c", "d"]  # counterclockwise order


@contextmanager
def raises(exception, match=None):
    """Tiny stdlib replacement for pytest.raises, keeping this suite dependency-free."""
    try:
        yield
    except exception as error:
        if match and not re.search(match, str(error)):
            raise AssertionError("%r does not match %r" % (str(error), match)) from error
    else:
        raise AssertionError("expected %s" % exception.__name__)


def card(rank, suit="C"):
    return Card(rank, suit)


def action_cards(*cards):
    return {"type": "play", "cards": [c.json() for c in cards]}


def ready_round(hands, *, turn="a", top=None, last=None):
    """Make a small, deliberately controlled mid-round state."""
    round_ = Round(PLAYERS, first_round=True, deck=standard_deck())
    round_.hands = {p: list(hands.get(p, [])) for p in PLAYERS}
    round_.active = {p for p in PLAYERS if round_.hands[p]}
    round_.finish_order = [p for p in PLAYERS if p not in round_.active]
    round_.phase, round_.turn = "play", turn
    round_.top_rank, round_.top_count = (top if top else (None, 0))
    round_.top_stack_count = round_.top_count
    round_.last_player = last
    # This is a mid-round helper, so the opening-three constraint is already met.
    round_.events = [{"type": "opening_play", "player": "fixture"}]
    return round_


def decline_completions(round_, *, until=None):
    """Explicitly decline pending prompts, optionally stopping at one player."""
    while round_.phase == "completion" and round_._actor() != until:
        round_.apply_action(round_._actor(), {"type": "decline_completion"})
    if until is not None:
        assert round_.phase == "completion" and round_._actor() == until


def take_turn(round_, player, action):
    """Apply a turn, then have all active players decline completion."""
    events = round_.apply_action(player, action)
    decline_completions(round_)
    return events


def assert_rejected_unchanged(round_, player, action):
    before = deepcopy(round_.__dict__)
    with raises(IllegalAction):
        round_.apply_action(player, action)
    assert round_.__dict__ == before


def test_deck_ranks_and_first_deal_start_with_three_of_clubs():
    deck = standard_deck()
    assert len(deck) == len(set(deck)) == 52
    assert [c.rank for c in deck[:4]] == ["3"] * 4
    round_ = Round(PLAYERS, first_round=True, deck=deck)
    assert all(len(round_.hands[p]) == 13 for p in PLAYERS)
    assert round_.turn == "a"  # deck puts 3C in the first player's hand
    assert round_.top_rank is None


def test_opening_play_must_contain_a_three_regardless_of_suit_and_cannot_be_a_bomb():
    deck = standard_deck()
    deck[1], deck[4] = deck[4], deck[1]  # a receives both 3C and 3D.
    round_ = Round(PLAYERS, first_round=True, deck=deck)
    with raises(IllegalAction, match="a 3"):
        round_.apply_action("a", action_cards(card("4", "C")))
    with raises(IllegalAction):
        round_.apply_action("a", {"type": "bomb", "card": card("2").json()})
    round_.apply_action("a", action_cards(card("3", "D")))
    assert (round_.top_rank, round_.top_count, round_.last_player) == ("3", 1, "a")


def test_legal_actions_lists_every_valid_card_combination():
    round_ = ready_round({"a": [card("8", "C"), card("8", "D"), card("8", "H")], "b": [card("7")], "c": [card("9")], "d": [card("10")]}, turn="a")
    plays = [action for action in round_.legal_actions("a") if action["type"] == "play"]
    expected = {
        tuple(choice)
        for count in range(1, 4)
        for choice in combinations(round_.hands["a"], count)
    }
    assert {tuple(Card.from_json(c) for c in a["cards"]) for a in plays} == expected
    assert len(plays) == 7

    round_ = ready_round({"a": [card("8", "C"), card("8", "D"), card("8", "H")], "b": [card("7"), card("7", "D")], "c": [card("9")], "d": [card("10")]}, turn="a", top=("7", 2), last="d")
    plays = [action for action in round_.legal_actions("a") if action["type"] == "play"]
    assert {tuple(Card.from_json(c) for c in a["cards"]) for a in plays} == set(combinations(round_.hands["a"], 2))
    assert len(plays) == 3


def test_later_round_draft_is_president_to_scum_and_only_faceups_are_announced():
    order = ["c", "a", "d", "b"]
    roles = dict(zip(order, range(1, 5)))
    deck = standard_deck()
    round_ = Round(PLAYERS, first_round=False, roles=roles, deck=deck)
    assert round_.message_for("c")["request"]["type"] == "draft"
    start = round_.events[0]
    assert start == {"type": "round_started", "first_round": False,
                     "draft_order": order, "face_up": [deck[i].json() for i in (12, 25, 38, 51)]}
    for player, pile in zip(order, [3, 2, 1, 0]):
        assert round_._actor() == player
        round_.apply_action(player, {"type": "choose_pile", "pile": pile})
        assert round_.hands[player] == deck[pile * 13:(pile + 1) * 13]
        for observer in PLAYERS:
            state = round_.message_for(observer)
            assert Counter(Card.from_json(c) for c in state["hand"]) == Counter(round_.hands[observer])
            assert "hands" not in state and "piles" not in state
            assert state["events"][0] == start
            assert all(set(e) == {"type", "player", "pile"} for e in state["events"] if e["type"] == "pile_chosen")
    assert round_.phase == "trade_request"
    assert all(len(round_.hands[p]) == 13 for p in PLAYERS)
    assert start["draft_order"] == order  # historical events must not be mutated by drafting


def test_later_round_trade_retries_failed_request_and_hides_returned_card():
    roles = dict(zip(PLAYERS, range(1, 5)))
    round_ = Round(PLAYERS, first_round=False, roles=roles, deck=standard_deck())
    for player, pile in zip(PLAYERS, range(4)):
        round_.apply_action(player, {"type": "choose_pile", "pile": pile})
    # President a asks Scum d.  A failed request is public and may be retried.
    round_.hands["a"] = [card("A", "H")]
    round_.hands["d"] = [card("3", "S")]
    round_.apply_action("a", {"type": "trade_request", "rank": "K"})
    assert round_.phase == "trade_request"
    assert round_.events[-1] == {"type": "trade_requested", "initiator": "a", "receiver": "d", "rank": "K", "success": False}
    round_.apply_action("a", {"type": "trade_request", "rank": "3"})
    assert card("3", "S") in round_.hands["a"]
    round_.apply_action("a", {"type": "trade_return", "card": card("A", "H").json()})
    assert card("A", "H") in round_.hands["d"]
    assert round_.events[-1] == {"type": "trade_completed", "initiator": "a", "receiver": "d"}
    assert "card" not in round_.events[-1]


def test_later_round_completes_all_three_trades_then_starts_with_scum():
    roles = dict(zip(PLAYERS, range(1, 5)))
    round_ = Round(PLAYERS, first_round=False, roles=roles, deck=standard_deck())
    for player, pile in zip(PLAYERS, range(4)):
        round_.apply_action(player, {"type": "choose_pile", "pile": pile})

    # Keep only the cards relevant to the exchange protocol in this fixture.
    round_.hands = {
        "a": [card("A", "H"), card("K", "H")],
        "b": [card("J", "H")],
        "c": [card("5", "S")],
        "d": [card("3", "S"), card("4", "S")],
    }
    for requested, returned in (("3", card("A", "H")), ("4", card("K", "H"))):
        round_.apply_action("a", {"type": "trade_request", "rank": requested})
        round_.apply_action("a", {"type": "trade_return", "card": returned.json()})
    round_.apply_action("b", {"type": "trade_request", "rank": "5"})
    round_.apply_action("b", {"type": "trade_return", "card": card("J", "H").json()})

    assert round_.phase == "play" and round_.turn == "d"
    assert round_.hands["a"] == [card("3", "S"), card("4", "S")]
    assert round_.hands["b"] == [card("5", "S")]
    assert round_.hands["c"] == [card("J", "H")]
    assert round_.hands["d"] == [card("A", "H"), card("K", "H")]
    take_turn(round_, "d", action_cards(card("K", "H")))
    assert round_.top_rank == "K"  # later-round leads need not contain a 3


def test_play_must_match_count_rank_and_can_pass_then_play_later():
    round_ = ready_round({"a": [card("Q")], "b": [card("6"), card("6", "D"), card("J"), card("J", "D"), card("K")], "c": [card("9"), card("9", "D"), card("A")], "d": [card("10"), card("10", "D")]},
                         turn="b", top=("7", 2), last="a")
    assert_rejected_unchanged(round_, "b", action_cards(card("J")))
    assert_rejected_unchanged(round_, "b", action_cards(card("6"), card("6", "D")))
    assert action_cards(card("J"), card("J", "D")) in round_.legal_actions("b")
    take_turn(round_, "b", {"type": "pass"})
    assert round_.turn == "c"
    take_turn(round_, "c", action_cards(card("9"), card("9", "D")))
    assert round_.turn == "d"
    take_turn(round_, "d", {"type": "pass"})
    take_turn(round_, "a", {"type": "pass"})
    assert round_.turn == "b" and round_.top_rank == "9"
    take_turn(round_, "b", action_cards(card("J"), card("J", "D")))
    assert round_.top_rank == "J" and round_.last_player == "b"


def test_equal_rank_skips_next_player_and_return_to_last_player_clears_board():
    round_ = ready_round({"a": [card("7"), card("K")], "b": [card("7", "D"), card("Q")], "c": [card("8")], "d": [card("9")]},
                         turn="b", top=("7", 1), last="a")
    take_turn(round_, "b", action_cards(card("7", "D")))
    assert round_.turn == "d"  # c was skipped
    take_turn(round_, "d", {"type": "pass"})
    assert round_.turn == "a"
    take_turn(round_, "a", {"type": "pass"})
    assert round_.top_rank is None
    assert round_.turn == "b"  # board cleared back to most recent player


def test_completion_has_priority_over_skip_and_can_be_self_completion():
    # A player can play one of their four cards then immediately complete themself.
    round_ = ready_round({"a": [card("7", "C"), card("7", "D"), card("7", "H"), card("7", "S")], "b": [card("8")], "c": [card("9")], "d": [card("10")]}, turn="a")
    round_.apply_action("a", action_cards(card("7", "C")))
    assert round_.phase == "completion" and round_._actor() == "a"
    round_.apply_action("a", {"type": "complete", "cards": [card("7", "D").json(), card("7", "H").json(), card("7", "S").json()]})
    assert round_.top_rank is None and round_.turn == "b"  # a finished; b inherits the clear

    # a's equal play would skip b, but b can complete before that skip happens.
    round_ = ready_round({"a": [card("7", "D"), card("J")], "b": [card("7", "H"), card("7", "S")], "c": [card("8")], "d": [card("9")]}, turn="a", top=("7", 1), last="d")
    round_.apply_action("a", action_cards(card("7", "D")))
    decline_completions(round_, until="b")
    assert round_.phase == "completion" and round_._actor() == "b"
    round_.apply_action("b", {"type": "complete", "cards": [card("7", "H").json(), card("7", "S").json()]})
    assert round_.top_rank is None and round_.turn == "c"  # b finished; c inherits the clear


def test_declining_all_completions_resumes_the_pending_turn():
    round_ = ready_round({"a": [card("8"), card("K")], "b": [card("8", "D"), card("8", "H"), card("8", "S")], "c": [card("9")], "d": [card("10")]}, turn="a")
    round_.apply_action("a", action_cards(card("8")))
    decline_completions(round_, until="b")
    assert round_.phase == "completion" and round_._actor() == "b"
    round_.apply_action("b", {"type": "decline_completion"})
    decline_completions(round_)
    assert round_.phase == "play" and round_.turn == "b"
    assert (round_.top_rank, round_.top_count, round_.last_player) == ("8", 1, "a")


def test_play_that_makes_exactly_four_of_a_kind_clears_to_that_player():
    round_ = ready_round({"a": [card("8")], "b": [card("7", "C"), card("7", "D"), card("K")], "c": [card("9")], "d": [card("10")]},
                         turn="b", top=("7", 2), last="a")
    round_.apply_action("b", action_cards(card("7", "C"), card("7", "D")))
    assert round_.phase == "play"
    assert round_.top_rank is None
    assert round_.turn == "b"


def test_four_card_lead_clears_immediately_and_finished_leader_passes_the_clear():
    for player in PLAYERS:
        for finishes in (False, True):
            hands = {p: [card(rank)] for p, rank in zip(PLAYERS, ("8", "9", "10", "J"))}
            four = [card("7", suit) for suit in ("C", "D", "H", "S")]
            hands[player] = four + ([] if finishes else [card("K")])
            round_ = ready_round(hands, turn=player)
            action = action_cards(*four)
            assert action in round_.legal_actions(player)
            events = round_.apply_action(player, action)
            expected = PLAYERS[(PLAYERS.index(player) + 1) % 4] if finishes else player
            assert round_.phase == "play" and round_._actor() == expected
            assert (round_.top_rank, round_.top_count, round_.top_stack_count) == (None, 0, 0)
            assert round_.last_player is None
            assert round_.finish_order == ([player] if finishes else [])
            assert events[-1] == {"type": "board_cleared", "player": expected}


def test_first_round_can_open_with_all_four_threes():
    deck = standard_deck()
    for index in (1, 2, 3):
        deck[index], deck[index * 4] = deck[index * 4], deck[index]
    round_ = Round(PLAYERS, first_round=True, deck=deck)
    action = action_cards(*(card("3", suit) for suit in ("C", "D", "H", "S")))
    assert action in round_.legal_actions("a")
    events = round_.apply_action("a", action)
    assert events[0] == {"type": "opening_play", "player": "a", "cards": action["cards"]}
    assert events[-1] == {"type": "board_cleared", "player": "a"}
    assert round_.phase == "play" and round_._actor() == "a"
    assert (round_.top_rank, round_.top_count, round_.top_stack_count) == (None, 0, 0)


def test_equal_rank_skips_the_next_active_player_across_finished_seats():
    for player in PLAYERS:
        for retired in (p for p in PLAYERS if p != player):
            hands = {p: [card(rank)] for p, rank in zip(PLAYERS, ("8", "9", "10", "J")) if p != retired}
            hands[player] = [card("7", "D"), card("K")]
            round_ = ready_round(hands, turn=player, top=("7", 1), last=retired)
            take_turn(round_, player, action_cards(card("7", "D")))
            offset = PLAYERS.index(player) + 1
            seats = PLAYERS[offset:] + PLAYERS[:offset]
            remaining = [p for p in seats if p != retired]
            assert round_.phase == "play" and round_._actor() == remaining[1]
            assert (round_.top_rank, round_.top_count, round_.top_stack_count) == ("7", 1, 2)
            assert round_.last_player == player


def test_bomb_clears_board_and_finishing_on_bomb_is_last_place():
    # Defensive handling of a supplied legacy state. In normal play an only-2s
    # hand is announced and retired before its owner could play the last bomb.
    round_ = ready_round({"a": [card("2")], "b": [card("8")], "c": [card("9")], "d": [card("10")]}, turn="a", top=("7", 1), last="d")
    round_.apply_action("a", {"type": "bomb", "card": card("2").json()})
    assert "a" not in round_.active and round_.bomb_finishers == ["a"]
    assert round_.top_rank is None and round_.turn == "b"


def test_bomb_finisher_ranks_after_all_normal_finishers():
    round_ = ready_round({"a": [card("2")], "b": [card("8")], "c": [card("9")], "d": [card("10")]},
                         turn="a", top=("7", 1), last="d")
    round_.apply_action("a", {"type": "bomb", "card": card("2").json()})
    take_turn(round_, "b", action_cards(card("8")))
    take_turn(round_, "c", action_cards(card("9")))
    assert round_.done
    assert round_.rankings == {"b": 1, "c": 2, "d": 3, "a": 4}


def test_bomb_cannot_start_a_cleared_board_and_bad_card_payloads_are_illegal_actions():
    round_ = ready_round({"a": [card("2")], "b": [card("8")], "c": [card("9")], "d": [card("10")]}, turn="a")
    with raises(IllegalAction, match="cleared board"):
        round_.apply_action("a", {"type": "bomb", "card": card("2").json()})
    with raises(IllegalAction, match="invalid cards"):
        round_.apply_action("a", {"type": "play", "cards": [{"rank": "not-a-rank", "suit": "C"}]})


def test_later_round_requires_a_complete_previous_role_assignment():
    with raises(ValueError, match="roles"):
        Round(PLAYERS, first_round=False, roles={}, deck=standard_deck())


def test_empty_hand_places_player_and_clear_is_inherited_only_when_play_returns_to_them():
    round_ = ready_round({"a": [card("8")], "b": [card("9"), card("K")], "c": [card("10")], "d": [card("J")]}, turn="a")
    take_turn(round_, "a", action_cards(card("8")))
    assert "a" not in round_.active and round_.finish_order == ["a"]
    assert round_.turn == "b" and round_.top_rank == "8"
    take_turn(round_, "b", {"type": "pass"})
    take_turn(round_, "c", {"type": "pass"})
    take_turn(round_, "d", {"type": "pass"})
    assert round_.turn == "b" and round_.top_rank is None


def test_finishing_play_still_allows_completion_and_preserves_its_skip():
    round_ = ready_round({"a": [card("7", "D")], "b": [card("8")],
                          "c": [card("7", "H"), card("7", "S")], "d": [card("9")]},
                         top=("7", 1), last="d")
    round_.apply_action("a", action_cards(card("7", "D")))
    decline_completions(round_, until="c")
    assert round_.phase == "completion" and round_._actor() == "c"
    round_.apply_action("c", {"type": "decline_completion"})
    decline_completions(round_)
    assert round_.phase == "play" and round_.turn == "c"  # b was skipped
    assert round_.top_rank == "7" and round_.top_stack_count == 2


def test_only_twos_are_announced_done_without_being_played():
    round_ = ready_round({"a": [card("7"), card("K")], "b": [card("8"), card("2"), card("2", "D")], "c": [card("9")], "d": [card("10")]})
    take_turn(round_, "a", action_cards(card("7")))
    take_turn(round_, "b", action_cards(card("8")))
    assert "b" not in round_.active
    assert any(e == {"type": "twos_announced", "player": "b"} for e in round_.events)
    assert round_.hands["b"] == [card("2"), card("2", "D")]
    assert round_.bomb_finishers == ["b"] and round_.finish_order == []
    take_turn(round_, "c", action_cards(card("9")))
    take_turn(round_, "d", action_cards(card("10")))
    assert round_.rankings == {"c": 1, "d": 2, "a": 3, "b": 4}
    assert not any(e["type"] == "bombed" for e in round_.events)


def test_round_finished_is_emitted_once():
    round_ = ready_round({"a": [card("7")], "b": [card("8")], "c": [card("9")], "d": [card("10")]})
    take_turn(round_, "a", action_cards(card("7")))
    take_turn(round_, "b", action_cards(card("8")))
    take_turn(round_, "c", action_cards(card("9")))
    assert round_.done
    assert [event["type"] for event in round_.events].count("round_finished") == 1


def test_game_persists_roles_and_random_client_uses_json_protocol():
    game = Game(PLAYERS, rng=random.Random(4))
    clients = {p: RandomClient(random.Random(i)) for i, p in enumerate(PLAYERS)}
    first = game.start_round()
    game.run(clients)
    roles = game.finish_round()
    assert set(roles.values()) == {1, 2, 3, 4}
    later = game.start_round()
    assert later.roles == roles and later.first_round is False
    state = later.message_for(later._actor())
    assert state["type"] == "state" and state["request"]["legal_actions"]


def test_opening_starter_follows_three_of_clubs_in_every_seat():
    for seat, player in enumerate(PLAYERS):
        deck = standard_deck()
        deck[0], deck[seat] = deck[seat], deck[0]
        round_ = Round(PLAYERS, first_round=True, deck=deck)
        assert round_._actor() == player and round_.phase == "play"
        assert card("3") in round_.hands[player]
        assert Counter(c for hand in round_.hands.values() for c in hand) == Counter(deck)
        actions = round_.legal_actions(player)
        assert actions and all(a["type"] == "play" and all(c["rank"] == "3" for c in a["cards"]) for a in actions)
        assert_rejected_unchanged(round_, player, {"type": "pass"})


def test_all_rank_boundaries_compare_in_numeric_card_order():
    ranks = ["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
    for lower, higher in zip(ranks, ranks[1:]):
        round_ = ready_round({"a": [card(lower), card(higher), card("2")],
                              "b": [card("K", "D")], "c": [card("Q", "H")], "d": [card("J", "S")]},
                             top=(higher, 1), last="d")
        assert_rejected_unchanged(round_, "a", action_cards(card(lower)))
        round_.top_rank = lower
        take_turn(round_, "a", action_cards(card(higher)))
        assert round_.top_rank == higher and round_.turn == "b"


def test_single_bomb_beats_any_play_size_and_active_bomber_keeps_clear():
    for count in (1, 2, 3):
        round_ = ready_round({"a": [card("2"), card("2", "D"), card("8")],
                              "b": [card("9")], "c": [card("10")], "d": [card("J")]},
                             top=("A", count), last="d")
        bombs = [a for a in round_.legal_actions("a") if a["type"] == "bomb"]
        assert bombs == [{"type": "bomb", "card": card("2", suit).json()} for suit in ("C", "D")]
        round_.apply_action("a", bombs[1])
        assert round_.phase == "play" and round_.turn == "a" and "a" in round_.active
        assert round_.top_rank is None and round_.top_count == round_.top_stack_count == 0
        assert round_.hands["a"] == [card("2"), card("8")]
        assert_rejected_unchanged(round_, "a", bombs[0])  # cannot lead another bomb


def test_multiple_only_twos_announcements_fill_bottom_places_in_order():
    for penalty_count in (2, 3):
        hands = {p: [card(rank)] for p, rank in zip(PLAYERS, ("7", "8", "9", "10"))}
        for player, suit in zip(PLAYERS[:penalty_count], ("C", "D", "H")):
            hands[player].append(card("2", suit))
        round_ = ready_round(hands)
        for player, rank in zip(PLAYERS[:3], ("7", "8", "9")):
            take_turn(round_, player, action_cards(card(rank)))
        order = PLAYERS[penalty_count:] + PLAYERS[:penalty_count]
        assert round_.done and round_.rankings == dict(zip(order, range(1, 5)))
        assert [e["player"] for e in round_.events if e["type"] == "twos_announced"] == PLAYERS[:penalty_count]
        assert all(round_.hands[p] == [card("2", suit)] for p, suit in zip(PLAYERS[:penalty_count], ("C", "D", "H")))


def test_completion_leaving_only_twos_is_penalized_and_clear_is_inherited():
    round_ = ready_round({"a": [card("7"), card("K")],
                          "b": [card("7", s) for s in ("D", "H", "S")] + [card("2")],
                          "c": [card("8")], "d": [card("9")]})
    round_.apply_action("a", action_cards(card("7")))
    decline_completions(round_, until="b")
    round_.apply_action("b", {"type": "complete", "cards": [card("7", s).json() for s in ("D", "H", "S")]})
    assert round_.phase == "play" and round_.turn == "c" and round_.top_rank is None
    assert round_.hands["b"] == [card("2")] and round_.bomb_finishers == ["b"]
    take_turn(round_, "c", action_cards(card("8")))
    take_turn(round_, "d", action_cards(card("9")))
    assert round_.rankings == {"c": 1, "d": 2, "a": 3, "b": 4}


def test_completion_can_be_reconsidered_after_a_pass():
    round_ = ready_round({"a": [card("7"), card("K")], "b": [card("8")],
                          "c": [card("7", s) for s in ("D", "H", "S")] + [card("Q")], "d": [card("9")]})
    take_turn(round_, "a", action_cards(card("7")))  # c initially declines
    assert round_.turn == "b"
    round_.apply_action("b", {"type": "pass"})
    decline_completions(round_, until="c")
    completion = {"type": "complete", "cards": [card("7", s).json() for s in ("D", "H", "S")]}
    assert completion in round_.legal_actions("c")
    round_.apply_action("c", completion)
    assert round_.top_rank is None and round_.turn == "c" and round_.hands["c"] == [card("Q")]


def test_last_pass_allows_completion_before_the_board_returns_to_its_owner():
    round_ = ready_round({"a": [card("7"), card("K")], "b": [card("8")], "c": [card("9")],
                          "d": [card("7", s) for s in ("D", "H", "S")] + [card("Q")]})
    take_turn(round_, "a", action_cards(card("7")))
    take_turn(round_, "b", {"type": "pass"})
    take_turn(round_, "c", {"type": "pass"})
    round_.apply_action("d", {"type": "pass"})
    assert round_.top_rank == "7" and round_._actor() == "d"
    round_.apply_action("d", {"type": "complete", "cards": [card("7", s).json() for s in ("D", "H", "S")]})
    assert round_.top_rank is None and round_.turn == "d"


def test_completion_prompts_do_not_disclose_other_players_cards():
    left = ready_round({"a": [card("7"), card("K")], "b": [card("8")],
                        "c": [card("7", "D"), card("7", "H"), card("7", "S"), card("Q")],
                        "d": [card("9"), card("J"), card("10")]})
    right = ready_round({"a": [card("7"), card("K")], "b": [card("8")],
                         "c": [card("7", "D"), card("9"), card("J"), card("Q")],
                         "d": [card("7", "H"), card("7", "S"), card("10")]})
    for round_ in (left, right):
        round_.apply_action("a", action_cards(card("7")))
    for actor in PLAYERS:
        assert left._actor() == right._actor() == actor
        assert left.message_for("b") == right.message_for("b")
        if actor == "c":
            assert any(a["type"] == "complete" for a in left.legal_actions(actor))
            assert right.legal_actions(actor) == [{"type": "decline_completion"}]
        for round_ in (left, right):
            assert_rejected_unchanged(round_, actor, {"type": "pass"})
            round_.apply_action(actor, {"type": "decline_completion"})
    assert left.message_for("b") == right.message_for("b")
    assert left.phase == right.phase == "play" and left.turn == right.turn == "b"


def test_returned_trade_card_is_private_in_every_observers_full_message():
    rounds = []
    for returned in (card("A", "H"), card("K", "H")):
        round_ = Round(PLAYERS, first_round=False, roles=dict(zip(PLAYERS, range(1, 5))), deck=standard_deck())
        for player, pile in zip(PLAYERS, range(4)):
            round_.apply_action(player, {"type": "choose_pile", "pile": pile})
        round_.hands["a"] = [card("A", "H"), card("K", "H")]
        round_.hands["d"] = [card("3", "S")]
        round_.apply_action("a", {"type": "trade_request", "rank": "Q"})
        round_.apply_action("a", {"type": "trade_request", "rank": "3"})
        round_.apply_action("a", {"type": "trade_return", "card": returned.json()})
        for observer in PLAYERS:
            state = round_.message_for(observer)
            assert state["events"][-3:] == [
                {"type": "trade_requested", "initiator": "a", "receiver": "d", "rank": "Q", "success": False},
                {"type": "trade_requested", "initiator": "a", "receiver": "d", "rank": "3", "success": True},
                {"type": "trade_completed", "initiator": "a", "receiver": "d"},
            ]
            assert Counter(Card.from_json(c) for c in state["hand"]) == Counter(round_.hands[observer])
        assert round_.hands["d"] == [returned] and returned not in round_.hands["a"]
        rounds.append(round_)
    for observer in ("b", "c"):
        assert rounds[0].message_for(observer) == rounds[1].message_for(observer)


def test_messages_and_action_events_are_independent_snapshots():
    round_ = Round(PLAYERS, first_round=True, deck=standard_deck())
    before = deepcopy(round_.__dict__)
    state = round_.message_for("a")
    state["events"][0]["type"] = "opening_play"
    state["hand"][0]["rank"] = "A"
    state["request"]["legal_actions"][0]["cards"][0]["rank"] = "A"
    state["active_players"].clear()
    assert round_.__dict__ == before
    assert_rejected_unchanged(round_, "a", action_cards(card("4")))
    events = round_.apply_action("a", action_cards(card("3")))
    before = deepcopy(round_.__dict__)
    events[0]["cards"][0]["rank"] = "A"
    events[0]["type"] = "corrupted"
    assert round_.__dict__ == before


def test_turn_history_records_only_accepted_actions_as_independent_private_snapshots():
    round_ = Round(PLAYERS, first_round=True, deck=standard_deck())
    action = action_cards(card("3"))
    assert_rejected_unchanged(round_, "a", action_cards(card("4")))
    round_.apply_action("a", action)
    action["cards"][0]["rank"] = "A"

    assert round_.turn_history == [{
        "player": "a",
        "action": {"type": "play", "cards": [card("3").json()]},
    }]
    assert "turn_history" not in round_.message_for("b")


def test_rankings_and_persisted_roles_do_not_alias_returned_data():
    round_ = ready_round({"a": [card("7")], "b": [card("8")], "c": [card("9")], "d": [card("10")]})
    take_turn(round_, "a", action_cards(card("7")))
    take_turn(round_, "b", action_cards(card("8")))
    events = take_turn(round_, "c", action_cards(card("9")))
    expected = {"a": 1, "b": 2, "c": 3, "d": 4}
    state = round_.message_for("a")
    state["rankings"]["a"] = 99
    state["events"][-1]["rankings"]["b"] = 99
    events[-1]["rankings"]["c"] = 99
    assert round_.rankings == round_.events[-1]["rankings"] == expected
    game = Game(PLAYERS)
    game.round = round_
    game.round_number = 1
    roles = game.finish_round()
    roles["d"] = 99
    assert game.roles == round_.rankings == expected
    assert game.start_round().roles == expected


def test_invalid_play_payloads_and_wrong_actors_preserve_state():
    round_ = ready_round({"a": [card("7"), card("8"), card("2")],
                          "b": [card("9")], "c": [card("10")], "d": [card("J")]})
    invalid = [None, [], {}, {"type": "unknown"}, {"type": "pass"},
               action_cards(), action_cards(card("7"), card("8")),
               action_cards(card("7"), card("7")), action_cards(card("7"), card("7", "D")),
               action_cards(card("2")), {"type": "play", "cards": {}},
               {"type": "play", "cards": [None]},
               {"type": "play", "cards": [{"rank": "7"}]},
               {"type": "play", "cards": [{"rank": "7", "suit": "X"}]},
               {"type": "play", "cards": [{"rank": [], "suit": "C"}]}]
    for action in invalid:
        assert_rejected_unchanged(round_, "a", action)
    assert round_.legal_actions("b") == []
    assert_rejected_unchanged(round_, "b", action_cards(card("9")))
    assert_rejected_unchanged(round_, "unknown", action_cards(card("7")))
    round_.top_rank, round_.top_count, round_.top_stack_count, round_.last_player = "6", 1, 1, "d"
    for action in ({"type": "bomb"}, {"type": "bomb", "card": None},
                   {"type": "bomb", "card": card("7").json()},
                   {"type": "bomb", "card": card("2", "D").json()}):
        assert_rejected_unchanged(round_, "a", action)


def test_invalid_draft_and_trade_actions_preserve_state():
    round_ = Round(PLAYERS, first_round=False, roles=dict(zip(PLAYERS, range(1, 5))), deck=standard_deck())
    for pile in (-1, 4, True, "0", None):
        assert_rejected_unchanged(round_, "a", {"type": "choose_pile", "pile": pile})
    assert_rejected_unchanged(round_, "b", {"type": "choose_pile", "pile": 0})
    round_.apply_action("a", {"type": "choose_pile", "pile": 0})
    assert_rejected_unchanged(round_, "b", {"type": "choose_pile", "pile": 0})
    for player, pile in zip(PLAYERS[1:], range(1, 4)):
        round_.apply_action(player, {"type": "choose_pile", "pile": pile})
    for rank in ("invalid", None, [], {}):
        assert_rejected_unchanged(round_, "a", {"type": "trade_request", "rank": rank})
    assert_rejected_unchanged(round_, "b", {"type": "trade_request", "rank": "A"})
    round_.apply_action("a", {"type": "trade_request", "rank": "A"})
    assert round_.phase == "trade_return"
    assert_rejected_unchanged(round_, "a", {"type": "trade_return", "card": card("8").json()})
    assert_rejected_unchanged(round_, "a", {"type": "trade_return", "card": {"rank": "3"}})
    assert_rejected_unchanged(round_, "a", {"type": "trade_request", "rank": "A"})


def test_invalid_completions_preserve_state():
    round_ = ready_round({"a": [card("7"), card("K")], "b": [card("8")],
                          "c": [card("7", s) for s in ("D", "H", "S")] + [card("Q")], "d": [card("9")]})
    round_.apply_action("a", action_cards(card("7")))
    assert_rejected_unchanged(round_, "c", {"type": "complete", "cards": [card("7", s).json() for s in ("D", "H", "S")]})
    decline_completions(round_, until="c")
    for cards in ([], [card("7", "D")], [card("7", "D")] * 3,
                  [card("7", "D"), card("7", "H"), card("Q")],
                  [card("7", "C"), card("7", "D"), card("7", "H")]):
        assert_rejected_unchanged(round_, "c", {"type": "complete", "cards": [c.json() for c in cards]})


def test_round_ends_before_sole_remaining_player_can_complete():
    round_ = ready_round({"a": [card("8")], "b": [card("9")], "c": [card("7")],
                          "d": [card("7", s) for s in ("D", "H", "S")]})
    take_turn(round_, "a", action_cards(card("8")))
    take_turn(round_, "b", action_cards(card("9")))
    take_turn(round_, "c", {"type": "pass"})
    take_turn(round_, "d", {"type": "pass"})
    assert round_.turn == "c" and round_.top_rank is None
    round_.apply_action("c", action_cards(card("7")))
    assert round_.done and round_.rankings == dict(zip(PLAYERS, range(1, 5)))
    assert round_.hands["d"] == [card("7", s) for s in ("D", "H", "S")]
    assert all(round_.legal_actions(p) == [] and round_.message_for(p)["request"] is None for p in PLAYERS)
    assert_rejected_unchanged(round_, "d", {"type": "complete", "cards": [c.json() for c in round_.hands["d"]]})


def test_game_run_accepts_finishing_on_exact_action_limit():
    game = Game(PLAYERS)
    game.round = ready_round({"a": [card("7")], "b": [card("8")]})
    with raises(RuntimeError, match="action limit"):
        game.run({}, max_actions=0)
    assert not game.round.done
    assert game.run({"a": RandomClient(random.Random(0))}, max_actions=1) is game.round
    assert game.round.done
    assert game.run({}, max_actions=0) is game.round


def test_start_round_rejects_every_unfinished_phase_without_changing_the_game():
    game = Game(PLAYERS, rng=random.Random(4))
    clients = {p: RandomClient(random.Random(i)) for i, p in enumerate(PLAYERS)}
    observed = set()
    for _ in range(2):
        round_ = game.start_round()
        checked = set()
        for _ in range(2000):
            if round_.done:
                break
            if round_.phase not in checked:
                before = deepcopy(round_.__dict__)
                roles, number, rng_state = deepcopy(game.roles), game.round_number, game.rng.getstate()
                with raises(RuntimeError, match="current round has not finished"):
                    game.start_round()
                assert game.round is round_ and round_.__dict__ == before
                assert game.roles == roles and game.round_number == number
                assert game.rng.getstate() == rng_state
                checked.add(round_.phase)
            player = round_._actor()
            round_.apply_action(player, clients[player].respond(round_.message_for(player)))
        assert round_.done
        observed.update(checked)
        game.finish_round()
    assert observed == {"draft", "trade_request", "trade_return", "play", "completion"}


def test_start_round_uses_latest_rankings_without_an_explicit_finish_call():
    game = Game(PLAYERS)
    game.start_round(deck=standard_deck())
    for order in (PLAYERS, PLAYERS[1:] + PLAYERS[:1]):
        # Control the finishing order to ensure that the previous roles are stale.
        round_ = ready_round({p: [card(rank)] for p, rank in zip(order, ("7", "8", "9", "10"))}, turn=order[0])
        game.round = round_
        for player, rank in zip(order[:3], ("7", "8", "9")):
            take_turn(round_, player, action_cards(card(rank)))
        expected = dict(zip(order, range(1, 5)))
        assert round_.rankings == expected and game.roles != expected
        later = game.start_round(deck=standard_deck())
        assert later is not round_ and later.roles == game.roles == expected
        assert later.roles is not game.roles and game.roles is not round_.rankings
        assert later.phase == "draft" and later._actor() == order[0]
    assert game.round_number == 3


def test_failed_start_after_a_finished_round_preserves_round_and_metadata():
    game = Game(PLAYERS)
    game.start_round(deck=standard_deck())
    round_ = ready_round({"a": [card("7")], "b": [card("8")]})
    game.round = round_
    round_.apply_action("a", action_cards(card("7")))
    before = deepcopy(round_.__dict__)
    with raises(ValueError, match="deck"):
        game.start_round(deck=[])
    assert game.round is round_ and round_.__dict__ == before
    assert game.round_number == 1 and game.roles is None
    assert game.start_round(deck=standard_deck()).roles == round_.rankings


class GameplayChecks:
    """Check rules independently of the engine's action generator and turn helpers.

    Keep our own completion respondents and pending skip. Derive destinations
    from the seating order, never from the engine's pending transition/cursor.
    Drafting and trading have separate directed protocol tests above.
    """

    RANKS = ("3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")

    def __init__(self, round_):
        self.opening_required = round_.first_round
        self.started = False
        self.respondents = []
        self.pending = None

    def assert_legal(self, round_, player, action):
        kind = action["type"]
        if kind in ("play", "complete"):
            cards = [Card.from_json(c) for c in action["cards"]]
            assert cards and not (Counter(cards) - Counter(round_.hands[player]))
            assert len({c.rank for c in cards}) == 1 and cards[0].rank != "2"
            rank = cards[0].rank
        if round_.phase == "completion":
            assert kind in ("complete", "decline_completion")
            if kind == "complete":
                assert rank == round_.top_rank and len(cards) + round_.top_stack_count == 4
        elif kind == "play":
            if round_.top_rank is None:
                assert not self.opening_required or rank == "3"
            else:
                assert len(cards) == round_.top_count
                assert self.RANKS.index(rank) >= self.RANKS.index(round_.top_rank)
        elif kind == "bomb":
            bomb = Card.from_json(action["card"])
            assert round_.top_rank is not None and bomb.rank == "2" and bomb in round_.hands[player]
        else:
            assert kind == "pass" and round_.top_rank is not None

    def apply(self, round_, player, action):
        if round_.phase not in ("play", "completion"):
            round_.apply_action(player, action)
            return
        if not self.started:
            starter = (next(p for p in round_.players if card("3") in round_.hands[p])
                       if round_.first_round else next(p for p, role in round_.roles.items() if role == 4))
            assert round_.phase == "play" and player == starter
            assert (round_.top_rank, round_.top_count, round_.top_stack_count) == (None, 0, 0)
            self.started = True
        for offered in round_.legal_actions(player):
            self.assert_legal(round_, player, offered)
        self.assert_legal(round_, player, action)
        top = (round_.top_rank, round_.top_count, round_.top_stack_count, round_.last_player)
        kind = action["type"]
        clear_owner = None
        opens_window = False
        advances = False
        if round_.phase == "play":
            if kind == "play":
                rank, count = action["cards"][0]["rank"], len(action["cards"])
                equal = rank == top[0]
                stack = top[2] + count if equal else count
                top = (rank, count, stack, player)
                self.opening_required = False
                if stack == 4:
                    clear_owner = player
                else:
                    self.pending = (player, equal)
                    opens_window = True
            elif kind == "bomb":
                clear_owner = player
            else:
                self.pending = (player, False)
                opens_window = True
        else:
            assert self.respondents and player == self.respondents.pop(0)
            if kind == "complete":
                clear_owner = player
            else:
                advances = not self.respondents

        round_.apply_action(player, action)
        if round_.done:
            assert round_._actor() is None and len(round_.active) == 0
            return
        seats = round_.players
        if opens_window:
            offset = seats.index(player)
            order = seats[offset:] + seats[:offset]
            self.respondents = [p for p in order if p in round_.active]
        if advances:
            previous, skip = self.pending
            offset = seats.index(previous) + 1
            order = seats[offset:] + seats[:offset]
            eligible = [p for p in order if p in round_.active]
            destination = eligible[int(skip)]
            if order.index(top[3]) <= order.index(destination):
                clear_owner = top[3]
            else:
                assert round_.phase == "play" and round_._actor() == destination
            self.pending = None
        if clear_owner is not None:
            offset = seats.index(clear_owner)
            order = seats[offset:] + seats[:offset]
            expected = next(p for p in order if p in round_.active)
            assert round_.phase == "play" and round_._actor() == expected
            top = (None, 0, 0, None)
            self.pending, self.respondents = None, []
        elif self.respondents:
            assert round_.phase == "completion" and round_._actor() == self.respondents[0]
        assert (round_.top_rank, round_.top_count, round_.top_stack_count, round_.last_player) == top


def exercise_seeded_games(seed_count):
    """Check whole games against gameplay rules and conservation/placement invariants."""
    total_actions = 0
    deck_counts = Counter(standard_deck())
    for seed in range(seed_count):
        game = Game(PLAYERS, rng=random.Random(seed))
        clients = {p: RandomClient(random.Random(seed * 4 + i)) for i, p in enumerate(PLAYERS)}
        for _ in range(3):
            round_ = game.start_round()
            gameplay = GameplayChecks(round_)
            for _ in range(2000):
                if round_.done:
                    break
                player = round_._actor()
                message = json.loads(json.dumps(round_.message_for(player)))
                assert message["request"]["legal_actions"]
                action = json.loads(json.dumps(clients[player].respond(message)))
                gameplay.apply(round_, player, action)
                total_actions += 1
                cards = [c for hand in round_.hands.values() for c in hand]
                cards += [c for pile in getattr(round_, "piles", []) for c in pile]
                for event in round_.events:
                    if event["type"] in ("opening_play", "played", "completed"):
                        cards.extend(Card.from_json(c) for c in event["cards"])
                    elif event["type"] == "bombed":
                        cards.append(Card.from_json(event["card"]))
                assert Counter(cards) == deck_counts, (seed, action)
                if round_.phase in ("play", "completion"):
                    assert round_._actor() in round_.active
                    assert all(any(c.rank != "2" for c in round_.hands[p]) for p in round_.active)
            assert round_.done, (seed, "round failed to terminate")
            assert set(round_.rankings) == set(PLAYERS) and set(round_.rankings.values()) == {1, 2, 3, 4}
            assert sum(e["type"] == "round_finished" for e in round_.events) == 1
            penalties = [e["player"] for e in round_.events if e["type"] == "twos_announced" or (e["type"] == "player_done" and e.get("bomb_finish"))]
            order = sorted(PLAYERS, key=round_.rankings.get)
            if penalties:
                assert order[-len(penalties):] == penalties
            roles = game.finish_round()
            assert roles == round_.rankings
    return total_actions


def test_seeded_multiround_games_conserve_cards_and_obey_penalty_order():
    exercise_seeded_games(8)


# Keep the tests runnable with either ``python -m unittest`` (no third-party
# dependencies) or pytest when a project later chooses to add it.
def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            suite.addTest(unittest.FunctionTestCase(test))
    return suite


if __name__ == "__main__":
    unittest.main()
