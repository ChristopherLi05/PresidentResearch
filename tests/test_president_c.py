"""Compile the C port and compare it with the Python engine, using only stdlib.

Requires GCC or Clang (or set CC to the compiler executable). Other Python
tests remain usable without a C toolchain: this class is skipped in that case.
"""

import ctypes as ct
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from src.president import RANKS, SUITS, Round, standard_deck
from src.president_c import _Action as Action


ROOT = Path(__file__).resolve().parents[1]
KINDS = ["choose_pile", "trade_request", "trade_return", "play", "pass",
         "bomb", "complete", "decline_completion"]
PLAYERS = ["a", "b", "c", "d"]


def card_id(card):
    return RANKS.index(card.rank) * 4 + SUITS.index(card.suit)


def card_json(value):
    return {"rank": RANKS[value // 4], "suit": SUITS[value % 4]}


def action_json(action):
    kind = KINDS[action.type]
    result = {"type": kind}
    if kind == "choose_pile":
        result["pile"] = action.pile
    elif kind == "trade_request":
        result["rank"] = RANKS[action.rank]
    elif kind in ("bomb", "trade_return"):
        result["card"] = card_json(action.cards[0])
    elif kind in ("play", "complete"):
        result["cards"] = [card_json(c) for c in action.cards[:action.count]]
    return result


OBSERVE = ct.CFUNCTYPE(None, ct.c_void_p, ct.c_void_p)
RESPOND = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_void_p, ct.POINTER(Action))


class Client(ct.Structure):
    _fields_ = [("context", ct.c_void_p), ("observe", OBSERVE), ("respond", RESPOND)]


class PresidentCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = os.environ.get("CC") or shutil.which("gcc") or shutil.which("clang")
        if not compiler:
            raise unittest.SkipTest("GCC or Clang required for C engine tests")
        cls.temp = tempfile.TemporaryDirectory(prefix="president-c-")
        cls.addClassCleanup(cls.temp.cleanup)
        library = Path(cls.temp.name) / ("president.dll" if os.name == "nt" else "libpresident.so")
        command = [compiler, "-std=c11", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
                   "-O2", "-shared", str(ROOT / "src/president.c"), "-o", str(library)]
        if os.name != "nt":
            command.insert(1, "-fPIC")
        subprocess.run(command, check=True, capture_output=True, text=True)
        cls.lib = ct.CDLL(str(library))
        # Windows cannot remove a loaded DLL; unload after all callbacks/rounds.
        if os.name == "nt":
            import _ctypes
            cls.addClassCleanup(_ctypes.FreeLibrary, cls.lib._handle)
        signatures = {
            "round_create": (ct.c_int, [ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_char_p),
                ct.c_int, ct.POINTER(ct.c_int), ct.POINTER(ct.c_int), ct.c_size_t, ct.c_uint64]),
            "round_destroy": (None, [ct.c_void_p]),
            "round_done": (ct.c_int, [ct.c_void_p]),
            "round_actor": (ct.c_int, [ct.c_void_p]),
            "round_legal_actions": (ct.c_size_t, [ct.c_void_p, ct.c_int, ct.POINTER(Action), ct.c_size_t]),
            "round_apply": (ct.c_int, [ct.c_void_p, ct.c_int, ct.POINTER(Action), ct.POINTER(ct.c_size_t)]),
            "round_message_json": (ct.c_void_p, [ct.c_void_p, ct.c_int]),
            "round_events_json": (ct.c_void_p, [ct.c_void_p, ct.c_size_t]),
            "round_event_count": (ct.c_size_t, [ct.c_void_p]),
            "round_history_count": (ct.c_size_t, [ct.c_void_p]),
            "round_history_at": (ct.c_int, [ct.c_void_p, ct.c_size_t, ct.POINTER(ct.c_int), ct.POINTER(Action)]),
            "round_rankings": (ct.c_int, [ct.c_void_p, ct.POINTER(ct.c_int)]),
            "free": (None, [ct.c_void_p]),
            "status_string": (ct.c_char_p, [ct.c_int]),
            "card": (ct.c_int, [ct.c_char_p, ct.c_char_p]),
            "game_create": (ct.c_int, [ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_char_p), ct.c_uint64]),
            "game_destroy": (None, [ct.c_void_p]),
            "game_start_round": (ct.c_int, [ct.c_void_p, ct.POINTER(ct.c_int), ct.c_size_t]),
            "game_round": (ct.c_void_p, [ct.c_void_p]),
            "game_round_number": (ct.c_size_t, [ct.c_void_p]),
            "game_finish_round": (ct.c_int, [ct.c_void_p, ct.POINTER(ct.c_int)]),
            "game_run": (ct.c_int, [ct.c_void_p, ct.POINTER(Client), ct.c_size_t]),
        }
        for name, (restype, argtypes) in signatures.items():
            function = getattr(cls.lib, "president_" + name)
            function.restype, function.argtypes = restype, argtypes

    def names(self, players=PLAYERS):
        return (ct.c_char_p * 4)(*(p.encode() for p in players))

    def read_json(self, pointer):
        self.assertTrue(pointer, "JSON allocation failed")
        try:
            return json.loads(ct.string_at(pointer))
        finally:
            self.lib.president_free(pointer)

    def state(self, native, player):
        return self.read_json(self.lib.president_round_message_json(native, player))

    def make_round(self, deck=None, roles=None, players=PLAYERS, seed=42):
        native = ct.c_void_p()
        cards = None if deck is None else (ct.c_int * len(deck))(*(card_id(c) for c in deck))
        native_roles = None if roles is None else (ct.c_int * 4)(*(roles[p] for p in players))
        self.assertEqual(self.lib.president_round_create(ct.byref(native), self.names(players),
            roles is None, native_roles, cards, 0 if deck is None else len(deck), seed), 0)
        self.addCleanup(self.lib.president_round_destroy, native)
        return native

    def offered(self, native, p):
        n = self.lib.president_round_legal_actions(native, p, None, 0)
        out = (Action * n)()
        self.assertEqual(self.lib.president_round_legal_actions(native, p, out, n), n)
        return out

    def reject_unchanged(self, native, p, action):
        before = [self.state(native, i) for i in range(4)]
        history = self.lib.president_round_history_count(native)
        start = ct.c_size_t(99999)
        pointer = None if action is None else ct.byref(action)
        self.assertEqual(self.lib.president_round_apply(native, p, pointer, ct.byref(start)), 2)
        self.assertEqual(start.value, 99999)
        self.assertEqual([self.state(native, i) for i in range(4)], before)
        self.assertEqual(self.lib.president_round_history_count(native), history)

    def test_construction_validation_and_seeded_shuffle(self):
        names = self.names()
        deck = (ct.c_int * 52)(*range(52))
        good_roles = (ct.c_int * 4)(1, 2, 3, 4)
        for first, roles, cards, count in [
            (0, None, deck, 52), (1, good_roles, deck, 52),
            (0, (ct.c_int * 4)(1, 1, 3, 4), deck, 52),
            (0, (ct.c_int * 4)(0, 2, 3, 4), deck, 52),
            (1, None, deck, 51), (1, None, None, 52),
            (1, None, (ct.c_int * 52)(*([0] * 52)), 52),
            (1, None, (ct.c_int * 52)(*range(1, 53)), 52),
        ]:
            result = ct.c_void_p(1)
            self.assertEqual(self.lib.president_round_create(ct.byref(result), names,
                first, roles, cards, count, 1), 1)
            self.assertIsNone(result.value)
        result = ct.c_void_p()
        self.assertEqual(self.lib.president_round_create(ct.byref(result), self.names(["a"] * 4),
            1, None, deck, 52, 0), 1)
        for seed in (0, 1, 2**64 - 1):
            a, b = self.make_round(seed=seed), self.make_round(seed=seed)
            self.assertEqual(self.state(a, 0), self.state(b, 0))
        self.assertEqual(self.lib.president_card(b"10", b"H"), 30)
        self.assertEqual(self.lib.president_card(b"bad", b"H"), -1)
        self.assertEqual(self.lib.president_card(None, b"H"), -1)

    def test_four_card_opening_combinations_and_rejected_actions(self):
        deck = standard_deck()
        for i in (1, 2, 3):
            deck[i], deck[i * 4] = deck[i * 4], deck[i]
        native = self.make_round(deck)
        py = Round(PLAYERS, first_round=True, deck=deck)
        self.assertEqual(self.state(native, 0), py.message_for("a"))
        offered = self.offered(native, 0)
        self.assertEqual(len(offered), 15)
        self.assertEqual([action_json(a) for a in offered], py.legal_actions("a"))
        for action in [None, Action(type=99), Action(type=4), Action(type=3, count=-1),
                       Action(type=3, count=5), Action(type=3, count=2, cards=(0, 0, 0, 0)),
                       Action(type=3, count=1, cards=(52, 0, 0, 0)),
                       Action(type=3, count=1, cards=(-1, 0, 0, 0)),
                       Action(type=3, count=1, cards=(8, 0, 0, 0))]:
            self.reject_unchanged(native, 0, action)
        for p in (-1, 1, 4):
            self.reject_unchanged(native, p, offered[0])
        # Input order need not match combinations() order.
        action = Action(type=3, count=4, cards=(3, 2, 1, 0))
        self.assertEqual(self.lib.president_round_apply(native, 0, ct.byref(action), None), 0)
        py.apply_action("a", action_json(action))
        self.assertEqual(self.state(native, 0), py.message_for("a"))
        action.cards[0] = 51
        recorded, p = Action(), ct.c_int()
        self.assertEqual(self.lib.president_round_history_at(native, 0, ct.byref(p), ct.byref(recorded)), 0)
        self.assertEqual(action_json(recorded), py.turn_history[0]["action"])

    def test_json_escaping_and_snapshot_ownership(self):
        players = ['a"\\\n', "b\t\x01", "café", "雪"]
        native = self.make_round(standard_deck(), players=players)
        py = Round(players, first_round=True, deck=standard_deck())
        saved = self.lib.president_round_message_json(native, 0)
        self.assertEqual(json.loads(ct.string_at(saved)), py.message_for(players[0]))
        original = ct.string_at(saved)
        action = self.offered(native, 0)[0]
        self.assertEqual(self.lib.president_round_apply(native, 0, ct.byref(action), None), 0)
        self.assertEqual(ct.string_at(saved), original)
        self.lib.president_free(saved)

    def test_python_parity_over_full_multiround_games(self):
        observed_events, observed_phases = set(), set()
        total = 0
        for seed in range(12):
            rng = random.Random(seed)
            roles = None
            for round_index in range(3):
                deck = standard_deck()
                rng.shuffle(deck)
                native = self.make_round(deck, roles)
                py = Round(PLAYERS, first_round=roles is None, roles=roles, deck=deck)
                checked = set()
                for turn in range(3000):
                    if py.done:
                        break
                    p = PLAYERS.index(py._actor())
                    with self.subTest(seed=seed, round=round_index, turn=turn, phase=py.phase):
                        self.assertEqual(self.lib.president_round_actor(native), p)
                        self.assertEqual(self.state(native, p), py.message_for(PLAYERS[p]))
                        # Also verify a non-actor never receives another hand/request.
                        observer = (p + 1) % 4
                        self.assertEqual(self.state(native, observer), py.message_for(PLAYERS[observer]))
                        offered = self.offered(native, p)
                        self.assertEqual([action_json(a) for a in offered], py.legal_actions(PLAYERS[p]))
                        observed_phases.add(py.phase)
                        if py.phase not in checked:
                            self.reject_unchanged(native, p, Action(type=99))
                            checked.add(py.phase)
                        a = rng.choice(offered)
                        # Sometimes reorder multi-card input: validation must not
                        # depend on matching a generated combination byte-for-byte.
                        if a.type in (3, 6) and a.count > 1 and turn % 2:
                            a.cards[:a.count] = a.cards[:a.count][::-1]
                        expected = py.apply_action(PLAYERS[p], action_json(a))
                        start = ct.c_size_t()
                        self.assertEqual(self.lib.president_round_apply(native, p, ct.byref(a), ct.byref(start)), 0)
                        self.assertEqual(self.read_json(self.lib.president_round_events_json(native, start.value)), expected)
                        observed_events.update(e["type"] for e in expected)
                        recorded, recorded_player = Action(), ct.c_int()
                        self.assertEqual(self.lib.president_round_history_at(native, turn, ct.byref(recorded_player), ct.byref(recorded)), 0)
                        self.assertEqual(recorded_player.value, p)
                        self.assertEqual(action_json(recorded), py.turn_history[-1]["action"])
                        total += 1
                self.assertTrue(py.done)
                self.assertTrue(self.lib.president_round_done(native))
                self.assertEqual(self.lib.president_round_actor(native), -1)
                for p in range(4):
                    self.assertEqual(self.state(native, p), py.message_for(PLAYERS[p]))
                    self.assertEqual(len(self.offered(native, p)), 0)
                self.reject_unchanged(native, 0, Action(type=4))
                rankings = (ct.c_int * 4)()
                self.assertEqual(self.lib.president_round_rankings(native, rankings), 0)
                self.assertEqual(list(rankings), [py.rankings[p] for p in PLAYERS])
                roles = py.rankings
        self.assertGreater(total, 5000)
        self.assertEqual(observed_phases, {"draft", "trade_request", "trade_return", "play", "completion"})
        self.assertTrue({"completed", "bombed", "twos_announced", "board_cleared", "round_finished",
                         "trade_requested", "trade_completed"} <= observed_events)

    def test_game_callbacks_limits_and_automatic_role_carryover(self):
        game = ct.c_void_p()
        self.assertEqual(self.lib.president_game_create(ct.byref(game), self.names(), 42), 0)
        self.addCleanup(self.lib.president_game_destroy, game)
        observations, errors = [], []
        rng = random.Random(4)

        @OBSERVE
        def observe(context, message):
            try:
                observations.append(json.loads(ct.string_at(message)))
                # Deliberately edit the observer's copy, including the actor's.
                ct.memset(message, ord("!"), 1)
            except Exception as error:
                errors.append(error)

        @RESPOND
        def respond(context, message, out):
            try:
                state = json.loads(ct.string_at(message))
                native = self.lib.president_game_round(game)
                p = PLAYERS.index(state["player"])
                offered = self.offered(native, p)
                out[0] = rng.choice(offered)
                return 0
            except Exception as error:
                errors.append(error)
                return 6

        clients = (Client * 4)(*(Client(None, observe, respond) for _ in range(4)))
        placements = (ct.c_int * 4)()
        self.assertEqual(self.lib.president_game_finish_round(game, placements), 4)
        self.assertEqual(self.lib.president_game_run(game, clients, 0), 5)
        self.assertEqual(len(observations), 4)
        native = self.lib.president_game_round(game)
        self.assertEqual(self.lib.president_game_start_round(game, None, 0), 4)
        self.assertEqual(self.lib.president_game_round(game), native)
        # A one-action limit must succeed when that exact action finishes.
        for _ in range(3000):
            result = self.lib.president_game_run(game, clients, 1)
            self.assertIn(result, (0, 5))
            if result == 0:
                break
        else:
            self.fail("game did not finish")
        self.assertFalse(errors)
        self.assertTrue(all(state["phase"] == "finished" for state in observations[-4:]))
        self.assertEqual(self.lib.president_game_finish_round(game, placements), 0)
        self.assertEqual(sorted(placements), [1, 2, 3, 4])
        self.assertEqual(self.lib.president_game_run(game, clients, 0), 0)
        self.assertEqual(self.lib.president_game_start_round(game, (ct.c_int * 1)(0), 1), 1)
        self.assertEqual(self.lib.president_game_round(game), native)
        self.assertEqual(self.lib.president_game_round_number(game), 1)
        self.assertEqual(self.lib.president_game_start_round(game, None, 0), 0)
        self.assertEqual(self.lib.president_game_round_number(game), 2)
        next_state = self.state(self.lib.president_game_round(game), 0)
        self.assertEqual(next_state["roles"], dict(zip(PLAYERS, placements)))
        self.assertEqual(next_state["phase"], "draft")
        self.assertEqual(self.lib.president_game_run(game, clients, 3000), 0)
        self.assertFalse(errors)

    def test_python_wrapper_with_existing_agents_and_seed_compatibility(self):
        from src.agent import BasicAgent, RandomAgent
        from src.president import Game as PythonGame
        from src.president_c import Game as NativeGame

        with patch("src.president_c._load_library", return_value=self.lib):
            for policy in (BasicAgent, RandomAgent):
                python_game = PythonGame(PLAYERS, rng=random.Random(7))
                with NativeGame(PLAYERS, rng=random.Random(7)) as native_game:
                    clients = [{p: policy(random.Random(i)) if policy is RandomAgent else policy()
                                for i, p in enumerate(PLAYERS)} for _ in range(2)]
                    previous = None
                    for _ in range(3):
                        py, native = python_game.start_round(), native_game.start_round()
                        self.assertEqual(native.message_for("a"), py.message_for("a"))
                        before_rng = native_game.rng.getstate()
                        with self.assertRaisesRegex(RuntimeError, "not finished"):
                            native_game.start_round()
                        self.assertEqual(native_game.rng.getstate(), before_rng)
                        python_game.run(clients[0], max_actions=3000)
                        native_game.run(clients[1], max_actions=3000)
                        self.assertEqual(native.rankings, py.rankings)
                        self.assertEqual(native.turn_history, py.turn_history)
                        self.assertEqual(native.events, py.events)
                        for p in PLAYERS:
                            self.assertEqual(native.message_for(p), py.message_for(p))
                        self.assertEqual(native_game.finish_round(), python_game.finish_round())
                        if previous is not None:
                            self.assertTrue(previous.done)  # old rounds remain valid
                            previous.close()
                        previous = native

    def test_python_wrapper_validation_and_lifetime(self):
        from src.president import IllegalAction
        from src.president_c import Round as NativeRound

        with patch("src.president_c._load_library", return_value=self.lib):
            with NativeRound(PLAYERS, first_round=True, deck=standard_deck()) as native:
                for action in [None, [], {}, {"type": []}, {"type": "play", "cards": []},
                               {"type": "play", "cards": [None]},
                               {"type": "play", "cards": [{"rank": [], "suit": "C"}]},
                               {"type": "play", "cards": [{"rank": "3"}]},
                               {"type": "choose_pile", "pile": 2**40},
                               {"type": "choose_pile", "pile": True},
                               {"type": "trade_request", "rank": []},
                               {"type": "bomb", "card": None}]:
                    before = native.message_for("a")
                    with self.assertRaises(IllegalAction):
                        native.apply_action("a", action)
                    self.assertEqual(native.message_for("a"), before)
                    self.assertEqual(native.turn_history, [])
                self.assertEqual(native.legal_actions("missing"), [])
                with self.assertRaises(KeyError):
                    native.message_for("missing")
                action = native.legal_actions("a")[0]
                action["metadata"] = {"tag": [1]}
                native.apply_action("a", action)
                action["metadata"]["tag"].append(2)
                self.assertEqual(native.turn_history[0]["action"]["metadata"], {"tag": [1]})
                events = native.events
                events.clear()
                self.assertTrue(native.events)
            native.close()  # idempotent
            with self.assertRaisesRegex(RuntimeError, "closed"):
                native.message_for("a")
            with self.assertRaisesRegex(RuntimeError, "closed"):
                native.apply_action("a", {"type": "pass"})
            for players in (["a"] * 4, ["a", "b", "c", "d\0bad"], [1, 2, 3, 4]):
                with self.assertRaises(ValueError):
                    NativeRound(players, first_round=True)
            with self.assertRaises(ValueError):
                NativeRound(PLAYERS, first_round=False, roles={p: 1 for p in PLAYERS})
            with self.assertRaises(ValueError):
                NativeRound(PLAYERS, first_round=True, deck=[])


if __name__ == "__main__":
    unittest.main()
