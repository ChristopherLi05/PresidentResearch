"""Python interface to the C engine, using the standard-library ctypes module.

Build once with ``python -m src.president_c --build``, then import Round/Game
from this module in place of src.president. The gameplay runs in C; the Python
Game adapter retains the existing client notification/snapshot behavior.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
import ctypes as ct
from functools import lru_cache
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import weakref

from src.president import (
    Card, Game as _PythonGame, IllegalAction, RANKS, RANK_VALUE, ROLE_NAMES,
    SUITS, standard_deck,
)

__all__ = ["Card", "Game", "IllegalAction", "Round", "RANKS", "RANK_VALUE",
           "ROLE_NAMES", "SUITS", "standard_deck", "build_library"]

_ROOT = Path(__file__).resolve().parents[1]
_KINDS = ("choose_pile", "trade_request", "trade_return", "play", "pass",
          "bomb", "complete", "decline_completion")


class _Action(ct.Structure):
    _fields_ = [("type", ct.c_int), ("pile", ct.c_int), ("rank", ct.c_int),
               ("count", ct.c_int), ("cards", ct.c_int * 4)]


def _library_path():
    if os.environ.get("PRESIDENT_LIBRARY"):
        return Path(os.environ["PRESIDENT_LIBRARY"]).resolve()
    filename = ("president.dll" if os.name == "nt" else
                "libpresident.dylib" if sys.platform == "darwin" else "libpresident.so")
    return _ROOT / "build" / filename


def build_library(*, compiler=None, output=None):
    """Build with GCC/Clang. CC may name an alternative compiler executable.

    Run before constructing any Round objects; Windows locks loaded DLLs.
    No packages are downloaded or installed.
    """
    compiler = compiler or os.environ.get("CC") or shutil.which("gcc") or shutil.which("clang")
    if not compiler:
        raise RuntimeError("GCC or Clang is required; install one or set CC to its executable")
    target = Path(output).resolve() if output else _library_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [str(compiler), "-std=c11", "-O2", "-Wall", "-Wextra", "-Wpedantic",
               "-Werror", "-shared", str(_ROOT / "src/president.c"), "-o", str(target)]
    if os.name != "nt":
        command.insert(1, "-fPIC")
    if sys.platform == "darwin":
        command[command.index("-shared")] = "-dynamiclib"
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"C engine build failed:\n{error.stderr}") from error
    return target


@lru_cache(maxsize=None)
def _load_library(path):
    if not Path(path).is_file():
        raise RuntimeError("C engine has not been built. Run: python -m src.president_c --build")
    lib = ct.CDLL(str(path))
    signatures = {
        "round_create": (ct.c_int, [ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_char_p),
            ct.c_int, ct.POINTER(ct.c_int), ct.POINTER(ct.c_int), ct.c_size_t, ct.c_uint64]),
        "round_destroy": (None, [ct.c_void_p]),
        "round_done": (ct.c_int, [ct.c_void_p]),
        "round_actor": (ct.c_int, [ct.c_void_p]),
        "round_legal_actions": (ct.c_size_t, [ct.c_void_p, ct.c_int, ct.POINTER(_Action), ct.c_size_t]),
        "round_apply": (ct.c_int, [ct.c_void_p, ct.c_int, ct.POINTER(_Action), ct.POINTER(ct.c_size_t)]),
        "round_message_json": (ct.c_void_p, [ct.c_void_p, ct.c_int]),
        "round_events_json": (ct.c_void_p, [ct.c_void_p, ct.c_size_t]),
        "round_rankings": (ct.c_int, [ct.c_void_p, ct.POINTER(ct.c_int)]),
        "status_string": (ct.c_char_p, [ct.c_int]),
        "free": (None, [ct.c_void_p]),
    }
    for name, (restype, argtypes) in signatures.items():
        function = getattr(lib, "president_" + name)
        function.restype, function.argtypes = restype, argtypes
    return lib


def _check(lib, status):
    if status == 0:
        return
    message = lib.president_status_string(status).decode()
    if status == 1:
        raise ValueError(message)
    if status == 2:
        raise IllegalAction(message)
    if status == 3:
        raise MemoryError(message)
    raise RuntimeError(message)


def _card_id(value):
    if not isinstance(value, Mapping):
        raise IllegalAction("invalid card")
    try:
        return RANKS.index(value["rank"]) * 4 + SUITS.index(value["suit"])
    except (KeyError, ValueError, TypeError) as error:
        raise IllegalAction("invalid card") from error


def _encode_action(value):
    if not isinstance(value, Mapping):
        raise IllegalAction("action must be an object")
    try:
        kind = value.get("type")
        action = _Action(type=_KINDS.index(kind))
    except ValueError as error:
        raise IllegalAction("unknown action") from error
    if kind == "choose_pile":
        pile = value.get("pile")
        if type(pile) is not int or not 0 <= pile < 4:
            raise IllegalAction("choose an available pile")
        action.pile = pile
    elif kind == "trade_request":
        try:
            action.rank = RANKS.index(value.get("rank"))
        except ValueError as error:
            raise IllegalAction("request a valid rank") from error
    elif kind in ("trade_return", "bomb"):
        action.cards[0] = _card_id(value.get("card"))
    elif kind in ("play", "complete"):
        cards = value.get("cards")
        if not isinstance(cards, list) or not 1 <= len(cards) <= 4:
            raise IllegalAction("cards must be a nonempty list of up to four cards")
        action.count = len(cards)
        action.cards[:action.count] = [_card_id(c) for c in cards]
    return action


def _decode_action(action):
    def card(value):
        return {"rank": RANKS[value // 4], "suit": SUITS[value % 4]}

    kind = _KINDS[action.type]
    result = {"type": kind}
    if kind == "choose_pile":
        result["pile"] = action.pile
    elif kind == "trade_request":
        result["rank"] = RANKS[action.rank]
    elif kind in ("trade_return", "bomb"):
        result["card"] = card(action.cards[0])
    elif kind in ("play", "complete"):
        result["cards"] = [card(c) for c in action.cards[:action.count]]
    return result


class Round:
    """C-backed Round with the public dictionary API of src.president.Round.

    Use close() or a context manager for prompt cleanup; garbage collection is
    also safe. State properties return copies and cannot be assigned to set up
    artificial mid-round states. Do not use a single round concurrently.
    """

    def __init__(self, players, *, first_round, roles=None, deck=None, rng=None):
        players = list(players)
        if (len(players) != 4 or any(not isinstance(p, str) or "\0" in p for p in players)
                or len(set(players)) != 4):
            raise ValueError("exactly four distinct string player ids without NUL characters are required")
        if bool(first_round) == (roles is not None):
            raise ValueError("roles are required exactly for later rounds")
        if roles is not None and (not isinstance(roles, Mapping) or set(roles) != set(players)
                or any(type(v) is not int for v in roles.values()) or set(roles.values()) != {1, 2, 3, 4}):
            raise ValueError("roles must assign ranks 1..4 to these players")
        cards = list(deck) if deck is not None else standard_deck()
        if any(not isinstance(c, Card) for c in cards) or len(cards) != 52 or len(set(cards)) != 52:
            raise ValueError("deck must contain each of the 52 cards once")
        # Use Python's shuffle at the boundary to preserve random.Random seed
        # and RNG-consumption compatibility with the existing engine.
        if deck is None:
            (rng or random.Random()).shuffle(cards)
        self._lib = _load_library(str(_library_path()))
        self._players = tuple(players)
        self._roles = dict(roles or {})
        self.first_round = bool(first_round)
        self._turn_history = []
        self._event_cache = []
        self._pointer = ct.c_void_p()
        names = (ct.c_char_p * 4)(*(p.encode("utf-8") for p in players))
        native_roles = None if roles is None else (ct.c_int * 4)(*(roles[p] for p in players))
        native_deck = (ct.c_int * 52)(*(RANK_VALUE[c.rank] * 4 + SUITS.index(c.suit) for c in cards))
        _check(self._lib, self._lib.president_round_create(ct.byref(self._pointer), names,
            bool(first_round), native_roles, native_deck, 52, 0))
        self._finalizer = weakref.finalize(self, self._lib.president_round_destroy, self._pointer)
        self._event_cache = self._json(self._lib.president_round_events_json(self._pointer, 0))

    def _require_open(self):
        if not self._finalizer.alive:
            raise RuntimeError("round is closed")

    def close(self):
        self._finalizer()

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *exception):
        self.close()

    def _json(self, pointer):
        if not pointer:
            raise MemoryError("C engine could not allocate a JSON snapshot")
        try:
            return json.loads(ct.string_at(pointer))
        finally:
            self._lib.president_free(pointer)

    @property
    def players(self):
        return list(self._players)

    @property
    def roles(self):
        return self._roles.copy()

    @property
    def done(self):
        self._require_open()
        return bool(self._lib.president_round_done(self._pointer))

    @property
    def phase(self):
        return self.message_for(self._players[0])["phase"]

    @property
    def rankings(self):
        self._require_open()
        if not self.done:
            return None
        values = (ct.c_int * 4)()
        _check(self._lib, self._lib.president_round_rankings(self._pointer, values))
        return dict(zip(self._players, values))

    @property
    def events(self):
        self._require_open()
        return deepcopy(self._event_cache)

    @property
    def turn_history(self):
        self._require_open()
        return deepcopy(self._turn_history)

    @property
    def hands(self):
        return {p: [Card.from_json(c) for c in self.message_for(p)["hand"]] for p in self._players}

    def _actor(self):
        self._require_open()
        seat = self._lib.president_round_actor(self._pointer)
        return None if seat < 0 else self._players[seat]

    def message_for(self, player):
        self._require_open()
        try:
            seat = self._players.index(player)
        except ValueError as error:
            raise KeyError(player) from error
        return self._json(self._lib.president_round_message_json(self._pointer, seat))

    def legal_actions(self, player):
        self._require_open()
        if player not in self._players:
            return []
        seat = self._players.index(player)
        count = self._lib.president_round_legal_actions(self._pointer, seat, None, 0)
        actions = (_Action * count)()
        self._lib.president_round_legal_actions(self._pointer, seat, actions, count)
        return [_decode_action(a) for a in actions]

    def apply_action(self, player, action):
        self._require_open()
        if self.done or self._actor() != player:
            raise IllegalAction("it is not this player's turn/request")
        native = _encode_action(action)
        recorded = {"player": player, "action": deepcopy(dict(action))}
        start = ct.c_size_t()
        _check(self._lib, self._lib.president_round_apply(self._pointer, self._players.index(player),
            ct.byref(native), ct.byref(start)))
        self._turn_history.append(recorded)
        events = self._json(self._lib.president_round_events_json(self._pointer, start.value))
        self._event_cache.extend(events)
        return deepcopy(events)

    def _public_message(self, events):
        message = self.message_for(self._players[0])
        for key in ("player", "hand", "request"):
            del message[key]
        message["events"] = events
        return message

    def _private_message(self, player, public):
        private = self.message_for(player)
        return {**public, **{key: private[key] for key in ("player", "hand", "request")}}


class Game(_PythonGame):
    """Existing Python client adapter with all round transitions executed in C."""

    def start_round(self, *, deck=None):
        if self.round is not None and not self.round.done:
            raise RuntimeError("current round has not finished")
        roles = dict(self.round.rankings) if self.round is not None else self.roles
        next_round = Round(self.players, first_round=self.round_number == 0,
                           roles=roles, deck=deck, rng=self.rng)
        self.roles = roles
        self.round = next_round
        self.round_number += 1
        return self.round

    def _notify_clients(self, clients, history=None):
        # Reuse the original notification implementation and its frozen public
        # data contract. Retain one append-only source for SnapshotHistory.
        from src.snapshots import SnapshotHistory, freeze

        history = history if history is not None else SnapshotHistory()
        public = self.round._public_message(None)
        public = {key: freeze(value) for key, value in public.items() if key != "events"}
        public["events"] = history.snapshot(self.round._event_cache)
        request = None
        for player, client in clients.items():
            message = self.round._private_message(player, public)
            if message["request"] is not None:
                request = {**message, "hand": deepcopy(message["hand"]),
                           "request": deepcopy(message["request"])}
            client.observe(message)
        return request

    def close(self):
        if self.round is not None:
            self.round.close()

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="compile the C shared library")
    parser.add_argument("--compiler", help="GCC/Clang executable (default: CC or PATH)")
    parser.add_argument("--output", type=Path, help="shared library output path")
    args = parser.parse_args()
    if not args.build:
        parser.error("specify --build to compile the library")
    print(build_library(compiler=args.compiler, output=args.output))


if __name__ == "__main__":
    main()
