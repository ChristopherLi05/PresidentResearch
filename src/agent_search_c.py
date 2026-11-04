"""ctypes bridge for native Agent4 rollouts. Build: python -m src.agent_search_c --build.

Python samples worlds and supplies weights; C performs all moves and scoring
inside each batch without JSON conversion or callbacks into Python.
"""
import argparse
import ctypes as ct
from functools import lru_cache
from math import isfinite
import os
from pathlib import Path
import shutil
import subprocess
import sys

from src.president import RANKS, SUITS
from src.president_c import _Action, _encode_action

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ("CONTROL_RESERVE_WEIGHT", "GROUP_WEIGHT", "CARD_WEIGHT", "SINGLE_WEIGHT",
           "HAND_CONTROL_WEIGHT", "BOMB_GROUP_PENALTY", "BOMB_COVER_PENALTY", "BOMB_VALUE",
           "LEAD_CONTROL_WEIGHT", "BOMB_STRANDING_PENALTY", "PASS_COST", "PASS_DANGER_WEIGHT",
           "CLEAR_COST", "OPPONENT_FINISH_PENALTY", "NEXT_RESPONSE_PENALTY", "NEXT_FINISH_PENALTY",
           "NEXT_COMPLETION_PENALTY", "NEXT_BOMB_PENALTY")
PHASES = ("draft", "trade_request", "trade_return", "play", "completion", "finished")


class Position(ct.Structure):
    _fields_ = [("cards", (ct.c_int * 52) * 4), ("sizes", ct.c_int * 4),
                ("active", ct.c_int * 4), ("finish", ct.c_int * 4), ("finish_count", ct.c_int),
                ("bombs", ct.c_int * 4), ("bomb_count", ct.c_int)] + [
        (name, ct.c_int) for name in ("top_rank", "top_count", "stack", "last_player", "turn",
                                     "opened", "first_round", "phase")] + [
        ("completion", ct.c_int * 4)] + [(name, ct.c_int) for name in (
            "completion_count", "cursor", "pending_player", "pending_skip")] + [("rankings", ct.c_int * 4)]


def library_path():
    name = "agent_search.dll" if os.name == "nt" else "libagent_search.dylib" if sys.platform == "darwin" else "libagent_search.so"
    return ROOT / "build" / name


def build_library(*, output=None, compiler=None):
    compiler = compiler or os.environ.get("CC") or shutil.which("gcc") or shutil.which("clang")
    if not compiler:
        raise RuntimeError("GCC or Clang is required to build native search")
    target = Path(output).resolve() if output else library_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [str(compiler), "-std=c11", "-O2", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
               "-DPRESIDENT_SEARCH", "-D_POSIX_C_SOURCE=200809L", "-shared",
               str(ROOT / "src/president.c"), "-o", str(target)]
    if os.name != "nt":
        command.insert(1, "-fPIC")
        command.append("-lm")
    if sys.platform == "darwin":
        command[command.index("-shared")] = "-dynamiclib"
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Native search build failed:\n{error.stderr}") from error
    return target


@lru_cache(maxsize=4)
def load_library(path):
    lib = ct.CDLL(str(path))
    lib.president_search_version.argtypes = []
    lib.president_search_version.restype = ct.c_int
    if lib.president_search_version() != 1:
        raise RuntimeError("Native search ABI mismatch; rebuild the library")
    base = [ct.POINTER(Position), ct.c_int, ct.POINTER(_Action)]
    signatures = {
        "policy": [ct.POINTER(Position), ct.POINTER(ct.c_double), ct.POINTER(_Action)],
        "batch": base + [ct.c_int, ct.POINTER(ct.c_double), ct.c_double, ct.POINTER(ct.c_double)],
        "scores": base + [ct.c_int, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)],
        "rollout": base + [ct.POINTER(ct.c_double), ct.POINTER(Position), ct.POINTER(ct.c_double)],
    }
    for name, args in signatures.items():
        function = getattr(lib, "president_search_" + name)
        function.argtypes, function.restype = args, ct.c_int
    lib.president_hand_value.argtypes = [ct.POINTER(ct.c_int)] * 4 + [
        ct.c_int, ct.POINTER(ct.c_double), ct.c_int, ct.POINTER(ct.c_double)]
    lib.president_hand_value.restype = ct.c_int
    return lib


@lru_cache(maxsize=1)
def hand_library():
    """Resolve once per process; rebuilding a loaded DLL needs a new process."""
    path = library_path()
    sources = ("president.c", "president.h", "agent_search_native.inc", "agent_search_native.h")
    if not path.is_file() or any((ROOT / "src" / name).stat().st_mtime_ns > path.stat().st_mtime_ns for name in sources):
        return None
    try:
        return load_library(str(path))
    except (OSError, AttributeError, RuntimeError):
        return None


def hand_value(agent, hand_items, unseen, sizes, *, lead):
    """Accelerate cache misses while preserving custom-helper dispatch."""
    mode = getattr(agent, "evaluation_backend", "auto")
    if mode == "python":
        return None
    from src.agent import ExperiencedAgent2
    helpers = ("_holding", "_control", "_control_cached", "_hand_cost", "_hand_cost_cached")
    if lead:
        helpers += ("_lead_cost", "_lead_cost_cached")
    for name in helpers:
        actual, expected = getattr(type(agent), name), getattr(ExperiencedAgent2, name)
        if name in agent.__dict__ or getattr(actual, "__func__", actual) is not getattr(expected, "__func__", expected):
            if mode == "c":
                raise RuntimeError("Custom evaluation helpers require evaluation_backend='python'")
            return None
    values = tuple(getattr(agent, name) for name in WEIGHTS[:10])
    valid = (len(sizes) <= 3 and len(unseen) == 13
             and all(type(n) is int and 0 <= n <= 52 for n in sizes)
             and all(type(n) is int and 0 <= n <= 4 for n in unseen)
             and all(r in RANKS and type(n) is int and 0 <= n <= 4 for r, n in hand_items)
             and all(isinstance(v, (int, float)) and isfinite(v) for v in values))
    if not valid:
        if mode == "c":
            raise ValueError("Hand evaluation inputs exceed the native evaluator's domain")
        return None
    lib = hand_library()
    if lib is None:
        if mode == "c":
            raise RuntimeError("Native evaluator unavailable; run python -m src.agent_search_c --build")
        return None
    counts = dict(hand_items)
    order = [RANKS.index(r) for r, _ in hand_items]
    order += [r for r in range(13) if r not in order]
    result = ct.c_double()
    status = lib.president_hand_value(
        (ct.c_int * 13)(*(counts.get(r, 0) for r in RANKS)),
        (ct.c_int * 13)(*unseen), (ct.c_int * 13)(*order),
        (ct.c_int * 3)(*sizes), len(sizes),
        (ct.c_double * 18)(*values), lead, ct.byref(result))
    if status != 1:
        raise ValueError("Native hand evaluator rejected the inputs")
    return result.value


def supported(agent):
    """Weight-only customizations work; Python policy overrides need Python."""
    from src.agent import ExperiencedAgent4
    helpers = ("_play_scorer", "_hand_cost", "_hand_cost_cached", "_lead_cost", "_lead_cost_cached",
               "_control", "_control_cached", "_holding", "_turn_order_danger", "_unseen", "_opponents",
               "_counts", "_action_cards", "_opponent_counts")
    for name in helpers:
        if name in agent.__dict__:
            return False
        actual, expected = getattr(type(agent), name), getattr(ExperiencedAgent4, name)
        if getattr(actual, "__func__", actual) is not getattr(expected, "__func__", expected):
            return False
    return all(isinstance(getattr(agent, name), (int, float)) and isfinite(getattr(agent, name)) for name in WEIGHTS)


def backend(agent):
    if agent.search_backend == "python":
        return None
    if not supported(agent):
        if agent.search_backend == "c":
            raise RuntimeError("Custom Python policy overrides require search_backend='python'")
        return None
    try:
        path = library_path()
        sources = ("president.c", "president.h", "agent_search_native.inc", "agent_search_native.h")
        if not path.is_file() or any((ROOT / "src" / name).stat().st_mtime_ns > path.stat().st_mtime_ns for name in sources):
            raise RuntimeError("Native search is missing or stale; run python -m src.agent_search_c --build")
        return load_library(str(path))
    except (OSError, AttributeError, RuntimeError):
        if agent.search_backend == "c":
            raise
        return None


def pack(state):
    result = Position()
    seats = {p: i for i, p in enumerate(state.players)}
    for p, i in seats.items():
        cards = state.hands[p]
        if len(cards) > 52:
            raise ValueError("invalid hand size")
        result.sizes[i] = len(cards)
        result.cards[i][:len(cards)] = [RANKS.index(c.rank)*4 + SUITS.index(c.suit) for c in cards]
        result.active[i] = p in state.active
        result.rankings[i] = (state.rankings or {}).get(p, 0)
    for key, values in (("finish", state.finish_order), ("bombs", state.bomb_finishers),
                        ("completion", state.completion_order)):
        getattr(result, key)[:len(values)] = [seats[p] for p in values]
    result.finish_count, result.bomb_count = len(state.finish_order), len(state.bomb_finishers)
    result.top_rank = -1 if state.top_rank is None else RANKS.index(state.top_rank)
    result.top_count, result.stack = state.top_count, state.top_stack_count
    result.last_player, result.turn = seats.get(state.last_player, -1), seats.get(state.turn, -1)
    result.opened = any(e.get("type") == "opening_play" for e in state.events)
    result.first_round, result.phase = state.first_round, PHASES.index(state.phase)
    result.completion_count, result.cursor = len(state.completion_order), state._completion_cursor
    pending = state._pending_advance
    result.pending_player = -1 if pending is None else seats[pending[0]]
    result.pending_skip = False if pending is None else pending[1]
    return result


def weights(agent):
    return (ct.c_double * len(WEIGHTS))(*(getattr(agent, name) for name in WEIGHTS))


class NativeSearchRejected(ValueError):
    """The native simulation could not process this sampled world."""


def batch(lib, agent, world, player, actions, budget_ms):
    position = pack(world)
    native_actions = (_Action * len(actions))(*(_encode_action(a) for a in actions))
    scores = (ct.c_double * len(actions))()
    status = lib.president_search_batch(ct.byref(position), world.players.index(player), native_actions,
                                        len(actions), weights(agent), budget_ms, scores)
    if status < 0:
        raise NativeSearchRejected("Native search rejected the position or root action")
    return list(scores) if status else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", required=True)
    parser.parse_args()
    print(build_library())
