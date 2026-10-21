"""Exact, suit-independent minimax for the last two active players.

Usage: ``result = solve_round(round_)``; apply ``result.best_action`` for
``result.player``. Both hands must be known to the caller. This adapter reads
the engine directly (it does not infer hidden cards from protocol messages).
Results describe optimal win/loss play, with no fastest-win tie breaker.
"""

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any

from src.president import RANKS, RANK_VALUE, Round


@dataclass(frozen=True)
class _State:
    hands: tuple[tuple[int, ...], tuple[int, ...]]
    actor: int
    rank: int = -1
    count: int = 0
    stack: int = 0
    # Respondents left in this completion window; empty means normal play.
    window: tuple[int, ...] = ()
    clear_after: bool = False
    next_actor: int = 0
    # Needed when the last play belonged to an already finished player.
    pass_clears: bool = True
    pass_actor: int = 0
    winner: int = -1


@dataclass(frozen=True)
class _Move:
    kind: str
    rank: int = -1
    count: int = 0


@dataclass(frozen=True)
class SolveResult:
    player: str
    best_action: dict[str, Any]
    forced_result: str
    states_evaluated: int
    cache_hits: int


def _moves(s: _State) -> tuple[_Move, ...]:
    hand = s.hands[s.actor]
    if s.window:
        needed = 4 - s.stack
        completion = (_Move("complete", s.rank, needed),) if hand[s.rank] >= needed else ()
        return completion + (_Move("decline_completion"),)
    plays = tuple(
        _Move("play", rank, count)
        for rank in range(12)
        for count in (range(hand[rank], 0, -1) if s.rank < 0 else (s.count,))
        if hand[rank] >= count and (s.rank < 0 or rank >= s.rank)
    )
    if s.rank < 0:
        return plays
    return plays + ((_Move("bomb", 12, 1),) if hand[12] else ()) + (_Move("pass"),)


def _clear(s: _State, actor: int) -> _State:
    return _State(s.hands, actor)


def _step(s: _State, m: _Move) -> _State:
    actor = s.actor
    if m.kind == "decline_completion":
        remaining = s.window[1:]
        if remaining:
            return replace(s, actor=remaining[0], window=remaining)
        if s.clear_after:
            return _clear(s, s.next_actor)
        return replace(s, actor=s.next_actor, window=(), clear_after=False, next_actor=0)
    if m.kind == "pass":
        # If this pass does not reach the finished last player, the next
        # player's pass will. That clear is inherited by the current actor.
        # Carry the next turn's pass behavior through the completion window.
        return replace(s, window=(actor, 1 - actor),
                       clear_after=s.pass_clears, next_actor=s.pass_actor,
                       pass_clears=True,
                       pass_actor=s.pass_actor if s.pass_clears else actor)

    hand = list(s.hands[actor])
    hand[m.rank] -= m.count
    hands = list(s.hands)
    hands[actor] = tuple(hand)
    child = replace(s, hands=tuple(hands))
    if not any(hand):
        return replace(child, winner=1 - actor if m.kind == "bomb" else actor)
    if not any(hand[:12]):
        return replace(child, winner=1 - actor)
    if m.kind in {"bomb", "complete"}:
        return _clear(child, actor)
    equal = m.rank == s.rank
    stack = s.stack + m.count if equal else m.count
    if stack == 4:
        return _clear(child, actor)
    return _State(child.hands, actor, m.rank, m.count, stack,
                  (actor, 1 - actor), equal, actor if equal else 1 - actor,
                  True, actor)


def _from_round(round_: Round, players: list[str]) -> _State:
    def index(player: str) -> int:
        return players.index(player)

    actor = round_._actor()
    hands = tuple(tuple(sum(c.rank == rank for c in round_.hands[p])
                        for rank in RANKS) for p in players)
    # Preserve inherited-board behavior even when last_player has finished.
    clears = round_._returns_to_last_player(actor, False)
    target = round_.last_player if clears else round_._next_active(actor)
    if target not in players:
        target = round_._next_active(target)
    state = _State(hands, index(actor),
                   -1 if round_.top_rank is None else RANK_VALUE[round_.top_rank],
                   round_.top_count, round_.top_stack_count,
                   pass_clears=clears, pass_actor=index(target))
    if round_.phase == "completion":
        previous, skip = round_._pending_advance
        clears = round_._returns_to_last_player(previous, skip)
        target = round_.last_player if clears else round_._next_active(previous, skip)
        if target not in players:
            target = round_._next_active(target)
        state = replace(state,
                        window=tuple(index(p) for p in round_.completion_order[round_._completion_cursor:]),
                        clear_after=clears, next_actor=index(target))
        # After this window, a pass is made by the pending next actor.
        if not clears:
            pass_clears = round_._returns_to_last_player(target, False)
            pass_target = round_.last_player if pass_clears else round_._next_active(target)
            if pass_target not in players:
                pass_target = round_._next_active(pass_target)
            state = replace(state, pass_clears=pass_clears, pass_actor=index(pass_target))
    return state


def solve_round(round_: Round) -> SolveResult:
    """Return a legal optimal action without modifying ``round_``.

    Accepts play or completion decisions with exactly two active players.
    Each call owns its memo table, so unrelated games do not retain memory.
    Search is exhaustive in the worst case; there is no heuristic/time cutoff.
    """
    players = round_._active_in_order()
    if round_.done or len(players) != 2 or round_.phase not in {"play", "completion"}:
        raise ValueError("solver requires a play/completion decision with two active players")
    if any(not any(c.rank != "2" for c in round_.hands[p]) for p in players):
        raise ValueError("active hands must contain a non-bomb card")
    root = _from_round(round_, players)

    @lru_cache(maxsize=None)
    def winner(s: _State) -> int:
        if s.winner >= 0:
            return s.winner
        for move in _moves(s):
            if winner(_step(s, move)) == s.actor:
                return s.actor
        return 1 - s.actor

    moves = _moves(root)
    best = moves[0]
    outcome = 1 - root.actor
    for move in moves:
        if winner(_step(root, move)) == root.actor:
            best, outcome = move, root.actor
            break
    player = players[root.actor]
    # Resolve suit-equivalent moves back to an engine-provided legal action.
    for action in round_.legal_actions(player):
        if action["type"] != best.kind:
            continue
        cards = action.get("cards", [action["card"]] if "card" in action else [])
        if best.rank < 0 or (len(cards) == best.count and cards[0]["rank"] == RANKS[best.rank]):
            info = winner.cache_info()
            return SolveResult(player, action, "win" if outcome == root.actor else "loss",
                               info.misses, info.hits)
    raise RuntimeError("solver action does not match engine legal actions")
