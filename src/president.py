"""A small, deterministic-friendly engine for the four-player President variant.

The wire protocol is deliberately JSON-shaped: callers pass dictionaries to
``Round.apply_action`` and consume dictionaries returned by ``message_for``.
Cards are represented as ``{"rank": "A", "suit": "S"}`` in that protocol.
No networking policy is imposed here; :meth:`Game.run` is a convenient adapter
for in-process clients, while a websocket/process supervisor can call the same
two methods itself.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
import random
from typing import Any, Iterable, Mapping, Optional, Sequence


RANKS = ("3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2")
SUITS = ("C", "D", "H", "S")
RANK_VALUE = {rank: value for value, rank in enumerate(RANKS)}
ROLE_NAMES = ("President", "Vice President", "Vice Scum", "Scum")


class IllegalAction(ValueError):
    """Raised when an action is not legal in the round's current state."""


@dataclass(frozen=True, order=True)
class Card:
    rank: str
    suit: str

    def __post_init__(self) -> None:
        if self.rank not in RANK_VALUE or self.suit not in SUITS:
            raise ValueError("invalid card: %s%s" % (self.rank, self.suit))

    def json(self) -> dict[str, str]:
        return {"rank": self.rank, "suit": self.suit}

    @classmethod
    def from_json(cls, value: Mapping[str, str]) -> "Card":
        return cls(value["rank"], value["suit"])


def standard_deck() -> list[Card]:
    return [Card(rank, suit) for rank in RANKS for suit in SUITS]


class Round:
    """One round, driven a single action at a time.

    ``roles`` maps player ids to their previous placement (1 is President).
    It must be omitted on the first round and supplied for later rounds.
    Supplying ``deck`` is useful for repeatable tests; its order is shuffled
    only when it is omitted.
    """

    def __init__(self, players: Sequence[str], *, first_round: bool,
                 roles: Optional[Mapping[str, int]] = None,
                 deck: Optional[Iterable[Card]] = None, rng: Optional[random.Random] = None) -> None:
        if len(players) != 4 or len(set(players)) != 4:
            raise ValueError("this implementation requires exactly four distinct players")
        if first_round == (roles is not None):
            raise ValueError("roles are required exactly for later rounds")
        self.players = list(players)
        self.first_round = first_round
        self.roles = dict(roles or {})
        if roles is not None and (set(roles) != set(players) or set(roles.values()) != {1, 2, 3, 4}):
            raise ValueError("roles must assign ranks 1..4 to these players")
        cards = list(deck) if deck is not None else standard_deck()
        if len(cards) != 52 or len(set(cards)) != 52:
            raise ValueError("deck must contain each of the 52 cards once")
        if deck is None:
            (rng or random.Random()).shuffle(cards)

        self.hands: dict[str, list[Card]] = {p: [] for p in players}
        self.active: set[str] = set(players)
        self.events: list[dict[str, Any]] = []
        # Unlike ``events``, this records every accepted input, including
        # private trade-return cards.  It intentionally is not sent by
        # ``message_for`` because that would reveal those cards to observers.
        self.turn_history: list[dict[str, Any]] = []
        self.finish_order: list[str] = []
        self.bomb_finishers: list[str] = []
        self.rankings: Optional[dict[str, int]] = None
        self.top_rank: Optional[str] = None
        self.top_count = 0
        # ``top_count`` is the last play's size (the counter-play rule).
        # This separately tracks the contiguous same-rank run for completion.
        self.top_stack_count = 0
        self.last_player: Optional[str] = None
        self.turn: Optional[str] = None
        self.phase = "draft" if not first_round else "play"
        self.completion_order: list[str] = []
        self._completion_cursor = 0
        self._pending_advance: Optional[tuple[str, bool]] = None
        self._trade_steps: list[tuple[str, str]] = []
        self._trade_index = 0
        self._pending_trade: Optional[tuple[str, str, str]] = None  # initiator, receiver, rank

        if first_round:
            for i, card in enumerate(cards):
                self.hands[players[i % 4]].append(card)
            self.turn = next(p for p in players if Card("3", "C") in self.hands[p])
            self._event("round_started", first_round=True, first_player=self.turn)
        else:
            self.piles = [cards[i * 13:(i + 1) * 13] for i in range(4)]
            self.draft_order = [p for p, _ in sorted(self.roles.items(), key=lambda x: x[1])]
            self._event("round_started", first_round=False,
                        face_up=[pile[-1].json() for pile in self.piles], draft_order=self.draft_order[:])

    # Public API ---------------------------------------------------------
    @property
    def done(self) -> bool:
        return self.phase == "finished"

    def message_for(self, player: str) -> dict[str, Any]:
        """Return an independent snapshot and private request for *player*."""
        if player not in self.hands:
            raise KeyError(player)
        request: Optional[dict[str, Any]] = None
        actor = self._actor()
        if actor == player and not self.done:
            request = {"type": self.phase, "legal_actions": self.legal_actions(player)}
        return {
            "type": "state", "player": player, "phase": self.phase,
            "hand": [c.json() for c in self._sorted_hand(player)],
            "turn": actor, "top": None if self.top_rank is None else {"rank": self.top_rank, "count": self.top_count, "stack_count": self.top_stack_count},
            "active_players": self._active_in_order(), "events": deepcopy(self.events),
            "request": request, "rankings": deepcopy(self.rankings),
        }

    def legal_actions(self, player: str) -> list[dict[str, Any]]:
        if self._actor() != player or self.done:
            return []
        if self.phase == "draft":
            return [{"type": "choose_pile", "pile": i} for i in range(4) if self.piles[i]]
        if self.phase == "trade_request":
            return [{"type": "trade_request", "rank": r} for r in RANKS]
        if self.phase == "trade_return":
            return [{"type": "trade_return", "card": c.json()} for c in self._sorted_hand(player)]
        if self.phase == "completion":
            needed = 4 - self.top_stack_count
            cards = [c for c in self.hands[player] if c.rank == self.top_rank]
            result = [{"type": "decline_completion"}]
            if len(cards) >= needed:
                result[0:0] = [
                    {"type": "complete", "cards": [c.json() for c in choice]}
                    for choice in combinations(cards, needed)
                ]
            return result
        # play
        plays: list[dict[str, Any]] = []
        by_rank = {r: [c for c in self.hands[player] if c.rank == r] for r in RANKS}
        if self.top_rank is None:
            for rank, cards in by_rank.items():
                if rank != "2":
                    # The opening play is the one exceptional cleared-board
                    # play: it must contain a 3.  Suits are only card identity;
                    # they do not affect play legality.
                    if self.first_round and not any(e["type"] == "opening_play" for e in self.events):
                        if rank != "3":
                            continue
                        for n in range(1, len(cards) + 1):
                            plays.extend(
                                {"type": "play", "cards": [c.json() for c in choice]}
                                for choice in combinations(cards, n)
                            )
                        continue
                    for n in range(1, len(cards) + 1):
                        plays.extend(
                            {"type": "play", "cards": [c.json() for c in choice]}
                            for choice in combinations(cards, n)
                        )
        else:
            plays.append({"type": "pass"})
            plays.extend({"type": "bomb", "card": c.json()} for c in by_rank["2"])
            for rank, cards in by_rank.items():
                if rank != "2" and RANK_VALUE[rank] >= RANK_VALUE[self.top_rank] and len(cards) >= self.top_count:
                    plays.extend(
                        {"type": "play", "cards": [c.json() for c in choice]}
                        for choice in combinations(cards, self.top_count)
                    )
        return plays

    def apply_action(self, player: str, action: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Apply one JSON action and record it.  Return new public events only.

        ``turn_history`` receives an independent ``player``/``action`` snapshot
        only after validation and state changes succeed; rejected actions leave
        both the round state and its history unchanged.
        """
        if self.done or self._actor() != player:
            raise IllegalAction("it is not this player's turn/request")
        if not isinstance(action, Mapping):
            raise IllegalAction("action must be an object")
        start = len(self.events)
        kind = action.get("type")
        if self.phase == "draft": self._choose_pile(player, action)
        elif self.phase == "trade_request": self._trade_request(player, action)
        elif self.phase == "trade_return": self._trade_return(player, action)
        elif self.phase == "completion": self._completion(player, action)
        elif kind == "pass": self._pass(player)
        elif kind == "bomb": self._bomb(player, action)
        elif kind == "play": self._play(player, action)
        else: raise IllegalAction("unknown action")
        self.turn_history.append({"player": player, "action": deepcopy(dict(action))})
        return deepcopy(self.events[start:])

    # Phase transitions --------------------------------------------------
    def _choose_pile(self, player: str, action: Mapping[str, Any]) -> None:
        if action.get("type") != "choose_pile" or type(action.get("pile")) is not int: raise IllegalAction("choose an available pile")
        pile = action["pile"]
        if pile not in range(4) or not self.piles[pile]: raise IllegalAction("pile unavailable")
        self.hands[player] = self.piles[pile]; self.piles[pile] = []
        self._event("pile_chosen", player=player, pile=pile)
        if all(not p for p in self.piles):
            self._trade_steps = [(self._by_role(1), self._by_role(4)), (self._by_role(1), self._by_role(4)), (self._by_role(2), self._by_role(3))]
            self.phase = "trade_request"; self._event("draft_complete")
        else: self.draft_order.pop(0)

    def _trade_request(self, player: str, action: Mapping[str, Any]) -> None:
        rank = action.get("rank")
        if action.get("type") != "trade_request" or not isinstance(rank, str) or rank not in RANK_VALUE: raise IllegalAction("request a valid rank")
        initiator, receiver = self._trade_steps[self._trade_index]
        if player != initiator: raise IllegalAction("wrong trade initiator")
        available = [c for c in self.hands[receiver] if c.rank == rank]
        self._event("trade_requested", initiator=initiator, receiver=receiver, rank=rank, success=bool(available))
        if not available: return
        self.hands[receiver].remove(sorted(available)[0]); self.hands[initiator].append(sorted(available)[0])
        self._pending_trade = (initiator, receiver, rank); self.phase = "trade_return"

    def _trade_return(self, player: str, action: Mapping[str, Any]) -> None:
        if action.get("type") != "trade_return": raise IllegalAction("return a card")
        initiator, receiver, _ = self._pending_trade or (None, None, None)
        if player != initiator: raise IllegalAction("wrong trade initiator")
        card = self._card_from_action(action, "card")
        self._remove(player, [card]); self.hands[receiver].append(card)
        self._event("trade_completed", initiator=initiator, receiver=receiver)  # returned card is private
        self._pending_trade = None; self._trade_index += 1
        if self._trade_index == len(self._trade_steps):
            self.phase = "play"; self.turn = self._by_role(4); self._event("trades_complete", first_player=self.turn)
            self._check_twos()
        else: self.phase = "trade_request"

    def _play(self, player: str, action: Mapping[str, Any]) -> None:
        cards = self._cards_from_action(action)
        if not cards or len({c.rank for c in cards}) != 1 or cards[0].rank == "2": raise IllegalAction("play equal-rank non-2 cards")
        rank = cards[0].rank
        if self.top_rank is None:
            # A 3 is mandatory only for the opening play; its suit is irrelevant.
            if self.first_round and not any(e["type"] == "opening_play" for e in self.events) and rank != "3": raise IllegalAction("opening play must contain a 3")
        elif len(cards) != self.top_count or RANK_VALUE[rank] < RANK_VALUE[self.top_rank]: raise IllegalAction("play does not beat the top")
        equal = rank == self.top_rank
        self._remove(player, cards)
        self.top_stack_count = self.top_stack_count + len(cards) if equal else len(cards)
        self.top_rank, self.top_count, self.last_player = rank, len(cards), player
        self._event("opening_play" if not any(e["type"] == "opening_play" for e in self.events) else "played", player=player, cards=[c.json() for c in cards])
        self._after_play(player, skip=equal)

    def _pass(self, player: str) -> None:
        if self.top_rank is None: raise IllegalAction("cannot pass on a cleared board")
        self._event("passed", player=player)
        self._offer_completions(player)

    def _bomb(self, player: str, action: Mapping[str, Any]) -> None:
        if self.top_rank is None:
            raise IllegalAction("cannot bomb on a cleared board")
        card = self._card_from_action(action, "card")
        if card.rank != "2": raise IllegalAction("a bomb is a single 2")
        if self.first_round and not any(e["type"] == "opening_play" for e in self.events):
            raise IllegalAction("opening play must contain a 3")
        self._remove(player, [card]); self._event("bombed", player=player, card=card.json())
        self._finish_if_empty(player, bomb=True)
        self._check_twos()
        if not self.done: self._clear_to(player)

    def _after_play(self, player: str, *, skip: bool) -> None:
        self._finish_if_empty(player, bomb=False)
        self._check_twos()
        if self.done: return
        # The play itself can supply the final matching cards.  That is an
        # immediate completion, not an opportunity for another player to
        # "complete" with zero cards.
        if self.top_stack_count == 4:
            self._clear_to(player); return
        self._offer_completions(player, skip)

    def _offer_completions(self, player: str, skip: bool = False) -> None:
        # Poll every active player, including those who can only decline.
        # Selecting respondents using their cards would reveal private ranks.
        # Open a fresh window after passes too: a previous decline is not final.
        self.completion_order = ([player] if player in self.active else []) + [
            p for p in self._active_after(player) if p != player
        ]
        self._completion_cursor = 0
        self._pending_advance = (player, skip)
        self.phase = "completion"

    def _completion(self, player: str, action: Mapping[str, Any]) -> None:
        if action.get("type") == "decline_completion":
            self._completion_cursor += 1
            if self._completion_cursor >= len(self.completion_order):
                assert self._pending_advance is not None
                previous_player, skip = self._pending_advance
                self._pending_advance = None
                self.phase = "play"
                self._advance(previous_player, skip)
            return
        if action.get("type") != "complete": raise IllegalAction("complete or decline")
        cards = self._cards_from_action(action)
        if len(cards) != 4 - self.top_stack_count or any(c.rank != self.top_rank for c in cards): raise IllegalAction("completion must make four of a kind")
        self._remove(player, cards); self._event("completed", player=player, cards=[c.json() for c in cards])
        self._finish_if_empty(player, bomb=False); self._check_twos()
        if not self.done: self._clear_to(player)

    # Mechanics ----------------------------------------------------------
    def _actor(self) -> Optional[str]:
        if self.phase == "draft": return self.draft_order[0]
        if self.phase == "trade_request": return self._trade_steps[self._trade_index][0]
        if self.phase == "trade_return": return self._pending_trade[0] if self._pending_trade else None
        if self.phase == "completion": return self.completion_order[self._completion_cursor]
        return self.turn

    def _advance(self, player: str, skip: bool = False) -> None:
        # ``last_player`` may already be done.  In that case it is not a
        # candidate turn, but play still returns to that seat and clears there.
        if self._returns_to_last_player(player, skip):
            assert self.last_player is not None
            self._clear_to(self.last_player)
        else:
            self.turn = self._next_active(player, skip)

    def _clear_to(self, player: str) -> None:
        self.top_rank = None; self.top_count = 0; self.top_stack_count = 0; self.last_player = None; self.phase = "play"
        self.completion_order = []; self._completion_cursor = 0; self._pending_advance = None
        self.turn = player if player in self.active else self._next_active(player)
        self._event("board_cleared", player=self.turn)

    def _finish_if_empty(self, player: str, *, bomb: bool) -> None:
        if self.hands[player]: return
        self.active.remove(player)
        (self.bomb_finishers if bomb else self.finish_order).append(player)
        self._event("player_done", player=player, bomb_finish=bomb)
        self._check_end()

    def _check_twos(self) -> None:
        if self.done: return
        for p in self._active_in_order():
            if self.hands[p] and all(c.rank == "2" for c in self.hands[p]):
                self.active.remove(p); self.bomb_finishers.append(p); self._event("twos_announced", player=p)
        self._check_end()

    def _check_end(self) -> None:
        if self.done: return
        if len(self.active) > 1: return
        if self.active:
            p = next(iter(self.active)); self.active.remove(p); self.finish_order.append(p); self._event("player_done", player=p, last_active=True)
        ordering = self.finish_order + self.bomb_finishers
        self.rankings = {p: i + 1 for i, p in enumerate(ordering)}
        self.phase = "finished"; self.turn = None; self._event("round_finished", rankings=self.rankings)

    def _next_active(self, player: str, skip: bool = False) -> str:
        i = self.players.index(player)
        for _ in range(len(self.players)):
            i = (i + 1) % len(self.players)
            if self.players[i] in self.active:
                if skip: skip = False; continue
                return self.players[i]
        raise RuntimeError("no active players")

    def _returns_to_last_player(self, player: str, skip: bool) -> bool:
        if self.last_player is None:
            return False
        i = self.players.index(player)
        for _ in range(len(self.players)):
            i = (i + 1) % len(self.players)
            candidate = self.players[i]
            if candidate == self.last_player:
                return True
            if candidate in self.active:
                if skip:
                    skip = False
                    continue
                return False
        return False

    def _active_after(self, player: str) -> list[str]:
        i = self.players.index(player)
        result = []
        for _ in range(len(self.players)):
            i = (i + 1) % len(self.players)
            candidate = self.players[i]
            if candidate in self.active:
                result.append(candidate)
        return result

    def _active_in_order(self) -> list[str]: return [p for p in self.players if p in self.active]
    def _by_role(self, role: int) -> str: return next(p for p, r in self.roles.items() if r == role)
    def _sorted_hand(self, player: str) -> list[Card]: return sorted(self.hands[player], key=lambda c: (RANK_VALUE[c.rank], c.suit))
    def _event(self, type_: str, **fields: Any) -> None: self.events.append({"type": type_, **deepcopy(fields)})
    def _cards_from_action(self, action: Mapping[str, Any]) -> list[Card]:
        if not isinstance(action.get("cards"), list): raise IllegalAction("cards must be a list")
        try:
            return [Card.from_json(c) for c in action["cards"]]
        except (KeyError, TypeError, ValueError) as error:
            raise IllegalAction("invalid cards") from error
    def _card_from_action(self, action: Mapping[str, Any], key: str) -> Card:
        if not isinstance(action.get(key), Mapping): raise IllegalAction("missing card")
        try:
            return Card.from_json(action[key])
        except (KeyError, TypeError, ValueError) as error:
            raise IllegalAction("invalid card") from error
    def _remove(self, player: str, cards: Sequence[Card]) -> None:
        remaining = self.hands[player][:]
        for card in cards:
            if card not in remaining: raise IllegalAction("card is not in hand")
            remaining.remove(card)
        self.hands[player] = remaining


class Game:
    """A sequence of rounds.  The previous round's placement determines roles."""
    def __init__(self, players: Sequence[str], *, rng: Optional[random.Random] = None) -> None:
        self.players = list(players); self.rng = rng or random.Random(); self.round_number = 0
        self.roles: Optional[dict[str, int]] = None; self.round: Optional[Round] = None

    def start_round(self, *, deck: Optional[Iterable[Card]] = None) -> Round:
        """Start a round, carrying forward the completed round's placements."""
        if self.round is not None and not self.round.done:
            raise RuntimeError("current round has not finished")
        roles = dict(self.round.rankings) if self.round is not None else self.roles
        next_round = Round(self.players, first_round=self.round_number == 0, roles=roles, deck=deck, rng=self.rng)
        self.roles = roles
        self.round = next_round
        self.round_number += 1; return self.round

    def finish_round(self) -> dict[str, int]:
        if not self.round or not self.round.done: raise RuntimeError("round has not finished")
        self.roles = dict(self.round.rankings or {}); return self.roles.copy()

    def run(self, clients: Mapping[str, "Client"], *, max_actions: int = 10000) -> Round:
        """Run the current round using in-process clients and return it."""
        if not self.round: self.start_round()
        assert self.round is not None
        for _ in range(max_actions):
            if self.round.done: return self.round
            player = self.round._actor()
            if player is None: raise RuntimeError("round has no requested player")
            self.round.apply_action(player, clients[player].respond(self.round.message_for(player)))
        if self.round.done: return self.round
        raise RuntimeError("action limit reached")
