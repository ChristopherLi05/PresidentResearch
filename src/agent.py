from collections import Counter
from math import comb
from typing import Any, Mapping, Optional
import random

from src.president import RANK_VALUE


class Agent:
    """Base class for a JSON-protocol player. Override :meth:`respond`.

    Hosts may also call :meth:`observe` with state snapshots between turns.
    Stateless clients can ignore those nonblocking updates; interactive clients
    can use them to keep their display current while another player acts.
    """

    def observe(self, message: Mapping[str, Any]) -> None:
        """Receive a nonblocking state update. The default client ignores it."""

    def respond(self, message: Mapping[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class RandomAgent(Agent):
    """Example client: passes half the time when passing is legal, else acts randomly."""

    def __init__(self, rng: Optional[random.Random] = None) -> None: self.rng = rng or random.Random()

    def respond(self, message: Mapping[str, Any]) -> dict[str, Any]:
        actions = message["request"]["legal_actions"]
        passes = [a for a in actions if a["type"] in {"pass", "decline_completion"}]
        if passes and self.rng.random() < .5: return self.rng.choice(passes)
        return self.rng.choice([a for a in actions if a not in passes] or actions)


class BasicAgent(Agent):
    """Always act when possible, playing low and drafting/requesting high.

    Equal-rank plays prefer the largest group. Trade returns use the lowest
    card. Failed requests are remembered from the public history of the
    current exchange, so this client needs no private state between responses.
    """

    def respond(self, message: Mapping[str, Any]) -> dict[str, Any]:
        request = message.get("request")
        if not request or not request["legal_actions"]:
            raise ValueError("no legal action requested")
        actions = request["legal_actions"]

        if request["type"] == "draft":
            start = next(e for e in reversed(message["events"])
                         if e["type"] == "round_started")
            return max(actions, key=lambda a: RANK_VALUE[start["face_up"][a["pile"]]["rank"]])

        if request["type"] == "trade_request":
            failed = set()
            for event in reversed(message["events"]):
                if event["type"] in {"trade_completed", "round_started"}:
                    break
                if (event["type"] == "trade_requested"
                        and event["initiator"] == message["player"]
                        and not event["success"]):
                    failed.add(event["rank"])
            candidates = [a for a in actions if a["rank"] not in failed]
            return max(candidates, key=lambda a: RANK_VALUE[a["rank"]])

        if request["type"] == "trade_return":
            return min(actions, key=lambda a: (RANK_VALUE[a["card"]["rank"]], a["card"]["suit"]))

        completions = [a for a in actions if a["type"] == "complete"]
        if completions:
            return completions[0]

        plays = [a for a in actions if a["type"] in {"play", "bomb"}]
        if plays:
            def play_key(action):
                cards = action["cards"] if action["type"] == "play" else [action["card"]]
                return (RANK_VALUE[cards[0]["rank"]], -len(cards),
                        tuple(sorted(c["suit"] for c in cards)))

            return min(plays, key=play_key)

        # The engine also polls players who cannot complete, and a turn may
        # have no playable cards. Only then is a pass/decline necessary.
        return actions[0]


class BasicAgentPlus(BasicAgent):
    """A :class:`BasicClient` that bombs instead of playing J or higher.

    Completion, drafting, and trade decisions use the base policy. On a normal
    turn, a legal bomb replaces the lowest ordinary legal play when that play
    is J, Q, K, or A.
    """

    def respond(self, message: Mapping[str, Any]) -> dict[str, Any]:
        request = message.get("request")
        if request and not any(a["type"] == "complete" for a in request["legal_actions"]):
            ordinary_plays = [a for a in request["legal_actions"] if a["type"] == "play"]
            bombs = [a for a in request["legal_actions"] if a["type"] == "bomb"]
            if (ordinary_plays and bombs
                    and min(RANK_VALUE[a["cards"][0]["rank"]]
                            for a in ordinary_plays) >= RANK_VALUE["J"]):
                return bombs[0]
        return super().respond(message)


class ExperiencedAgent(Agent):
    """A hand-shape-aware President policy.

    Unlike :class:`BasicClient`, this client deliberately manages ordinary
    rank groups as cover for its 2s.  It is entirely protocol driven: all
    opponent information is reconstructed from the public event stream.
    """

    HIGH = frozenset(("J", "Q", "K", "A"))
    LOW = frozenset(("3", "4", "5", "6", "7", "8"))

    @staticmethod
    def _counts(hand: list[Mapping[str, Any]]) -> Counter[str]:
        return Counter(card["rank"] for card in hand)

    @classmethod
    def _safety(cls, hand: list[Mapping[str, Any]]) -> tuple[int, int, bool]:
        counts = cls._counts(hand)
        groups = sum(rank != "2" for rank in counts)
        twos = counts["2"]
        return groups, twos, groups >= twos

    @staticmethod
    def _without(hand: list[Mapping[str, Any]], cards: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        remaining = list(hand)
        for card in cards:
            remaining.remove(card)
        return remaining

    @staticmethod
    def _action_cards(action: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        return list(action.get("cards", [])) if action["type"] != "bomb" else [action["card"]]

    @classmethod
    def _opponent_counts(cls, message: Mapping[str, Any]) -> dict[str, int]:
        """Derive public remaining-card counts; trades do not change totals."""
        counts = {player: 13 for player in message.get("players", [])}
        for event in message.get("events", []):
            # Protocol events always name a seated player.  Ignoring an
            # unknown name also makes the client safe with partial histories
            # supplied by UIs and test fixtures.
            if event.get("player") not in counts:
                continue
            if event["type"] in {"opening_play", "played", "completed"}:
                counts[event["player"]] -= len(event["cards"])
            elif event["type"] == "bombed":
                counts[event["player"]] -= 1
            elif event["type"] in {"player_done", "twos_announced"}:
                counts[event["player"]] = 0
        return counts

    @staticmethod
    def _failed_requests(message: Mapping[str, Any]) -> set[str]:
        """Failures since the last completed exchange belong to this request."""
        failed: set[str] = set()
        for event in reversed(message.get("events", [])):
            if event["type"] in {"trade_completed", "draft_complete", "round_started"}:
                break
            if (event["type"] == "trade_requested"
                    and event["initiator"] == message["player"]
                    and not event["success"]):
                failed.add(event["rank"])
        return failed

    def _trade_rank(self, message: Mapping[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any]:
        """Re-rank requests from the current hand after each failed request."""
        counts = self._counts(message["hand"])
        groups, twos, safe = self._safety(message["hand"])
        failed = self._failed_requests(message)
        offered = {action["rank"]: action for action in actions if action["rank"] not in failed}

        # A priority list, rather than remembered request order, is important:
        # a successful exchange changes the shape of the next request's hand.
        priorities: list[str] = []
        priorities += sorted((rank for rank, n in counts.items() if rank != "2" and n == 3),
                             key=RANK_VALUE.__getitem__, reverse=True)
        if counts["A"] == 1:
            priorities.append("A")
        if counts["2"] < 2:
            priorities.append("2")
        priorities += sorted((rank for rank in ("K", "Q", "J") if 0 < counts[rank] < 4),
                             key=RANK_VALUE.__getitem__, reverse=True)
        # A third 2 is useful only after the A/K matching opportunities.
        if counts["2"] == 2 and counts["A"] != 1 and counts["K"] == 0:
            priorities.append("2")
        priorities += [rank for rank in ("A", "K", "Q", "J") if counts[rank] == 0]
        # Do not seek a fourth 2 until high matching/singleton work is spent,
        # and never ask for a 2 that would leave too few ordinary groups.
        high_work_left = any(counts[rank] in (0, 1) for rank in self.HIGH)
        if counts["2"] == 3 and not high_work_left and groups >= 4:
            priorities.append("2")
        priorities += sorted((rank for rank, n in counts.items()
                              if rank not in self.HIGH and rank != "2" and n > 0),
                             key=RANK_VALUE.__getitem__, reverse=True)
        priorities += [rank for rank in ("10", "9", "8", "7", "6", "5", "4", "3") if counts[rank] == 0]

        for rank in priorities:
            if rank == "2" and (twos + 1 > groups or not safe and twos >= groups):
                continue
            if rank in offered:
                return offered[rank]
        # Failed ranks are the only ranks we know cannot work.  This fallback
        # keeps the client legal even in unusual already-unsafe hands.
        return max(offered.values(), key=lambda action: RANK_VALUE[action["rank"]])

    def _trade_return(self, message: Mapping[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any]:
        hand = message["hand"]
        counts = self._counts(hand)
        candidates = [a for a in actions if a["card"]["rank"] != "2" and counts[a["card"]["rank"]] < 4]
        if not candidates:  # Forced exceptional case: quads/2s are all that remain.
            candidates = list(actions)
        safe = [a for a in candidates if self._safety(self._without(hand, [a["card"]]))[2]]
        if safe:
            candidates = safe

        def key(action: dict[str, Any]) -> tuple[int, int, int, int]:
            rank = action["card"]["rank"]
            size = counts[rank]
            # A low triple safely becomes a pair and is specifically disposable.
            low_triple = rank in self.LOW and size == 3
            low_single = rank in self.LOW and size == 1
            creates_singleton = size == 2
            protects_high = rank in self.HIGH
            return (0 if low_triple else 1,
                    0 if low_single else 1,
                    int(creates_singleton) + int(size == 3 and not low_triple) + int(protects_high),
                    RANK_VALUE[rank])
        return min(candidates, key=key)

    def _completion(self, message: Mapping[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any]:
        completions = [a for a in actions if a["type"] == "complete"]
        if not completions:
            return next(a for a in actions if a["type"] == "decline_completion")
        hand = message["hand"]
        for action in completions:
            remaining = self._without(hand, action["cards"])
            if not remaining or self._safety(remaining)[2]:
                return action
        return next(a for a in actions if a["type"] == "decline_completion")

    def _play(self, message: Mapping[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any]:
        hand = message["hand"]
        counts = self._counts(hand)
        groups, twos, safe = self._safety(hand)
        plays = [a for a in actions if a["type"] in {"play", "bomb"}]
        passes = [a for a in actions if a["type"] == "pass"]
        if not plays:
            return passes[0]

        # Never deliberately finish by bombing: the engine assigns that a
        # bomb-finish penalty.  Ordinary legal finishes are always first.
        finishes = [a for a in plays if a["type"] != "bomb" and not self._without(hand, self._action_cards(a))]
        if finishes:
            return min(finishes, key=lambda a: RANK_VALUE[self._action_cards(a)[0]["rank"]])

        def after(action: dict[str, Any]) -> list[Mapping[str, Any]]:
            return self._without(hand, self._action_cards(action))

        bombs = [a for a in plays if a["type"] == "bomb"]
        ordinary = [a for a in plays if a["type"] == "play"]
        # Spending a 2 restores safety, and becomes mandatory in spirit once
        # each remaining ordinary group is needed as its cover.
        urgent_bombs = [a for a in bombs if self._safety(after(a))[2]
                        and (not safe or twos >= groups)]
        if urgent_bombs:
            return urgent_bombs[0]

        def play_key(action: dict[str, Any]) -> tuple[int, int, int, int, int]:
            cards = self._action_cards(action)
            rank = cards[0]["rank"]
            remaining = after(action)
            new_groups, new_twos, new_safe = self._safety(remaining)
            size = counts[rank]
            removes_group = rank != "2" and len(cards) == size
            splits_group = rank != "2" and len(cards) < size
            weakens_pair_or_triple = splits_group and size in (2, 3)
            # Lower values are preferred.  Safe actions that remove a whole
            # group outrank partial plays; high control cards are held back.
            return (0 if new_safe else 2,
                    0 if removes_group else 1,
                    int(weakens_pair_or_triple),
                    int(rank in self.HIGH), RANK_VALUE[rank])

        best = min(ordinary, key=play_key) if ordinary else None
        if best is not None:
            best_cards = self._action_cards(best)
            rank = best_cards[0]["rank"]
            sacrifice = (len(best_cards) < counts[rank] or
                         (rank in self.HIGH and len(best_cards) < counts[rank]))
            opponent_counts = self._opponent_counts(message)
            danger = any(0 < count <= 2 for player, count in opponent_counts.items()
                         if player != message["player"])
            board_rank = message.get("top", {}).get("rank") if message.get("top") else None
            low_board = board_rank is not None and RANK_VALUE[board_rank] <= RANK_VALUE["10"]
            # Passing is a real option only on a quiet low board when every
            # available ordinary response sacrifices hand structure.
            if passes and sacrifice and not danger and low_board and twos < groups:
                non_sacrifices = [a for a in ordinary if len(self._action_cards(a)) == counts[self._action_cards(a)[0]["rank"]]
                                  and self._safety(after(a))[2]]
                if not non_sacrifices:
                    return passes[0]
            return best
        # A lone non-urgent bomb is a sacrifice.  On a quiet low board, retain
        # it and pass; urgency and opponent danger were handled above.
        safe_bombs = [a for a in bombs if self._safety(after(a))[2]]
        if safe_bombs and passes:
            opponent_counts = self._opponent_counts(message)
            danger = any(0 < count <= 2 for player, count in opponent_counts.items()
                         if player != message["player"])
            board_rank = message.get("top", {}).get("rank") if message.get("top") else None
            low_board = board_rank is not None and RANK_VALUE[board_rank] <= RANK_VALUE["10"]
            if not danger and low_board:
                return passes[0]
        return safe_bombs[0] if safe_bombs else passes[0]

    def respond(self, message: Mapping[str, Any]) -> dict[str, Any]:
        request = message.get("request")
        if not request or not request.get("legal_actions"):
            raise ValueError("no legal action requested")
        actions = request["legal_actions"]
        phase = request["type"]
        if phase == "draft":
            start = next(event for event in reversed(message["events"]) if event["type"] == "round_started")
            return max(actions, key=lambda a: RANK_VALUE[start["face_up"][a["pile"]]["rank"]])
        if phase == "trade_request":
            return self._trade_rank(message, actions)
        if phase == "trade_return":
            return self._trade_return(message, actions)
        if phase == "completion":
            return self._completion(message, actions)
        return self._play(message, actions)


class ExperiencedAgent2(ExperiencedAgent):
    """Public-information policy with soft hand costs and one-lead lookahead.

    Response probabilities assume unseen cards are distributed uniformly.
    They are heuristic estimates, not deductions about an opponent's hand.
    Drafting retains the original policy because only the pile tops are known.
    """

    @classmethod
    def _unseen(cls, message: Mapping[str, Any]) -> Counter[str]:
        unseen = Counter({rank: 4 for rank in RANK_VALUE})
        unseen.subtract(cls._counts(message["hand"]))
        for event in message.get("events", []):
            if event["type"] in {"opening_play", "played", "completed"}:
                unseen.subtract(cls._counts(event.get("cards", [])))
            elif event["type"] == "bombed":
                unseen[event["card"]["rank"]] -= 1
        return Counter({rank: max(0, n) for rank, n in unseen.items()})

    @staticmethod
    def _holding(pool: int, copies: int, hand_size: int, needed: int) -> float:
        """Hypergeometric probability of holding at least needed copies."""
        hand_size = min(hand_size, pool)
        if needed > min(copies, hand_size) or pool <= 0:
            return 0.0
        return sum(comb(copies, k) * comb(pool - copies, hand_size - k)
                   for k in range(needed, min(copies, hand_size) + 1)
                   if 0 <= hand_size - k <= pool - copies) / comb(pool, hand_size)

    def _opponents(self, message: Mapping[str, Any]) -> list[tuple[str, int]]:
        counts = self._opponent_counts(message)
        active = message.get("active_players", message.get("players", []))
        seats = list(message.get("players", []))
        if message.get("player") in seats:
            i = seats.index(message["player"])
            seats = seats[i + 1:] + seats[:i]
        return [(p, max(0, counts[p])) for p in seats
                if p in active and p != message["player"] and counts[p] > 0]

    def _control(self, rank: str, width: int, unseen: Counter[str],
                 opponents: list[tuple[str, int]]) -> float:
        """Estimate keeping the lead; a quad clears before anyone responds."""
        if width == 4:
            return 1.0
        pool = sum(unseen.values())
        control = 1.0
        for _, size in opponents:
            response = self._holding(pool, unseen["2"], size, 1)
            response += sum(self._holding(pool, unseen[r], size, width)
                            for r in RANK_VALUE if r != "2" and RANK_VALUE[r] >= RANK_VALUE[rank])
            # Completion is possible out of turn and has a different width.
            response += self._holding(pool, unseen[rank], size, 4 - width)
            control *= 1.0 - min(1.0, response)
        return control

    def _hand_cost(self, counts: Counter[str], unseen: Counter[str],
                   opponents: list[tuple[str, int]]) -> float:
        ordinary = {r: n for r, n in counts.items() if r != "2" and n > 0}
        twos = counts["2"]
        if not ordinary:
            return 0.0 if not twos else 60.0 + twos
        # Groups represent future turns, but control and singles change their cost.
        cost = sum(2.5 + 0.18 * n + (0.45 if n == 1 else 0.0)
                   - 1.7 * self._control(r, n, unseen, opponents)
                   for r, n in ordinary.items())
        # Splitting pairs/triples provides cover, at the price of extra turns.
        cover = sum(n for n in ordinary.values())
        cost += 2.0 * max(0, twos - (len(ordinary) - 1))
        cost += 5.0 * max(0, twos - (cover - 1))
        return cost - 0.8 * twos

    def _lead_cost(self, counts: Counter[str], unseen: Counter[str],
                   opponents: list[tuple[str, int]]) -> float:
        """Evaluate the best forced lead after a bomb or completion."""
        choices = []
        for rank, n in counts.items():
            if rank == "2" or n <= 0:
                continue
            for width in range(1, n + 1):
                rest = counts.copy()
                rest[rank] -= width
                if sum(rest.values()) == 0:
                    choices.append(-20.0)
                    continue
                control = self._control(rank, width, unseen, opponents)
                value = self._hand_cost(rest, unseen, opponents) - 1.5 * control
                # If nobody beats this lead, we must lead again; all-2s or
                # a last ordinary group with bombs is a dangerous continuation.
                ordinary_cards = sum(v for r, v in rest.items() if r != "2")
                if rest["2"] and ordinary_cards <= rest["2"]:
                    value += 8.0 * control
                choices.append(value)
        return min(choices) if choices else self._hand_cost(counts, unseen, opponents)

    def _choose_play(self, message: Mapping[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any]:
        unseen = self._unseen(message)
        opponents = self._opponents(message)
        hand = self._counts(message["hand"])
        top = message.get("top") or {}
        pool = sum(unseen.values())
        cache: dict[tuple[str, int], float] = {}

        def score(action: dict[str, Any]) -> float:
            kind = action["type"]
            if kind in {"pass", "decline_completion"}:
                danger = sum(1.0 / size for _, size in opponents if size <= 3)
                return self._hand_cost(hand, unseen, opponents) + 0.4 + 2.0 * danger
            cards = self._action_cards(action)
            rank, width = cards[0]["rank"], len(cards)
            key = (kind + rank, width)
            if key in cache:
                return cache[key]
            rest = hand.copy()
            rest[rank] -= width
            if not sum(rest.values()):
                return 100.0 if kind == "bomb" else -100.0
            clears = kind in {"bomb", "complete"} or width == 4 or (
                rank == top.get("rank") and top.get("stack_count", 0) + width == 4)
            if clears:
                value = 0.35 + self._lead_cost(rest, unseen, opponents)
                ordinary_cards = sum(v for r, v in hand.items() if r != "2")
                if kind == "bomb" and hand["2"] > ordinary_cards:
                    # Even splitting every group cannot fund these bombs.
                    # Make progress toward the earliest penalty placement
                    # rather than waiting for a fully repairable position.
                    value = min(value, self._hand_cost(rest, unseen, opponents))
            else:
                control = self._control(rank, width, unseen, opponents)
                value = self._hand_cost(rest, unseen, opponents) - 1.5 * control
                # A matching response skips the next active seat, but that
                # player can still complete out of turn.
                for i, (_, size) in enumerate(opponents):
                    finish = 0.0
                    if size == width and not (i == 0 and rank == top.get("rank")):
                        finish += sum(self._holding(pool, unseen[r], size, width)
                                      for r in RANK_VALUE if r != "2" and RANK_VALUE[r] >= RANK_VALUE[rank])
                    stack = width + (top.get("stack_count", 0) if rank == top.get("rank") else 0)
                    if size == 4 - stack:
                        finish += self._holding(pool, unseen[rank], size, size)
                    value += 12.0 * min(1.0, finish)
                ordinary_cards = sum(v for r, v in rest.items() if r != "2")
                if rest["2"] and ordinary_cards <= rest["2"]:
                    value += 8.0 * control
            cache[key] = value
            return value

        return min(actions, key=score)

    def _play(self, message: Mapping[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any]:
        return self._choose_play(message, actions)

    def _completion(self, message: Mapping[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any]:
        return self._choose_play(message, actions)

    def _trade_value(self, counts: Counter[str], returned: str, unseen: Counter[str],
                     opponents: list[tuple[str, int]]) -> float:
        # Charge a modest price for strengthening the recipient, especially
        # with a bomb. The recipient's private matching cards are unknown.
        gift = 1.2 if returned == "2" else 0.035 * RANK_VALUE[returned]
        return self._hand_cost(counts, unseen, opponents) + gift

    def _trade_return(self, message: Mapping[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any]:
        unseen, opponents = self._unseen(message), self._opponents(message)
        def score(action):
            counts = self._counts(message["hand"])
            rank = action["card"]["rank"]
            counts[rank] -= 1
            outside = unseen.copy()
            outside[rank] += 1
            return self._trade_value(counts, rank, outside, opponents)
        return min(actions, key=score)

    def _trade_rank(self, message: Mapping[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any]:
        hand = self._counts(message["hand"])
        unseen, opponents = self._unseen(message), self._opponents(message)
        failed = self._failed_requests(message)
        candidates = [a for a in actions if a["rank"] not in failed and hand[a["rank"]] < 4]
        def score(action):
            rank = action["rank"]
            received = hand.copy()
            received[rank] += 1
            values = []
            for returned, n in received.items():
                if n <= 0:
                    continue
                final = received.copy()
                final[returned] -= 1
                outside = unseen.copy()
                outside[rank] = max(0, outside[rank] - 1)
                outside[returned] += 1
                values.append(self._trade_value(final, returned, outside, opponents))
            return min(values), -RANK_VALUE[rank]
        return min(candidates, key=score)
