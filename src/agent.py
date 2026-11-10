from collections import Counter
from functools import lru_cache
from math import comb, isfinite
from typing import Any, Mapping, Optional
import random

from src.president import Card, RANK_VALUE, Round, standard_deck
from src.solver import solve_round


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

    CONTROL_RESERVE_WEIGHT = 0.0
    GROUP_WEIGHT = 2.5
    CARD_WEIGHT = 0.18
    SINGLE_WEIGHT = 0.45
    HAND_CONTROL_WEIGHT = 1.7
    BOMB_GROUP_PENALTY = 2.0
    BOMB_COVER_PENALTY = 5.0
    BOMB_VALUE = 0.8
    LEAD_CONTROL_WEIGHT = 1.5
    BOMB_STRANDING_PENALTY = 8.0
    PASS_COST = 0.4
    PASS_DANGER_WEIGHT = 2.0
    CLEAR_COST = 0.35
    OPPONENT_FINISH_PENALTY = 12.0
    TRADE_BOMB_COST = 1.2
    TRADE_RANK_WEIGHT = 0.035

    def __init__(self, *, evaluation_backend: str = "auto"):
        if evaluation_backend not in {"auto", "python", "c"}:
            raise ValueError("evaluation_backend must be auto, python, or c")
        self.evaluation_backend = evaluation_backend

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
    @lru_cache(maxsize=8192)
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
        return self._control_cached(rank, width, tuple(unseen[r] for r in RANK_VALUE),
                                    tuple(size for _, size in opponents))

    @classmethod
    @lru_cache(maxsize=8192)
    def _control_cached(cls, rank: str, width: int, unseen_counts: tuple[int, ...],
                        opponent_sizes: tuple[int, ...]) -> float:
        # Snapshot all probability inputs: callers mutate Counters during trades.
        # Player identities do not affect control; preserve size order for exact
        # floating-point equivalence. Bounded caches can safely span decisions.
        if width == 4:
            return 1.0
        unseen = dict(zip(RANK_VALUE, unseen_counts))
        pool = sum(unseen.values())
        control = 1.0
        for size in opponent_sizes:
            response = cls._holding(pool, unseen["2"], size, 1)
            response += sum(cls._holding(pool, unseen[r], size, width)
                            for r in RANK_VALUE if r != "2" and RANK_VALUE[r] >= RANK_VALUE[rank])
            # Completion is possible out of turn and has a different width.
            response += cls._holding(pool, unseen[rank], size, 4 - width)
            control *= 1.0 - min(1.0, response)
        return control

    def _hand_cost(self, counts: Counter[str], unseen: Counter[str],
                   opponents: list[tuple[str, int]]) -> float:
        return self._hand_cost_cached(tuple(counts.items()), tuple(unseen[r] for r in RANK_VALUE),
                                      tuple(size for _, size in opponents))

    @lru_cache(maxsize=8192)
    def _hand_cost_cached(self, hand_items: tuple[tuple[str, int], ...],
                          unseen_counts: tuple[int, ...], opponent_sizes: tuple[int, ...]) -> float:
        from src.agent_search_c import hand_value
        native = hand_value(self, hand_items, unseen_counts, opponent_sizes, lead=False)
        if native is not None:
            return native
        # Preserve hand iteration order so floating-point sums and ties stay exact.
        counts = Counter(dict(hand_items))
        unseen = Counter(dict(zip(RANK_VALUE, unseen_counts)))
        opponents = [(str(i), size) for i, size in enumerate(opponent_sizes)]
        ordinary = {r: n for r, n in counts.items() if r != "2" and n > 0}
        twos = counts["2"]
        if not ordinary:
            return 0.0 if not twos else 60.0 + twos
        # Groups represent future turns, but control and singles change their cost.
        # Reuse these probabilities for the reserve bonus, preserving group
        # and arithmetic order so scores and tie-breaking stay identical.
        controls = [self._control(r, n, unseen, opponents)
                    for r, n in ordinary.items()]
        cost = sum(self.GROUP_WEIGHT + self.CARD_WEIGHT * n + (self.SINGLE_WEIGHT if n == 1 else 0.0)
                   - self.HAND_CONTROL_WEIGHT * control
                   for n, control in zip(ordinary.values(), controls))
        if self.CONTROL_RESERVE_WEIGHT:
            # Value control more when it supports other vulnerable groups.
            # Squaring emphasizes reliable control; saturation gives extra
            # control groups diminishing value, without rank/phase cutoffs.
            reserve = sum(p * p for p in controls)
            support = sum(p * p * sum(1 - q for j, q in enumerate(controls) if i != j)
                          for i, p in enumerate(controls))
            # Saving control matters most while opponents still have work
            # left. Reduce that preference continuously as the round advances
            # and more sharply when even one opponent is near a finish.
            # Normalize to 1 at the initial 13-card hands; clamp temporary
            # trade hands so the bonus never exceeds its configured weight.
            sizes = [min(13, max(0, size)) for size in opponent_sizes]
            patience = 0.0
            if sizes:
                fullness = sum(sizes) / (13 * len(sizes))
                closest = min(sizes)
                patience = fullness * (16 * closest / (13 * (closest + 3)))
            cost -= self.CONTROL_RESERVE_WEIGHT * patience * support / ((1 + reserve) * len(controls))
        # Splitting pairs/triples provides cover, at the price of extra turns.
        cover = sum(n for n in ordinary.values())
        cost += self.BOMB_GROUP_PENALTY * max(0, twos - (len(ordinary) - 1))
        cost += self.BOMB_COVER_PENALTY * max(0, twos - (cover - 1))
        return cost - self.BOMB_VALUE * twos

    def _lead_cost(self, counts: Counter[str], unseen: Counter[str],
                   opponents: list[tuple[str, int]]) -> float:
        """Evaluate the best forced lead after a bomb or completion."""
        return self._lead_cost_cached(tuple(counts.items()), tuple(unseen[r] for r in RANK_VALUE),
                                      tuple(size for _, size in opponents))

    @lru_cache(maxsize=8192)
    def _lead_cost_cached(self, hand_items: tuple[tuple[str, int], ...],
                          unseen_counts: tuple[int, ...], opponent_sizes: tuple[int, ...]) -> float:
        from src.agent_search_c import hand_value
        native = hand_value(self, hand_items, unseen_counts, opponent_sizes, lead=True)
        if native is not None:
            return native
        counts = Counter(dict(hand_items))
        unseen = Counter(dict(zip(RANK_VALUE, unseen_counts)))
        opponents = [(str(i), size) for i, size in enumerate(opponent_sizes)]
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
                value = self._hand_cost(rest, unseen, opponents) - self.LEAD_CONTROL_WEIGHT * control
                # If nobody beats this lead, we must lead again; all-2s or
                # a last ordinary group with bombs is a dangerous continuation.
                ordinary_cards = sum(v for r, v in rest.items() if r != "2")
                if rest["2"] and ordinary_cards <= rest["2"]:
                    value += self.BOMB_STRANDING_PENALTY * control
                choices.append(value)
        return min(choices) if choices else self._hand_cost(counts, unseen, opponents)

    def _turn_order_danger(self, action: Mapping[str, Any], top: Mapping[str, Any],
                           unseen: Counter[str], opponents: list[tuple[str, int]]) -> float:
        """Return an optional candidate-specific danger cost.

        The base policy deliberately has no turn-order feature.  Subclasses
        can add one without changing ExperiencedAgent2/3 decisions.
        """
        return 0.0

    def _choose_play(self, message: Mapping[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any]:
        for action in actions:
            if action["type"] in {"play", "complete"} and len(action["cards"]) == len(message["hand"]):
                return action
        return min(actions, key=self._play_scorer(message))

    def _play_scorer(self, message: Mapping[str, Any]):
        """Decision-local scorer shared by the policy and shallow search."""
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
                return (self._hand_cost(hand, unseen, opponents) + self.PASS_COST
                        + self.PASS_DANGER_WEIGHT * danger
                        + self._turn_order_danger(action, top, unseen, opponents))
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
                value = self.CLEAR_COST + self._lead_cost(rest, unseen, opponents)
                ordinary_cards = sum(v for r, v in hand.items() if r != "2")
                if kind == "bomb" and hand["2"] > ordinary_cards:
                    # Even splitting every group cannot fund these bombs.
                    # Make progress toward the earliest penalty placement
                    # rather than waiting for a fully repairable position.
                    value = min(value, self._hand_cost(rest, unseen, opponents))
            else:
                control = self._control(rank, width, unseen, opponents)
                value = self._hand_cost(rest, unseen, opponents) - self.LEAD_CONTROL_WEIGHT * control
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
                    value += self.OPPONENT_FINISH_PENALTY * min(1.0, finish)
                ordinary_cards = sum(v for r, v in rest.items() if r != "2")
                if rest["2"] and ordinary_cards <= rest["2"]:
                    value += self.BOMB_STRANDING_PENALTY * control
            value += self._turn_order_danger(action, top, unseen, opponents)
            cache[key] = value
            return value

        return score

    def _play(self, message: Mapping[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any]:
        return self._choose_play(message, actions)

    def _completion(self, message: Mapping[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any]:
        return self._choose_play(message, actions)

    def _trade_value(self, counts: Counter[str], returned: str, unseen: Counter[str],
                     opponents: list[tuple[str, int]]) -> float:
        # Charge a modest price for strengthening the recipient, especially
        # with a bomb. The recipient's private matching cards are unknown.
        gift = self.TRADE_BOMB_COST if returned == "2" else self.TRADE_RANK_WEIGHT * RANK_VALUE[returned]
        return self._hand_cost(counts, unseen, opponents) + gift

    def _trade_fast_path_supported(self) -> bool:
        """Whether trade scoring can bypass dynamically dispatched helpers.

        The lookup scorer below is deliberately only an implementation of this
        class's existing scoring expression.  It must not bypass a subclass or
        instance patch which changes any helper used by the direct evaluator.
        Weight-only subclasses remain eligible.
        """
        helpers = ("_holding", "_control", "_control_cached", "_hand_cost",
                   "_hand_cost_cached", "_trade_value")
        if any(name in self.__dict__ for name in helpers):
            return False
        for name in helpers:
            actual = getattr(type(self), name)
            expected = getattr(ExperiencedAgent2, name)
            if getattr(actual, "__func__", actual) is not getattr(expected, "__func__", expected):
                return False
        weights = ("CONTROL_RESERVE_WEIGHT", "GROUP_WEIGHT", "CARD_WEIGHT", "SINGLE_WEIGHT",
                   "HAND_CONTROL_WEIGHT", "BOMB_GROUP_PENALTY", "BOMB_COVER_PENALTY",
                   "BOMB_VALUE", "TRADE_BOMB_COST", "TRADE_RANK_WEIGHT")
        return all(isinstance(getattr(self, name), (int, float))
                   and not isinstance(getattr(self, name), bool)
                   and isfinite(getattr(self, name)) for name in weights)

    @staticmethod
    def _trade_counts_supported(counts: Counter[str]) -> bool:
        """Check the compact lookup domain without changing public behavior."""
        return (all(rank in RANK_VALUE and isinstance(n, int) and not isinstance(n, bool)
                    and 0 <= n <= 4 for rank, n in counts.items())
                and all(0 <= counts[rank] <= 4 for rank in RANK_VALUE))

    def _trade_lookup_cost(self, pool: int, opponents: list[tuple[str, int]]):
        """Build an order-preserving hand-cost evaluator for one trade decision.

        Only calls to ``_holding`` are replaced with decision-local lookups.
        The response expression, group iteration, reserve calculation, and
        final arithmetic intentionally match ``_hand_cost_cached`` exactly.
        """
        ranks = tuple(RANK_VALUE)
        rank_index = {rank: index for index, rank in enumerate(ranks)}
        two_index = rank_index["2"]
        sizes = tuple(size for _, size in opponents)
        unique_sizes = tuple(dict.fromkeys(sizes))
        size_index = {size: index for index, size in enumerate(unique_sizes)}
        opponent_size_indexes = tuple(size_index[size] for size in sizes)
        # Indexed tables avoid tuple construction and hashing in the candidate
        # loop. Entry zero is unused because required widths start at one.
        holding = [[[0.0] + [self._holding(pool, copies, size, needed)
                              for needed in (1, 2, 3)]
                    for copies in range(min(4, pool) + 1)]
                   for size in unique_sizes]
        group_cost = [0.0] + [self.GROUP_WEIGHT + self.CARD_WEIGHT * count
                               + (self.SINGLE_WEIGHT if count == 1 else 0.0)
                               for count in range(1, 5)]
        reserve_weight = 0.0
        if self.CONTROL_RESERVE_WEIGHT:
            sizes_for_patience = [min(13, max(0, size)) for size in sizes]
            if sizes_for_patience:
                fullness = sum(sizes_for_patience) / (13 * len(sizes_for_patience))
                closest = min(sizes_for_patience)
                patience = fullness * (16 * closest / (13 * (closest + 3)))
                reserve_weight = self.CONTROL_RESERVE_WEIGHT * patience

        def control(rank_index_: int, width: int, unseen: list[int]) -> float:
            if width == 4:
                return 1.0
            factors = []
            for size_table in holding:
                response = size_table[unseen[two_index]][1]
                response += sum(size_table[unseen[other_index]][width]
                                for other_index in range(rank_index_, two_index))
                response += size_table[unseen[rank_index_]][4 - width]
                factors.append(1.0 - min(1.0, response))
            value = 1.0
            for size_index_ in opponent_size_indexes:
                value *= factors[size_index_]
            return value

        def hand_cost(counts: list[int] | Counter[str], unseen: list[int] | Counter[str],
                      order: tuple[int, ...] | None = None) -> float:
            # The Counter form is retained for direct differential tests. The
            # trade paths pass mutable arrays and an existing rank order, so
            # no Counter or cache-key tuple is built per candidate.
            if isinstance(counts, Counter):
                order = tuple(rank_index[rank] for rank in counts)
                counts = [counts[rank] for rank in ranks]
                unseen = [unseen[rank] for rank in ranks]
            assert order is not None
            ordinary = [(index, counts[index]) for index in order
                        if index != two_index and counts[index] > 0]
            twos = counts[two_index]
            if not ordinary:
                return 0.0 if not twos else 60.0 + twos
            controls = [control(index, count, unseen) for index, count in ordinary]
            cost = sum(group_cost[n]
                       - self.HAND_CONTROL_WEIGHT * value
                       for (_, n), value in zip(ordinary, controls))
            if reserve_weight:
                reserve = sum(value * value for value in controls)
                support = sum(value * value * sum(1 - other for j, other in enumerate(controls) if i != j)
                              for i, value in enumerate(controls))
                cost -= reserve_weight * support / ((1 + reserve) * len(controls))
            cover = sum(count for _, count in ordinary)
            cost += self.BOMB_GROUP_PENALTY * max(0, twos - (len(ordinary) - 1))
            cost += self.BOMB_COVER_PENALTY * max(0, twos - (cover - 1))
            return cost - self.BOMB_VALUE * twos

        return hand_cost

    def _trade_return_direct(self, message: Mapping[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any]:
        unseen, opponents = self._unseen(message), self._opponents(message)
        by_rank = {}
        for action in actions:
            by_rank.setdefault(action["card"]["rank"], action)
        def score(action):
            counts = self._counts(message["hand"])
            rank = action["card"]["rank"]
            counts[rank] -= 1
            outside = unseen.copy()
            outside[rank] += 1
            return self._trade_value(counts, rank, outside, opponents)
        return min(by_rank.values(), key=score)

    def _trade_rank_direct(self, message: Mapping[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any]:
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

    def _trade_return(self, message: Mapping[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any]:
        """Choose a return card using lookups when the direct expression is safe."""
        if not self._trade_fast_path_supported():
            return self._trade_return_direct(message, actions)
        hand, unseen = self._counts(message["hand"]), self._unseen(message)
        opponents = self._opponents(message)
        by_rank = {}
        for action in actions:
            by_rank.setdefault(action["card"]["rank"], action)
        if (not self._trade_counts_supported(hand) or not self._trade_counts_supported(unseen)
                or any(not isinstance(size, int) or isinstance(size, bool) or size < 0
                       for _, size in opponents)):
            return self._trade_return_direct(message, actions)
        ranks = tuple(RANK_VALUE)
        rank_index = {rank: index for index, rank in enumerate(ranks)}
        for action in by_rank.values():
            rank = action["card"]["rank"]
            if rank not in RANK_VALUE or hand[rank] <= 0:
                return self._trade_return_direct(message, actions)
        if any(hand[rank] + unseen[rank] > 4 for rank in ranks):
            return self._trade_return_direct(message, actions)
        cost = self._trade_lookup_cost(sum(unseen.values()) + 1, opponents)
        hand_counts = [hand[rank] for rank in ranks]
        unseen_counts = [unseen[rank] for rank in ranks]
        hand_order = tuple(rank_index[rank] for rank in hand)

        def score(action: dict[str, Any]) -> float:
            rank = action["card"]["rank"]
            index = rank_index[rank]
            hand_counts[index] -= 1
            unseen_counts[index] += 1
            value = cost(hand_counts, unseen_counts, hand_order)
            hand_counts[index] += 1
            unseen_counts[index] -= 1
            return value + (self.TRADE_BOMB_COST if rank == "2"
                            else self.TRADE_RANK_WEIGHT * RANK_VALUE[rank])

        return min(by_rank.values(), key=score)

    def _trade_rank(self, message: Mapping[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any]:
        """Choose a request rank with a decision-local probability lookup table."""
        if not self._trade_fast_path_supported():
            return self._trade_rank_direct(message, actions)
        failed = self._failed_requests(message)
        hand, unseen = self._counts(message["hand"]), self._unseen(message)
        opponents = self._opponents(message)
        candidates = [action for action in actions if action["rank"] not in failed
                      and action["rank"] in RANK_VALUE and hand[action["rank"]] < 4]
        if (not self._trade_counts_supported(hand) or not self._trade_counts_supported(unseen)
                or any(hand[rank] + unseen[rank] > 4 for rank in RANK_VALUE)
                or any(not isinstance(size, int) or isinstance(size, bool) or size < 0
                       for _, size in opponents)
                or any(unseen[action["rank"]] <= 0 for action in candidates)):
            return self._trade_rank_direct(message, actions)
        # Preserve the direct ValueError for an empty candidate set.
        if not candidates:
            return self._trade_rank_direct(message, actions)
        ranks = tuple(RANK_VALUE)
        rank_index = {rank: index for index, rank in enumerate(ranks)}
        cost = self._trade_lookup_cost(sum(unseen.values()), opponents)
        hand_counts = [hand[rank] for rank in ranks]
        unseen_counts = [unseen[rank] for rank in ranks]
        hand_order = tuple(rank_index[rank] for rank in hand)

        def score(action: dict[str, Any]) -> tuple[float, int]:
            requested = action["rank"]
            request_index = rank_index[requested]
            request_order = hand_order if hand_counts[request_index] else hand_order + (request_index,)
            hand_counts[request_index] += 1
            unseen_counts[request_index] -= 1
            best = None
            for returned_index in request_order:
                if hand_counts[returned_index] <= 0:
                    continue
                hand_counts[returned_index] -= 1
                unseen_counts[returned_index] += 1
                returned = ranks[returned_index]
                value = cost(hand_counts, unseen_counts, request_order)
                value += (self.TRADE_BOMB_COST if returned == "2"
                          else self.TRADE_RANK_WEIGHT * RANK_VALUE[returned])
                hand_counts[returned_index] += 1
                unseen_counts[returned_index] -= 1
                best = value if best is None or value < best else best
            hand_counts[request_index] -= 1
            unseen_counts[request_index] += 1
            return best, -RANK_VALUE[requested]

        return min(candidates, key=score)


class ExperiencedAgent3(ExperiencedAgent2):
    """ExperiencedAgent2 with exact play in a two-player endgame.

    The normal protocol keeps hands private, but no hidden-card guess is
    needed once exactly two players remain: every card absent from this
    player's hand and the public play history belongs to the other active
    player.  This policy rebuilds that small, fully-known position and uses
    :func:`src.solver.solve_round`; all earlier decisions retain
    :class:`ExperiencedAgent2`'s policy.
    """

    CONTROL_RESERVE_WEIGHT = 1.5

    @staticmethod
    def _publicly_played(message: Mapping[str, Any]) -> list[Card]:
        cards: list[Card] = []
        for event in message.get("events", []):
            if event.get("type") in {"opening_play", "played", "completed"}:
                cards.extend(Card.from_json(card) for card in event.get("cards", []))
            elif event.get("type") == "bombed":
                cards.append(Card.from_json(event["card"]))
        return cards

    @staticmethod
    def _completion_source(events: list[Mapping[str, Any]]) -> tuple[str, bool] | None:
        """Return the action that opened the current completion window."""
        for index in range(len(events) - 1, -1, -1):
            event = events[index]
            kind = event.get("type")
            if kind not in {"opening_play", "played", "passed"}:
                continue
            player = event.get("player")
            if not isinstance(player, str):
                return None
            if kind != "played":
                return player, False
            rank = event["cards"][0]["rank"]
            previous_rank = None
            for previous in reversed(events[:index]):
                if previous.get("type") == "board_cleared":
                    break
                if previous.get("type") in {"opening_play", "played"}:
                    previous_rank = previous["cards"][0]["rank"]
                    break
            return player, rank == previous_rank
        return None

    @classmethod
    def _endgame_round(cls, message: Mapping[str, Any]) -> Round | None:
        """Hydrate precisely the fields the solver needs from a wire snapshot."""
        active = list(message.get("active_players", []))
        player = message.get("player")
        phase = message.get("request", {}).get("type")
        if len(active) != 2 or player not in active or phase not in {"play", "completion"}:
            return None

        # Announced 2s leave a finished player's hand without exposing their
        # card identities or count.  They therefore cannot be assigned to the
        # remaining opponent exactly from the private protocol snapshot.
        if any(event.get("type") == "twos_announced" for event in message.get("events", [])):
            return None

        try:
            own_hand = [Card.from_json(card) for card in message["hand"]]
            played = cls._publicly_played(message)
        except (KeyError, TypeError, ValueError):
            return None
        # A valid game history partitions the deck into cards in our hand,
        # public cards already played, and the remaining opponent hand.
        known = own_hand + played
        if len(known) != len(set(known)):
            return None
        deck = set(standard_deck())
        if any(card not in deck for card in known):
            return None
        opponent = next(name for name in active if name != player)
        opponent_hand = sorted(deck - set(known), key=lambda c: (RANK_VALUE[c.rank], c.suit))
        if not opponent_hand:
            return None

        players = list(message.get("players", []))
        if len(players) != 4 or set(active) - set(players):
            return None
        round_ = Round(players, first_round=True, deck=standard_deck())
        round_.first_round = False  # This is necessarily past the opening play.
        round_.hands = {name: [] for name in players}
        round_.hands[player] = own_hand
        round_.hands[opponent] = opponent_hand
        round_.active = set(active)
        round_.phase = phase
        # A request is only sent to the current actor, so compact hosts that
        # omit the redundant public ``turn`` field remain solver-compatible.
        round_.turn = message.get("turn", player)
        top = message.get("top")
        round_.top_rank = None if top is None else top.get("rank")
        round_.top_count = 0 if top is None else top.get("count", 0)
        round_.top_stack_count = 0 if top is None else top.get("stack_count", 0)
        events = list(message.get("events", []))
        round_.events = events if any(e.get("type") == "opening_play" for e in events) else [{"type": "opening_play"}]
        last_play = next((e for e in reversed(events)
                          if e.get("type") in {"opening_play", "played"}), None)
        round_.last_player = None if last_play is None else last_play.get("player")

        if phase == "completion":
            source = cls._completion_source(events)
            if source is None or round_.turn not in active:
                return None
            previous, skip = source
            order = ([previous] if previous in round_.active else []) + [
                p for p in round_._active_after(previous) if p != previous]
            if round_.turn not in order:
                return None
            round_.completion_order = order
            round_._completion_cursor = order.index(round_.turn)
            round_._pending_advance = (previous, skip)
        return round_

    def respond(self, message: Mapping[str, Any]) -> dict[str, Any]:
        try:
            round_ = self._endgame_round(message)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            # Keep the parent policy available when a host sends a partial or
            # malformed history that cannot be reconstructed exactly.
            round_ = None
        if round_ is not None:
            try:
                result = solve_round(round_)
                actions = message["request"]["legal_actions"]
                if result.player == message["player"] and result.best_action in actions:
                    return result.best_action
            except (IndexError, KeyError, TypeError, ValueError):
                # Partial protocol snapshots remain supported by the parent
                # heuristic policy; exact play only needs a complete history.
                pass
        return super().respond(message)


class ExperiencedAgent3Tuned(ExperiencedAgent3):
    """ExperiencedAgent3 with weights selected by the ES checkpoint.

    The policy retains ExperiencedAgent3's exact two-player endgame solver.
    Only the inherited heuristic scoring weights differ. Values are copied from
    ``results/experienced3_es.json`` so arena runners can import this policy
    without depending on a generated results module.
    """

    CONTROL_RESERVE_WEIGHT = 1.7336002666902222
    GROUP_WEIGHT = 3.192850798859009
    CARD_WEIGHT = 0.0
    SINGLE_WEIGHT = 0.8410903295588946
    HAND_CONTROL_WEIGHT = 2.2238177323977313
    BOMB_GROUP_PENALTY = 2.0176728858951187
    BOMB_COVER_PENALTY = 4.884632378500758
    BOMB_VALUE = 0.8564299706364091
    LEAD_CONTROL_WEIGHT = 1.0164562907204948
    BOMB_STRANDING_PENALTY = 8.874649688289717
    PASS_COST = 0.2734239876318853
    PASS_DANGER_WEIGHT = 2.336944568174645
    CLEAR_COST = 0.3162716698012512
    OPPONENT_FINISH_PENALTY = 14.661944128451612
    TRADE_BOMB_COST = 1.1298037168270516
    TRADE_RANK_WEIGHT = 0.03073388581995172


class ExperiencedAgent4(ExperiencedAgent3Tuned):
    """Experimental successor to :class:`ExperiencedAgent3Tuned`.

    Adds turn-order danger and bounded, policy-guided Monte Carlo evaluation.
    The inherited exact two-player solver runs outside the search budget.
    """

    # The existing policy rewards global control.  These weights separately
    # price the more urgent risk from the opponent who acts next.
    NEXT_RESPONSE_PENALTY = 1.0
    NEXT_FINISH_PENALTY = 6.0
    NEXT_COMPLETION_PENALTY = 1.5

    def __init__(self, rng: Optional[random.Random] = None, *,
                 search_time_ms: float | None = 50.0, search_worlds: int = 8,
                 search_backend: str = "auto", evaluation_backend: str = "auto"):
        super().__init__(evaluation_backend=evaluation_backend)
        if search_backend not in {"auto", "python", "c"}:
            raise ValueError("search_backend must be auto, python, or c")
        if search_time_ms is not None and (not isfinite(search_time_ms) or search_time_ms < 0):
            raise ValueError("search_time_ms must be nonnegative or None")
        if isinstance(search_worlds, bool) or not isinstance(search_worlds, int) or search_worlds < 0:
            raise ValueError("search_worlds must be a nonnegative integer")
        self.rng = rng if rng is not None else random.Random(0)
        self.search_time_ms = search_time_ms
        self.search_worlds = search_worlds
        self.search_backend = search_backend
        self.last_search = {}

    def _choose_play(self, message, actions):
        from src.agent_search import choose
        return choose(self, message, actions)
    NEXT_BOMB_PENALTY = 1.5

    def _turn_order_danger(self, action: Mapping[str, Any], top: Mapping[str, Any],
                           unseen: Counter[str], opponents: list[tuple[str, int]]) -> float:
        """Estimate the next opponent's immediate tactical options.

        Probabilities use the same public-information hypergeometric model as
        the parent policy.  Rank events overlap, so the summed probabilities
        are capped at one; this is a conservative heuristic, not an exact
        joint distribution.
        """
        if not opponents or action["type"] in {"bomb", "complete"}:
            return 0.0

        kind = action["type"]
        if kind in {"pass", "decline_completion"}:
            if not top:
                return 0.0
            rank = top["rank"]
            width = top["count"]
            stack = top.get("stack_count", width)
            next_gets_normal_turn = True
        else:
            cards = self._action_cards(action)
            rank, width = cards[0]["rank"], len(cards)
            stack = width + (top.get("stack_count", 0) if rank == top.get("rank") else 0)
            if width == 4 or stack == 4:
                return 0.0
            # Matching the board opens a completion window, then skips the
            # next normal turn. That player can still complete the stack.
            next_gets_normal_turn = rank != top.get("rank")

        _, size = opponents[0]
        pool = sum(unseen.values())
        if size <= 0 or pool <= 0:
            return 0.0

        normal_response = 0.0
        bomb = 0.0
        finish = 0.0
        if next_gets_normal_turn:
            normal_response = sum(
                self._holding(pool, unseen[other], size, width)
                for other in RANK_VALUE
                if other != "2" and RANK_VALUE[other] >= RANK_VALUE[rank]
            )
            # A 2 immediately clears the board. Do not treat a final lone 2
            # as a normal tactical finish: this ruleset penalizes it instead.
            bomb = self._holding(pool, unseen["2"], size, 1) if size > 1 else 0.0
            if size == width:
                finish = normal_response

        needed = 4 - stack
        completion = self._holding(pool, unseen[rank], size, needed) if needed > 0 else 0.0
        if size == needed:
            finish = min(1.0, finish + completion)

        return (self.NEXT_RESPONSE_PENALTY * min(1.0, normal_response)
                + self.NEXT_BOMB_PENALTY * bomb
                + self.NEXT_FINISH_PENALTY * min(1.0, finish)
                + self.NEXT_COMPLETION_PENALTY * completion)


class SwitchPAgent(ExperiencedAgent):
    """Port of switch+'s kou2 evaluation/rules and keep_kou2 conservation.

    See docs/agents/switchp-agent.md for source mapping and rule changes.
    No ExperiencedAgent2 scoring, probabilities or lookahead are used. The
    six-policy tournament selector is not part of this single-policy port.
    """

    STRONG = 86

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        self.rng = rng or random.Random(0)
        self._prediction_results: dict[str, list[int]] = {}
        self._finished_seen = False

    @classmethod
    def _public_info(cls, message):
        unseen = Counter({r: 4 for r in RANK_VALUE})
        unseen.subtract(cls._counts(message["hand"]))
        for event in message.get("events", []):
            if event["type"] in {"opening_play", "played", "completed"}:
                unseen.subtract(cls._counts(event.get("cards", [])))
            elif event["type"] == "bombed":
                unseen[event["card"]["rank"]] -= 1
        counts = cls._opponent_counts(message)
        active = message.get("active_players", message.get("players", []))
        opponents = [counts[p] for p in active
                     if p != message.get("player") and counts.get(p, 0) > 0]
        return Counter({r: max(0, n) for r, n in unseen.items()}), opponents

    @classmethod
    def _strength(cls, rank, width, unseen, opponents, stack=None):
        """value_strong's integer x; 100 is the starting strength, not 100%."""
        stack = width if stack is None else stack
        if stack == 4:
            return 200  # President clear replaces the source's 8-clear bonus.
        value = 100
        players = len(opponents) + 1
        largest = max(opponents, default=0)
        for other in RANK_VALUE:
            if other == "2" or RANK_VALUE[other] < RANK_VALUE[rank]:
                continue
            copies = unseen[other]
            if min(copies, largest) < width:
                continue
            if width == 1 or players == 2:
                value -= 30
            else:
                k = copies - width - players + 5
                value -= 4 if k <= 0 else 9 if k == 1 else 15 if k == 2 else 24
        # Our bombs and out-of-turn completions can interrupt any non-clear.
        # Count them explicitly; neither is the source's wild-card Joker.
        if largest and unseen["2"]:
            value -= 30
        if min(largest, unseen[rank]) >= 4 - stack:
            value -= 30
        return max(1, value)

    @classmethod
    def _pattern(cls, counts, unseen, opponents):
        """pat_make/value_strong: rank groups, x, z, 1min and 2min."""
        groups = [(r, counts[r], cls._strength(r, counts[r], unseen, opponents))
                  for r in RANK_VALUE if r != "2" and counts[r] > 0]
        strengths = sorted(x for _, _, x in groups)
        highest = max((RANK_VALUE[r] for r in RANK_VALUE
                       if r != "2" and unseen[r]), default=0)
        z = 0
        for rank, width, x in groups:
            if width == 4:
                continue  # Source z skips groups of four or more.
            if x <= 30:
                z += 2
            elif x <= 60:
                z += 1
            elif x >= 91:
                z -= width if RANK_VALUE[rank] >= highest else 1
        return {"groups": groups, "count": len(groups), "z": z,
                "minimum": strengths[0] if strengths else 201,
                "second": strengths[1] if len(strengths) > 1 else 201}

    @classmethod
    def _lead_priority(cls, rank, width, x, pattern):
        """The source's y priorities, including its second-minimum branch."""
        if pattern["second"] >= cls.STRONG:
            if x < cls.STRONG:
                return 2
            if pattern["second"] >= 100:
                return 41 if x > 100 else x - 60
            return 9 if x > 100 else 110 - x
        if cls.STRONG <= x <= 100:
            return 110 - x
        # Source ord numbers 3 as 1, ..., A as 12.
        ordinal = RANK_VALUE[rank] - RANK_VALUE["3"] + 1
        priority = 40 + 2 * (14 - ordinal)
        priority += 8 if x > 100 else 4 * sum(n == width for _, n, _ in pattern["groups"])
        if pattern["count"] >= 5 and rank == "3":
            priority -= 7
        return priority

    @classmethod
    def _accept_follow(cls, before, after, strength, width):
        """kou_followgroup's ordered tests; return acceptance and adjusted z'."""
        z = after["z"] - int(strength >= cls.STRONG)
        if strength < cls.STRONG and after["count"] >= before["count"]:
            z += 1
        if strength >= cls.STRONG and after["second"] >= cls.STRONG:
            return True, z
        if strength > 100 and after["count"] <= before["count"]:
            return True, z
        if strength > cls.STRONG and after["count"] >= 4 and after["count"] - 1 <= z:
            return False, z
        if after["minimum"] <= 50 and after["second"] == 201 and strength <= 60:
            return False, z
        if after["second"] >= cls.STRONG:
            return True, z
        if width == 1 and z >= 4 and 40 <= strength <= 70:
            return False, z
        return z <= before["z"], z

    @staticmethod
    def _predict_leader(message):
        """keep_kou2's first-three-clears and cards-shed strength points."""
        shed, clears = Counter(), Counter()
        clear_count = 0
        for event in message.get("events", []):
            player = event.get("player")
            if event["type"] in {"opening_play", "played", "completed"}:
                shed[player] += len(event.get("cards", []))
            elif event["type"] == "bombed":
                shed[player] += 1
            elif event["type"] == "board_cleared":
                clears[player] += 1
                clear_count += 1
                if clear_count == 3:
                    break
        if clear_count < 3:
            return None
        best, best_score = None, 5
        for player in message.get("players", []):
            n = shed[player]
            score = 10 if clears[player] == 3 else 2 if clears[player] == 2 else 0
            score += 3 if n == 4 else 6 if n == 5 else 9 + 4 * (n - 6) if n >= 6 else 0
            if score > best_score:
                best, best_score = player, score
        return best

    def observe(self, message):
        finished = message.get("rankings")
        if not finished:
            self._finished_seen = False
        elif not self._finished_seen:
            self._finished_seen = True
            predicted = self._predict_leader(message)
            if predicted in finished:
                result = self._prediction_results.setdefault(predicted, [0, 0])
                result[0] += int(finished[predicted] == 1)
                result[1] += 1

    def _keep(self, message, action, unseen, opponents):
        """keep_kou2: conserve K+ on weak hands; spare strong A/2 groups."""
        rank, width = action["cards"][0]["rank"], len(action["cards"])
        if RANK_VALUE[rank] < RANK_VALUE["K"] or width == 4:
            return False
        if rank == "A" and width in (2, 3):
            return False
        events = message.get("events", [])
        if any(e["type"] in {"player_done", "twos_announced"} for e in events):
            return False
        # Reconstruct the post-trade initial hand and the hand after clear #2,
        # allowing stateless request-only hosts to use the same conditions.
        initial = self._counts(message["hand"])
        clear_count = sum(e["type"] == "board_cleared" for e in events)
        second_hand = initial.copy()
        seen_clears = clear_count
        initial_unseen = unseen.copy()
        second_unseen = unseen.copy()
        second_counts = self._opponent_counts(message)
        for event in reversed(events):
            if event["type"] == "board_cleared":
                seen_clears -= 1
            cards = event.get("cards", []) if event["type"] in {"opening_play", "played", "completed"} else []
            if event["type"] == "bombed":
                cards = [event["card"]]
            if event.get("player") == message.get("player"):
                initial.update(c["rank"] for c in cards)
                if seen_clears >= 2:
                    second_hand.update(c["rank"] for c in cards)
            else:
                initial_unseen.update(c["rank"] for c in cards)
                if seen_clears >= 2:
                    second_unseen.update(c["rank"] for c in cards)
                    if event.get("player") in second_counts:
                        second_counts[event["player"]] += len(cards)
        initial_opponents = [13 for p in message.get("players", []) if p != message.get("player")]
        second_opponents = [n for p, n in second_counts.items() if p != message.get("player") and n > 0]
        if clear_count < 3:
            played = sum(len(e.get("cards", [])) for e in events
                         if e["type"] in {"opening_play", "played", "completed"})
            played += sum(e["type"] == "bombed" for e in events)
            return played < 25 and self._pattern(initial, initial_unseen, initial_opponents)["z"] > 12
        if self._pattern(second_hand, second_unseen, second_opponents)["z"] <= 8:
            return False
        strong_cards = sum(RANK_VALUE[c["rank"]] >= RANK_VALUE["K"] for c in message["hand"])
        if strong_cards > 2 or not any(unseen[r] for r in RANK_VALUE
                                      if RANK_VALUE[r] > RANK_VALUE[rank]):
            return False
        predicted = self._predict_leader(message)
        if predicted is None or predicted == message.get("player"):
            return False
        players = message.get("players", [])
        accuracy = sum(wins / total for wins, total in self._prediction_results.values() if total)
        accuracy /= max(1, len(players))
        return self.rng.random() < accuracy

    def _play(self, message, actions):
        self.observe(message)
        hand = message["hand"]
        unseen, opponents = self._public_info(message)
        counts = self._counts(hand)
        before = self._pattern(counts, unseen, opponents)
        passes = [a for a in actions if a["type"] == "pass"]
        ordinary = sorted((a for a in actions if a["type"] == "play"),
                          key=lambda a: (RANK_VALUE[a["cards"][0]["rank"]], -len(a["cards"])))
        bombs = [a for a in actions if a["type"] == "bomb"]
        for action in ordinary:
            if len(action["cards"]) == len(hand):
                return action
        # President-only legality/penalty guard: spend bombs before cover runs out.
        if bombs and counts["2"] >= before["count"] and before["count"]:
            return bombs[0]
        viable = [a for a in ordinary if self._safety(self._without(hand, a["cards"]))[2]]
        if not passes:  # Must lead; use the original y ordering on whole groups.
            candidates = viable or ordinary
            whole = [a for a in candidates if len(a["cards"]) == counts[a["cards"][0]["rank"]]]
            return max(whole or candidates, key=lambda a: self._lead_priority(
                a["cards"][0]["rank"], len(a["cards"]),
                self._strength(a["cards"][0]["rank"], len(a["cards"]), unseen, opponents), before))
        chosen, chosen_x, chosen_z = None, 0, 0
        top = message.get("top") or {}
        for action in viable:
            rank, width = action["cards"][0]["rank"], len(action["cards"])
            stack = width + (top.get("stack_count", 0) if rank == top.get("rank") else 0)
            x = self._strength(rank, width, unseen, opponents, stack)
            after = self._pattern(self._counts(self._without(hand, action["cards"])), unseen, opponents)
            accepted, z = self._accept_follow(before, after, x, width)
            if not accepted:
                continue
            if chosen is not None and after["second"] >= self.STRONG:
                if chosen_x >= self.STRONG > x:
                    continue
                if x >= self.STRONG > chosen_x or x > self.STRONG == chosen_x:
                    chosen, chosen_x, chosen_z = action, x, z
                    continue
            if chosen is None or z < chosen_z:
                chosen, chosen_x, chosen_z = action, x, z
        if chosen is not None:
            if self._keep(message, chosen, unseen, opponents):
                return passes[0]
            return chosen
        # Bombs replace the source's standalone control-card fallback, but only
        # when the remaining ordinary hand has its near-finish shape or top is A.
        if bombs and before["count"] and (before["second"] >= self.STRONG or top.get("rank") == "A"):
            return bombs[0]
        return passes[0]

    def _completion(self, message, actions):
        # No completion window exists in switch+. Apply the same follow rules
        # with an immediate-clear strength and the President bomb-cover guard.
        unseen, opponents = self._public_info(message)
        before = self._pattern(self._counts(message["hand"]), unseen, opponents)
        for action in actions:
            if action["type"] != "complete":
                continue
            rest = self._without(message["hand"], action["cards"])
            if not rest:
                return action
            after = self._pattern(self._counts(rest), unseen, opponents)
            if self._safety(rest)[2] and self._accept_follow(before, after, 200, len(action["cards"]))[0]:
                return action
        return next(a for a in actions if a["type"] == "decline_completion")


class HumanRulesAgent3(ExperiencedAgent):
    """Fixed table-side rules extracted from ExperiencedAgent3Tuned.

    See ``docs/interpretation/experiencedagent3-human.md``. Every decision is a
    short ordered rule list over the hand's group count G, 2 count B, the
    board width, and the smallest opponent hand. With exactly two players
    left it uses the ExperiencedAgent3 exact solver, as the tuned agent does,
    and falls back to the rules when that position cannot be rebuilt.
    """

    def __init__(self, *, heads_up_solver: bool = True) -> None:
        self.heads_up_solver = heads_up_solver

    # -- shared helpers ------------------------------------------------------

    @staticmethod
    def _ordinary(counts: Counter[str]) -> list[str]:
        """Held ordinary ranks, lowest first."""
        return sorted((r for r, n in counts.items() if r != "2" and n), key=RANK_VALUE.__getitem__)

    @classmethod
    def _gb(cls, hand: list[Mapping[str, Any]]) -> tuple[int, int]:
        """Return (G, B): ordinary groups held and 2s held."""
        counts = cls._counts(hand)
        return len(cls._ordinary(counts)), counts["2"]

    @classmethod
    def _smallest_opponent(cls, message: Mapping[str, Any]) -> int:
        counts = cls._opponent_counts(message)
        active = message.get("active_players") or list(counts)
        sizes = [counts[p] for p in active
                 if p != message["player"] and counts.get(p, 0) > 0]
        return min(sizes, default=13)

    @classmethod
    def _rank(cls, action: Mapping[str, Any]) -> str:
        return cls._action_cards(action)[0]["rank"]

    @classmethod
    def _width(cls, action: Mapping[str, Any]) -> int:
        return len(cls._action_cards(action))

    @classmethod
    def _lowest(cls, options: list[dict[str, Any]]) -> dict[str, Any]:
        return min(options, key=lambda a: (RANK_VALUE[cls._rank(a)], -cls._width(a)))

    # -- trading -------------------------------------------------------------

    def _trade_rank(self, message: Mapping[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any]:
        counts = self._counts(message["hand"])
        failed = self._failed_requests(message)
        offered = {a["rank"]: a for a in actions
                   if a["rank"] not in failed and counts[a["rank"]] < 4}
        if not offered:
            offered = ({a["rank"]: a for a in actions if a["rank"] not in failed}
                       or {a["rank"]: a for a in actions})

        def desc(ranks):
            return sorted(ranks, key=RANK_VALUE.__getitem__, reverse=True)
        ordinary = [r for r in counts if r != "2" and counts[r]]
        triples = desc(r for r in ordinary if counts[r] == 3)
        singles = desc(r for r in ordinary if counts[r] == 1)
        pairs = desc(r for r in ordinary if counts[r] == 2)

        priorities: list[str] = []
        # T1: the fourth card of the highest triple. T2b: once a fourth-card
        # request has failed this exchange, move on to a 2.
        if not failed & set(triples):
            priorities += triples
        # T2: a 2, or the lone Ace when already holding two or more 2s.
        if counts["2"] >= 2 and counts["A"] == 1:
            priorities.append("A")
        priorities.append("2")
        # T2b: after a failed 2, pair the highest single; then other matches,
        # then unheld ranks from the top.
        priorities += singles + triples + pairs
        priorities += desc(r for r in RANK_VALUE if r != "2" and not counts[r])
        for rank in priorities:
            if rank in offered:
                return offered[rank]
        return max(offered.values(), key=lambda a: RANK_VALUE[a["rank"]])

    def _trade_return(self, message: Mapping[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any]:
        counts = self._counts(message["hand"])

        def key(action: dict[str, Any]) -> tuple[int, int]:
            rank = action["card"]["rank"]
            # T3: lowest single, then one card off the lowest triple, then
            # pairs and quads; a 2 only when nothing else is left.
            tier = 4 if rank == "2" else {1: 0, 3: 1, 2: 2}.get(counts[rank], 3)
            return tier, RANK_VALUE[rank]
        return min(actions, key=key)

    # -- completions ---------------------------------------------------------

    def _completion(self, message: Mapping[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any]:
        hand = message["hand"]
        for action in (a for a in actions if a["type"] == "complete"):
            remaining = self._without(hand, action["cards"])
            if not remaining:
                return action
            groups, twos = self._gb(remaining)
            # C1: decline when afterwards B >= G with B > 0.
            if not (twos > 0 and twos >= groups):
                return action
        return next(a for a in actions if a["type"] == "decline_completion")

    # -- play ----------------------------------------------------------------

    def _play(self, message: Mapping[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any]:
        hand = message["hand"]
        counts = self._counts(hand)
        passes = [a for a in actions if a["type"] == "pass"]

        def after(action):
            return self._without(hand, self._action_cards(action))

        # Rule 0.1: finish with a normal play.
        for action in actions:
            if action["type"] == "play" and not after(action):
                return action
        # Rule 0.2: never bomb out. Rule 0.3: never leave only 2s if avoidable.
        plays = [a for a in actions if a["type"] in {"play", "bomb"} and after(a)]
        plays = [a for a in plays if any(c["rank"] != "2" for c in after(a))] or plays
        if not plays:
            return passes[0] if passes else actions[0]

        ordinary = [a for a in plays if a["type"] == "play"]
        bombs = [a for a in plays if a["type"] == "bomb"]
        whole = [a for a in ordinary if self._width(a) == counts[self._rank(a)]]
        groups, twos = self._gb(hand)
        smallest = self._smallest_opponent(message)
        top = message.get("top")

        if top is None:
            return self._lead(ordinary or plays, whole, counts, groups, twos, smallest)

        width = top["count"]
        # F2: bomb rule.
        if bombs and (not whole or groups - twos <= 1 or (smallest == 1 and width == 1)):
            return bombs[0]
        # F1: lowest clean follow; do not pass.
        if whole:
            return self._lowest(whole)
        if not ordinary:
            return passes[0] if passes else bombs[0]
        # F3: no 2 and only group-breaking follows.
        if width == 1:
            triples = [a for a in ordinary if counts[self._rank(a)] == 3]
            if triples:
                return self._lowest(triples)
        if smallest in (2, 3) or not passes:
            return self._lowest(ordinary)
        return passes[0]

    def _lead(self, plays, whole, counts, groups, twos, smallest):
        # L3: an opponent on one card -> lead a whole pair or larger.
        if smallest == 1:
            big = [a for a in whole if self._width(a) >= 2]
            if big:
                return self._lowest(big)
        singles = [a for a in whole if counts[self._rank(a)] == 1]
        # L2 split: no singles and B >= G -> lead one card of the lowest group.
        if not singles and twos and twos >= groups:
            split = [a for a in plays if self._width(a) == 1]
            if split:
                return self._lowest(split)
        # L1: lowest single.
        if singles:
            return self._lowest(singles)
        # L2: lowest whole group; do not open a 13-card hand with a quad.
        if whole:
            options = whole
            if sum(counts.values()) >= 13:
                options = [a for a in whole if self._width(a) < 4] or whole
            return self._lowest(options)
        return self._lowest(plays)

    def respond(self, message: Mapping[str, Any]) -> dict[str, Any]:
        # Section 5: heads-up play is solved exactly when the position rebuilds.
        if self.heads_up_solver:
            try:
                round_ = ExperiencedAgent3._endgame_round(message)
                if round_ is not None:
                    result = solve_round(round_)
                    if (result.player == message["player"]
                            and result.best_action in message["request"]["legal_actions"]):
                        return result.best_action
            except (AttributeError, IndexError, KeyError, TypeError, ValueError):
                pass
        return super().respond(message)
