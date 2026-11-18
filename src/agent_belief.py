"""Opponent-hand inference for ExperiencedAgent4's sampled worlds.

Uniform sampling deals unseen cards with only public hand sizes. This module
instead weights each deal by how well it explains the public history under
the table-side rules of ``HumanRulesAgent3``
(docs/interpretation/experiencedagent3-human.md):

* draft: a chosen pile's face-up card stays with its drafter;
* trades: failed requests mean the receiver held none of that rank, successful
  ones give the initiator that rank, and the requested rank follows T1/T2/T2b;
  our own returned cards are pinned to their receiver;
* play: every opponent lead, follow, pass and bomb is compared with the
  rank-level rules (0, L1-L3, F1-F3); windows that closed without a completion
  are compared with C1;
* an active player never holds only 2s.

Played cards are public, so an opponent's ranks at any earlier moment are
``current counts + cards they played since``. Every observation is therefore a
cheap test on the sampled current hand. A matching action scores 1, a mismatch
``noise`` (near-certain facts use ``HARD``). A Metropolis chain over card swaps
between opponents samples from that likelihood. It is approximate inference
about a heuristic model of the opponents, not an exact posterior.
"""

from math import exp, log
from time import perf_counter

from src.president import Card, RANK_VALUE, RANKS, SUITS

R = len(RANKS)
TWO = RANKS.index("2")
ACE = RANKS.index("A")
assert TWO == R - 1
ORD = range(TWO)
PASS = ("pass",)
BOMB = ("bomb",)
HARD = 0.02      # mismatch factor for near-certain facts
RETURNED = 0.3   # a failed rank later reaching the receiver as a returned card
ALL_TWOS = 1e-9  # an active hand of only 2s contradicts the rules
# Chain steps before the first world and between worlds, per unpinned card.
# Chosen with belief_diagnostic.py: accuracy flattens beyond these. C steps
# are ~50x cheaper, so the native walk runs longer chains.
NATIVE_STEPS = (40.0, 0.5)
PYTHON_STEPS = (10.0, 0.25)


_idx = RANK_VALUE.__getitem__  # RANK_VALUE[rank] == RANKS.index(rank)


# -- rank-level HumanRulesAgent3 ------------------------------------------

def rules_play(c, top, smallest):
    """HumanRulesAgent3's play choice from rank counts.

    ``c`` holds 13 counts (index TWO is the 2s), ``top`` is ``None`` or
    ``(rank_index, width)``. Returns ``("play", rank, width)``, BOMB or PASS.
    """
    n = sum(c)
    twos = c[TWO]
    nontwo = n - twos
    if top is None:
        for q in ORD:
            if c[q] and c[q] == n:
                return "play", q, n
        plays = [(q, k) for q in ORD if c[q] for k in range(1, c[q] + 1)]
        bomb = False
        w = None
    else:
        r, w = top
        if w == n:
            for q in range(r, TWO):
                if c[q] == w:
                    return "play", q, w
        plays = [(q, w) for q in range(r, TWO) if c[q] >= w]
        bomb = twos > 0
    bomb = bomb and n > 1                       # rule 0.2: never bomb out
    kept = [(q, k) for q, k in plays if nontwo - k > 0]
    kept_bomb = bomb and nontwo > 0             # rule 0.3: never leave only 2s
    if kept or kept_bomb:
        plays, bomb = kept, kept_bomb
    if not plays and not bomb:
        return PASS
    whole = [(q, k) for q, k in plays if k == c[q]]
    groups = sum(1 for q in ORD if c[q])
    if top is None:
        if smallest == 1:                       # L3
            big = [q for q, k in whole if k >= 2]
            if big:
                q = min(big)
                return "play", q, c[q]
        singles = [q for q, k in whole if c[q] == 1]
        if not singles and twos and twos >= groups:   # L2 split
            split = [q for q, k in plays if k == 1]
            if split:
                return "play", min(split), 1
        if singles:                             # L1
            return "play", min(singles), 1
        if whole:                               # L2
            options = whole
            if n >= 13:
                options = [x for x in whole if x[1] < 4] or whole
            q, k = min(options, key=lambda x: (x[0], -x[1]))
            return "play", q, k
        q, k = min(plays, key=lambda x: (x[0], -x[1]))
        return "play", q, k
    if bomb and (not whole or groups - twos <= 1 or (smallest == 1 and w == 1)):   # F2
        return BOMB
    if whole:                                   # F1
        return "play", min(q for q, _ in whole), w
    if not plays:
        return PASS
    if w == 1:                                  # F3 on a single: break a triple
        triples = [q for q, _ in plays if c[q] == 3]
        if triples:
            return "play", min(triples), 1
    if smallest in (2, 3):
        return "play", min(q for q, _ in plays), w
    return PASS


def rules_trade(c, failed):
    """HumanRulesAgent3's requested rank index (T1, T2, T2b)."""
    offered = [r for r in range(R) if r not in failed and c[r] < 4]
    if not offered:
        offered = [r for r in range(R) if r not in failed] or list(range(R))
    offered = set(offered)
    desc = sorted((q for q in ORD if c[q]), reverse=True)
    triples = [q for q in desc if c[q] == 3]
    singles = [q for q in desc if c[q] == 1]
    pairs = [q for q in desc if c[q] == 2]
    priorities = []
    if not failed & set(triples):
        priorities += triples
    if c[TWO] >= 2 and c[ACE] == 1:
        priorities.append(ACE)
    priorities.append(TWO)
    priorities += singles + triples + pairs
    priorities += [q for q in reversed(ORD) if not c[q]]
    for rank in priorities:
        if rank in offered:
            return rank
    return max(offered)


def rules_completes(c, rank, needed):
    """C1: complete unless the remaining hand has B >= G with B > 0."""
    if c[rank] != needed:
        return False
    rest = list(c)
    rest[rank] -= needed
    if not sum(rest):
        return True
    twos = rest[TWO]
    groups = sum(1 for q in ORD if rest[q])
    return not (twos > 0 and twos >= groups)


# -- evidence ----------------------------------------------------------------

def _vec():
    return [0] * R


def _add(vec, cards, sign=1):
    for card in cards:
        vec[_idx(card["rank"])] += sign


def extract(message, player, opponents, returned=()):
    """Return ``(observations, pins)`` from public events and our own trades.

    ``observations[p]`` is a list of ``(offset, kind, data)``. The opponent's
    counts at that moment are ``current + offset``. ``pins`` maps a Card to
    the opponent who must still hold it, if it is unseen.
    """
    events = message.get("events", [])
    players = list(message["players"])
    opponents = set(opponents)
    played_total = {p: _vec() for p in players}
    obs = {p: [] for p in opponents}
    raw = []  # (p, kind, data, played_before snapshot or trade marker)

    # Trades: group requests into exchanges, ended by each success.
    transfers = []   # (exchange, initiator, receiver, rank_index)
    requests = []    # (exchange, initiator, receiver, rank_index, success, failed_before)
    exchange, failed = 0, set()
    for event in events:
        if event.get("type") != "trade_requested":
            continue
        rank = _idx(event["rank"])
        requests.append((exchange, event["initiator"], event["receiver"], rank,
                         bool(event["success"]), frozenset(failed)))
        if event["success"]:
            transfers.append((exchange, event["initiator"], event["receiver"], rank))
            exchange, failed = exchange + 1, set()
        else:
            failed.add(rank)

    def trade_offset(p, first):
        """Counts before exchange ``first`` relative to counts at play start."""
        vec = _vec()
        for k, initiator, receiver, rank in transfers:
            if k >= first:
                if initiator == p:
                    vec[rank] -= 1
                if receiver == p:
                    vec[rank] += 1
        return vec

    gave = {(receiver, rank) for _, _, receiver, rank in transfers}
    initiators = {initiator for _, initiator, _, _ in transfers}

    for k, initiator, receiver, rank, success, failed_before in requests:
        if initiator in opponents:
            raw.append((initiator, "trade", (failed_before, rank), trade_offset(initiator, k)))
        if not success and receiver in opponents:
            # The receiver had none then, but each later return to them could
            # be that rank (T3 returns the initiator's lowest single).
            returns = sum(1 for j, _, r, _ in transfers if j >= k and r == receiver)
            raw.append((receiver, "none", (rank, returns), trade_offset(receiver, k)))
        if success and initiator in opponents:
            raw.append((initiator, "some", rank, trade_offset(initiator, k + 1)))

    # Draft: face-up cards. Pin them unless a trade could have moved them.
    pins = {}
    start = events[0] if events else {}
    face_up = start.get("face_up") or []
    for event in events:
        if event.get("type") != "pile_chosen" or event.get("player") not in opponents:
            continue
        p = event["player"]
        card = face_up[event["pile"]]
        if p in initiators or (p, _idx(card["rank"])) in gave:
            raw.append((p, "some", _idx(card["rank"]), trade_offset(p, 0)))
        else:
            pins[Card.from_json(card)] = p
    for card, receiver in returned:
        if receiver in opponents and (receiver, _idx(card["rank"])) not in gave:
            pins.setdefault(Card.from_json(card), receiver)

    # Play: replay the public board.
    sizes = {p: 13 for p in players}
    active = set(players)
    top = None
    stack = 0
    for index, event in enumerate(events):
        kind = event.get("type")
        p = event.get("player")
        if kind in {"opening_play", "played", "passed", "bombed"} and p in opponents and len(active) > 2:
            smallest = min((sizes[q] for q in active if q != p and sizes[q] > 0), default=13)
            before = played_total[p][:]
            if kind == "opening_play" and start.get("first_round"):
                # First round: the lead must contain a 3; the rules lead them all.
                raw.append((p, "opening", len(event["cards"]), before))
            elif kind in {"played", "opening_play"}:  # later rounds also label the first lead opening_play
                observed = ("play", _idx(event["cards"][0]["rank"]), len(event["cards"]))
                raw.append((p, "play", (top, smallest, observed), before))
            elif kind == "passed" and top is not None:
                raw.append((p, "play", (top, smallest, PASS), before))
            elif kind == "bombed" and top is not None:
                raw.append((p, "play", (top, smallest, BOMB), before))
        if kind in {"opening_play", "played"}:
            rank = _idx(event["cards"][0]["rank"])
            width = len(event["cards"])
            stack = stack + width if top is not None and top[0] == rank else width
            top = (rank, width)
            _add(played_total[p], event["cards"])
            sizes[p] -= width
            if stack < 4:
                _window(events, index, rank, 4 - stack, active, opponents, played_total, raw)
        elif kind == "completed":
            _add(played_total[p], event["cards"])
            sizes[p] -= len(event["cards"])
        elif kind == "bombed":
            _add(played_total[p], [event["card"]])
            sizes[p] -= 1
        elif kind == "board_cleared":
            top, stack = None, 0
        elif kind in {"player_done", "twos_announced"}:
            active.discard(p)
            sizes[p] = 0

    for p, kind, data, before in raw:
        obs[p].append((_finalize_offset(kind, played_total[p], before), kind, data))
    return obs, pins


def _window(events, index, rank, needed, active, opponents, played_total, raw):
    """A completion window after ``events[index]`` that nobody took."""
    for later in events[index + 1:]:
        kind = later.get("type")
        if kind == "completed":
            return
        if kind in {"played", "bombed", "board_cleared", "round_finished"}:
            break
    for p in active:
        if p in opponents:
            raw.append((p, "nocomplete", (rank, needed), played_total[p][:]))


# Play-phase observations store cards played before; trade observations store
# a trade offset. extract() distinguishes them by kind.
_PLAY_KINDS = {"play", "opening", "nocomplete"}


def _finalize_offset(kind, total, before):
    if kind in _PLAY_KINDS:
        return tuple(total[i] - before[i] for i in range(R))
    return tuple(total[i] + before[i] for i in range(R))


# -- likelihood ----------------------------------------------------------------

def log_likelihood(counts, observations, noise):
    """Log weight of one opponent's current rank counts."""
    if not any(counts[q] for q in ORD):
        return log(ALL_TWOS)
    miss, hard = log(noise), log(HARD)
    total = 0.0
    for offset, kind, data in observations:
        c = [max(0, a + b) for a, b in zip(counts, offset)]
        if kind == "play":
            top, smallest, observed = data
            if rules_play(c, top, smallest) != observed:
                total += miss
        elif kind == "nocomplete":
            if rules_completes(c, *data):
                total += miss
        elif kind == "trade":
            failed, observed = data
            if rules_trade(c, failed) != observed:
                total += miss
        elif kind == "none":
            rank, returns = data
            if c[rank]:
                total += log(RETURNED) if c[rank] <= returns else hard
        elif kind == "some":
            if not c[data]:
                total += hard
        elif kind == "opening":
            if c[0] != data:
                total += miss
    return total


class BeliefSampler:
    """Draw opponent deals for one decision from a single Metropolis chain."""

    def __init__(self, prepared, player, message, rng, *, noise=0.2, returned=(),
                 burn_in_per_card=None, thin_per_card=None, native=None):
        """``native``: the loaded search library, to run the walk in C."""
        start = perf_counter()
        self.state, self.unseen, self.sizes = prepared
        self.player, self.rng, self.noise = player, rng, noise
        self.opponents = [p for p in self.state.players if p != player and p in self.state.active]
        self.observations, pins = extract(message, player, self.opponents, returned)
        unseen = set(self.unseen)
        self.pinned = {p: [] for p in self.opponents}
        for card, p in pins.items():
            if card in unseen:
                self.pinned[p].append(card)
        for p in self.opponents:   # Inconsistent pins are dropped, not trusted.
            if len(self.pinned[p]) > self.sizes[p]:
                self.pinned[p] = []
        taken = {c for cards in self.pinned.values() for c in cards}
        free = [c for c in self.unseen if c not in taken]
        rng.shuffle(free)
        self.free, offset = {}, 0
        for p in self.opponents:
            n = self.sizes[p] - len(self.pinned[p])
            self.free[p] = free[offset:offset + n]
            offset += n
        n_free = sum(len(v) for v in self.free.values())
        default_burn, default_thin = NATIVE_STEPS if native is not None else PYTHON_STEPS
        burn_in_per_card = default_burn if burn_in_per_card is None else burn_in_per_card
        thin_per_card = default_thin if thin_per_card is None else thin_per_card
        self.burn_in = round(burn_in_per_card * n_free)
        self.thin = max(1, round(thin_per_card * n_free))
        self.steps = self.accepted = 0
        self.burned = False
        self.native = None
        if native is not None:
            self._native_setup(native)
        else:
            self.counts = {p: self._count(self.pinned[p] + self.free[p]) for p in self.opponents}
            self.cache = {}
            self.ll = {p: self._ll(p, self.counts[p]) for p in self.opponents}
            self.movable = [p for p in self.opponents if self.free[p]]
        self.elapsed = perf_counter() - start

    def _native_setup(self, lib):
        import ctypes as ct
        from src.agent_search_c import BeliefObservation
        opponents = self.opponents
        cards = [c for p in opponents for c in self.free[p]]
        self.code_card = {_code(c): c for c in cards}
        obs = [o for p in opponents for o in self.observations[p]]
        array = (BeliefObservation * max(1, len(obs)))()
        for item, (offset, kind, data) in zip(array, obs):
            item.kind = NATIVE_KIND[kind]
            item.offset[:] = offset
            item.data[:] = native_data(kind, data)
        pinned = ((ct.c_int * 13) * 3)()
        for i, p in enumerate(opponents):
            pinned[i][:] = self._count(self.pinned[p])
        self.native = {
            "lib": lib, "n": len(opponents), "obs": array, "pinned": pinned,
            "slots": (ct.c_int * 3)(*[len(self.free[p]) for p in opponents]),
            "cards": (ct.c_int * max(1, len(cards)))(*[_code(c) for c in cards]),
            "counts": (ct.c_int * 3)(*[len(self.observations[p]) for p in opponents]),
            "rng": ct.c_uint64(self.rng.getrandbits(64)), "stats": (ct.c_int * 2)(),
        }

    def _native_walk(self, steps):
        import ctypes as ct
        n = self.native
        status = n["lib"].president_belief_walk(
            n["n"], n["slots"], n["cards"], n["pinned"], n["obs"], n["counts"], self.noise,
            steps, ct.byref(n["rng"]), n["stats"])
        if status != 1:
            raise ValueError("native belief walk rejected its inputs")
        self.steps, self.accepted = n["stats"][0], n["stats"][1]
        offset = 0
        for p in self.opponents:
            k = len(self.free[p])
            self.free[p] = [self.code_card[c] for c in n["cards"][offset:offset + k]]
            offset += k

    @staticmethod
    def _count(cards):
        counts = [0] * R
        for card in cards:
            counts[_idx(card.rank)] += 1
        return counts

    def _ll(self, p, counts):
        key = (p, tuple(counts))
        value = self.cache.get(key)
        if value is None:
            value = self.cache[key] = log_likelihood(counts, self.observations[p], self.noise)
        return value

    def _step(self):
        self.steps += 1
        if len(self.movable) < 2:
            return
        a, b = self.rng.sample(self.movable, 2)
        i = self.rng.randrange(len(self.free[a]))
        j = self.rng.randrange(len(self.free[b]))
        x, y = self.free[a][i], self.free[b][j]
        if x.rank == y.rank:
            return
        ca, cb = self.counts[a][:], self.counts[b][:]
        ca[_idx(x.rank)] -= 1; ca[_idx(y.rank)] += 1
        cb[_idx(y.rank)] -= 1; cb[_idx(x.rank)] += 1
        la, lb = self._ll(a, ca), self._ll(b, cb)
        delta = la + lb - self.ll[a] - self.ll[b]
        if delta >= 0 or self.rng.random() < exp(delta):
            self.free[a][i], self.free[b][j] = y, x
            self.counts[a], self.counts[b] = ca, cb
            self.ll[a], self.ll[b] = la, lb
            self.accepted += 1

    def sample(self):
        """Advance the chain and return a world in sample_world's format."""
        from src.agent_search import clone
        start = perf_counter()
        steps = self.thin if self.burned else self.burn_in
        if self.native is not None:
            self._native_walk(steps)
        else:
            for _ in range(steps):
                self._step()
        self.burned = True
        world = clone(self.state)
        for p in self.opponents:
            world.hands[p] = self.pinned[p] + self.free[p][:]
        self.elapsed += perf_counter() - start
        return world

    def log_weight(self):
        return sum(log_likelihood(self._count(self.pinned[p] + self.free[p]), self.observations[p], self.noise)
                   for p in self.opponents)


# -- native encoding -------------------------------------------------------------

NATIVE_KIND = {"play": 0, "nocomplete": 1, "trade": 2, "none": 3, "some": 4, "opening": 5}


def _code(card):
    return _idx(card.rank) * 4 + SUITS.index(card.suit)


def native_action(key):
    return -1 if key == PASS else -2 if key == BOMB else key[1] * 8 + key[2]


def native_data(kind, data):
    """The four ints of a C BeliefObservation for one Python observation."""
    if kind == "play":
        top, smallest, observed = data
        return [-1 if top is None else top[0], 0 if top is None else top[1], smallest, native_action(observed)]
    if kind == "trade":
        failed, observed = data
        return [sum(1 << r for r in failed), observed, 0, 0]
    if kind in {"nocomplete", "none"}:
        return [data[0], data[1], 0, 0]
    return [data, 0, 0, 0]
