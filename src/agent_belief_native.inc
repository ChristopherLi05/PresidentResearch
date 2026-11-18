/* Native Metropolis walk for modeled opponent hands (src/agent_belief.py).
 * Included by agent_search_native.inc, so it ships in the search library.
 * Python extracts the public evidence once per decision; C scores sampled
 * hands against the rank-level HumanRulesAgent3 rules and runs the swaps.
 * Rank indices follow RANKS: 0..11 are 3..A, 12 is the 2. */

enum { B_TWO = 12, B_ACE = 11, B_PASS = -1, B_BOMB = -2 };
enum BeliefKind { BK_PLAY, BK_NOCOMPLETE, BK_TRADE, BK_NONE, BK_SOME, BK_OPENING };
#define BELIEF_HARD 0.02
#define BELIEF_RETURNED 0.3
#define BELIEF_ALL_TWOS 1e-9

/* Encoded action: B_PASS, B_BOMB, or rank * 8 + width for a play. */
static int bplay(int q, int k) { return q * 8 + k; }

static int belief_rules_play(const int *c, int top, int w, int smallest) {
    int n = 0, twos = c[B_TWO], nontwo, groups = 0, q, any_play = 0, bomb, filtered, best;
    for (q = 0; q < 13; ++q) n += c[q];
    for (q = 0; q < B_TWO; ++q) groups += c[q] > 0;
    nontwo = n - twos;
    if (top < 0) {
        for (q = 0; q < B_TWO; ++q) if (c[q] && c[q] == n) return bplay(q, n);
        /* Rule 0.3: keep plays leaving an ordinary card, if any exist. */
        for (q = 0; q < B_TWO; ++q) if (c[q] && nontwo - 1 > 0) any_play = 1;
        filtered = any_play;
#define VALID(q, k) (c[q] >= (k) && (k) >= 1 && (!filtered || nontwo - (k) > 0))
#define WHOLE(q) (c[q] && VALID(q, c[q]))
        any_play = 0;
        for (q = 0; q < B_TWO; ++q) if (VALID(q, 1)) any_play = 1;
        if (!any_play) return B_PASS;
        if (smallest == 1)                                   /* L3 */
            for (q = 0; q < B_TWO; ++q) if (WHOLE(q) && c[q] >= 2) return bplay(q, c[q]);
        {
            int single = -1;
            for (q = 0; q < B_TWO && single < 0; ++q) if (WHOLE(q) && c[q] == 1) single = q;
            if (single < 0 && twos && twos >= groups)        /* L2 split */
                for (q = 0; q < B_TWO; ++q) if (VALID(q, 1)) return bplay(q, 1);
            if (single >= 0) return bplay(single, 1);         /* L1 */
        }
        best = -1;                                           /* L2 */
        for (q = 0; q < B_TWO; ++q)
            if (WHOLE(q) && !(n >= 13 && c[q] == 4)) { best = q; break; }
        if (best < 0) for (q = 0; q < B_TWO; ++q) if (WHOLE(q)) { best = q; break; }
        if (best >= 0) return bplay(best, c[best]);
        for (q = 0; q < B_TWO; ++q) if (VALID(q, 1)) {
            int k = c[q];
            while (k > 1 && !VALID(q, k)) --k;
            return bplay(q, k);
        }
        return B_PASS;
#undef VALID
#undef WHOLE
    }
    if (w == n) for (q = top; q < B_TWO; ++q) if (c[q] == w) return bplay(q, w);
    for (q = top; q < B_TWO; ++q) if (c[q] >= w) any_play = 1;
    bomb = twos > 0 && n > 1;                                /* rule 0.2 */
    {
        int kept = any_play && nontwo - w > 0, kept_bomb = bomb && nontwo > 0;
        if (kept || kept_bomb) { any_play = kept; bomb = kept_bomb; }  /* rule 0.3 */
    }
    if (!any_play && !bomb) return B_PASS;
    {
        int whole = -1;
        if (any_play) for (q = top; q < B_TWO; ++q) if (c[q] == w) { whole = q; break; }
        if (bomb && (whole < 0 || groups - twos <= 1 || (smallest == 1 && w == 1))) return B_BOMB;  /* F2 */
        if (whole >= 0) return bplay(whole, w);              /* F1 */
    }
    if (!any_play) return B_PASS;
    if (w == 1) for (q = top; q < B_TWO; ++q) if (c[q] == 3) return bplay(q, 1);   /* F3 */
    if (smallest == 2 || smallest == 3)
        for (q = top; q < B_TWO; ++q) if (c[q] >= w) return bplay(q, w);
    return B_PASS;
}

static int belief_rules_trade(const int *c, int failed) {
    int offered[13], any = 0, r, q, has_failed_triple = 0;
    int order[40], count = 0;
    for (r = 0; r < 13; ++r) { offered[r] = !(failed >> r & 1) && c[r] < 4; any |= offered[r]; }
    if (!any) for (r = 0; r < 13; ++r) { offered[r] = !(failed >> r & 1); any |= offered[r]; }
    if (!any) for (r = 0; r < 13; ++r) offered[r] = 1;
    for (q = 0; q < B_TWO; ++q) if (c[q] == 3 && (failed >> q & 1)) has_failed_triple = 1;
    if (!has_failed_triple) for (q = B_TWO - 1; q >= 0; --q) if (c[q] == 3) order[count++] = q;
    if (c[B_TWO] >= 2 && c[B_ACE] == 1) order[count++] = B_ACE;
    order[count++] = B_TWO;
    for (q = B_TWO - 1; q >= 0; --q) if (c[q] == 1) order[count++] = q;
    for (q = B_TWO - 1; q >= 0; --q) if (c[q] == 3) order[count++] = q;
    for (q = B_TWO - 1; q >= 0; --q) if (c[q] == 2) order[count++] = q;
    for (q = B_TWO - 1; q >= 0; --q) if (!c[q]) order[count++] = q;
    for (r = 0; r < count; ++r) if (offered[order[r]]) return order[r];
    for (r = 12; r >= 0; --r) if (offered[r]) return r;
    return 0;
}

static int belief_rules_completes(const int *c, int rank, int needed) {
    int rest = 0, groups = 0, twos, q;
    if (c[rank] != needed) return 0;
    for (q = 0; q < 13; ++q) rest += c[q];
    if (rest == needed) return 1;
    twos = c[B_TWO] - (rank == B_TWO ? needed : 0);
    for (q = 0; q < B_TWO; ++q) groups += (q == rank ? c[q] - needed : c[q]) > 0;
    return !(twos > 0 && twos >= groups);
}

static double belief_loglik(const int *counts, const BeliefObservation *obs, int count, double noise) {
    double total = 0, miss = log(noise), hard = log(BELIEF_HARD), returned = log(BELIEF_RETURNED);
    int i, q, c[13], ordinary = 0;
    for (q = 0; q < B_TWO; ++q) ordinary |= counts[q] > 0;
    if (!ordinary) return log(BELIEF_ALL_TWOS);
    for (i = 0; i < count; ++i) {
        const BeliefObservation *o = obs + i;
        const int *d = o->data;
        for (q = 0; q < 13; ++q) { c[q] = counts[q] + o->offset[q]; if (c[q] < 0) c[q] = 0; }
        switch (o->kind) {
        case BK_PLAY: if (belief_rules_play(c, d[0], d[1], d[2]) != d[3]) total += miss; break;
        case BK_NOCOMPLETE: if (belief_rules_completes(c, d[0], d[1])) total += miss; break;
        case BK_TRADE: if (belief_rules_trade(c, d[0]) != d[1]) total += miss; break;
        case BK_NONE: if (c[d[0]]) total += c[d[0]] <= d[1] ? returned : hard; break;
        case BK_SOME: if (!c[d[0]]) total += hard; break;
        case BK_OPENING: if (c[0] != d[0]) total += miss; break;
        default: return NAN;
        }
    }
    return total;
}

/* splitmix64: small, fast, and good enough for proposals. */
static uint64_t belief_next(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static int belief_below(uint64_t *s, int n) { return (int)(belief_next(s) % (uint64_t)n); }
static double belief_unit(uint64_t *s) { return (double)(belief_next(s) >> 11) * (1.0 / 9007199254740992.0); }

static int belief_valid(const BeliefObservation *obs, int count) {
    int i, q;
    for (i = 0; i < count; ++i) {
        if (obs[i].kind < BK_PLAY || obs[i].kind > BK_OPENING) return 0;
        for (q = 0; q < 13; ++q) if (obs[i].offset[q] < -52 || obs[i].offset[q] > 52) return 0;
    }
    return 1;
}

int president_belief_loglik(const int counts[13], const BeliefObservation *obs, int count,
                            double noise, double *out) {
    int q;
    if (!counts || !out || count < 0 || (count && !obs) || !(noise > 0 && noise <= 1)) return -1;
    for (q = 0; q < 13; ++q) if (counts[q] < 0 || counts[q] > 4) return -1;
    if (!belief_valid(obs, count)) return -1;
    *out = belief_loglik(counts, obs, count, noise);
    return 1;
}

int president_belief_walk(int opponents, const int slots[3], int *cards, const int pinned[3][13],
                          const BeliefObservation *obs, const int obs_count[3], double noise,
                          int steps, uint64_t *rng, int *stats) {
    int counts[3][13], start[3], movable[3], moving = 0, offset = 0, total = 0, p, q, i;
    const BeliefObservation *own[3];
    double ll[3];
    if (opponents < 1 || opponents > 3 || !slots || !cards || !pinned || !obs_count || !rng
            || !stats || steps < 0 || !(noise > 0 && noise <= 1)) return -1;
    for (p = 0; p < opponents; ++p) {
        if (slots[p] < 0 || obs_count[p] < 0) return -1;
        start[p] = offset; offset += slots[p];
        own[p] = obs + total; total += obs_count[p];
    }
    if (offset > 52 || (total && !obs) || !belief_valid(obs, total)) return -1;
    for (p = 0; p < opponents; ++p) {
        for (q = 0; q < 13; ++q) {
            if (pinned[p][q] < 0 || pinned[p][q] > 4) return -1;
            counts[p][q] = pinned[p][q];
        }
        for (i = 0; i < slots[p]; ++i) {
            int card = cards[start[p] + i];
            if (card < 0 || card >= 52) return -1;
            ++counts[p][card / 4];
        }
        for (q = 0; q < 13; ++q) if (counts[p][q] > 4) return -1;
        ll[p] = belief_loglik(counts[p], own[p], obs_count[p], noise);
        if (slots[p]) movable[moving++] = p;
    }
    for (; steps > 0; --steps) {
        int a, b, ia, ib, x, y, ca[13], cb[13];
        double la, lb, delta;
        ++stats[0];
        if (moving < 2) continue;
        a = belief_below(rng, moving);
        b = belief_below(rng, moving - 1);
        if (b >= a) ++b;
        a = movable[a]; b = movable[b];
        ia = start[a] + belief_below(rng, slots[a]);
        ib = start[b] + belief_below(rng, slots[b]);
        x = cards[ia] / 4; y = cards[ib] / 4;
        if (x == y) continue;
        memcpy(ca, counts[a], sizeof ca); memcpy(cb, counts[b], sizeof cb);
        --ca[x]; ++ca[y]; --cb[y]; ++cb[x];
        la = belief_loglik(ca, own[a], obs_count[a], noise);
        lb = belief_loglik(cb, own[b], obs_count[b], noise);
        delta = la + lb - ll[a] - ll[b];
        if (delta >= 0 || belief_unit(rng) < exp(delta)) {
            int t = cards[ia]; cards[ia] = cards[ib]; cards[ib] = t;
            memcpy(counts[a], ca, sizeof ca); memcpy(counts[b], cb, sizeof cb);
            ll[a] = la; ll[b] = lb;
            ++stats[1];
        }
    }
    return 1;
}
