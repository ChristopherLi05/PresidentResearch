/* Exact two-player minimax; move order and state semantics match solver.py.
 * Every solve owns its hash table. No globals, heuristic cutoff, or callbacks. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int hands[2][13], actor, rank, count, stack, window[2], window_count;
    int clear_after, next_actor, pass_clears, pass_actor, winner;
} State;
enum Kind { PLAY, BOMB, PASS, COMPLETE, DECLINE };
typedef struct { int kind, rank, count; } Move;
typedef struct { State state; uint64_t hash; int used, winner; } Entry;
typedef struct { Entry *entries; size_t capacity, size; uint64_t misses, hits; int failed; } Table;
typedef struct { int kind, rank, count, winner; uint64_t states, hits; } Result;

static uint64_t hash_state(const State *s) {
    const unsigned char *bytes = (const unsigned char *)s;
    uint64_t h = UINT64_C(14695981039346656037);
    size_t i;
    for (i = 0; i < sizeof *s; ++i) { h ^= bytes[i]; h *= UINT64_C(1099511628211); }
    return h;
}
static size_t slot(const Table *t, const State *s, uint64_t hash) {
    size_t i = (size_t)hash & (t->capacity-1);
    while (t->entries[i].used && (t->entries[i].hash != hash || memcmp(&t->entries[i].state,s,sizeof *s)))
        i = (i+1) & (t->capacity-1);
    return i;
}
static int grow(Table *t) {
    Entry *old = t->entries, *next;
    size_t old_capacity = t->capacity, capacity = old_capacity ? old_capacity*2 : 1024, i;
    if (capacity < old_capacity || capacity > SIZE_MAX/sizeof *next) return 0;
    next = calloc(capacity,sizeof *next);
    if (!next) return 0;
    t->entries = next; t->capacity = capacity;
    for (i = 0; i < old_capacity; ++i) if (old[i].used)
        next[slot(t,&old[i].state,old[i].hash)] = old[i];
    free(old); return 1;
}
static int moves(const State *s, Move out[50]) {
    int rank, n = 0, count;
    const int *hand = s->hands[s->actor];
    if (s->window_count) {
        int needed = 4-s->stack;
        if (hand[s->rank] >= needed) out[n++] = (Move){COMPLETE,s->rank,needed};
        out[n++] = (Move){DECLINE,-1,0}; return n;
    }
    for (rank = 0; rank < 12; ++rank) {
        if (s->rank < 0) {
            for (count = hand[rank]; count > 0; --count) out[n++] = (Move){PLAY,rank,count};
        } else if (rank >= s->rank && hand[rank] >= s->count)
            out[n++] = (Move){PLAY,rank,s->count};
    }
    if (s->rank >= 0) {
        if (hand[12]) out[n++] = (Move){BOMB,12,1};
        out[n++] = (Move){PASS,-1,0};
    }
    return n;
}
static State cleared(const State *s, int actor) {
    State result;
    memset(&result,0,sizeof result);
    memcpy(result.hands,s->hands,sizeof result.hands);
    result.actor = actor; result.rank = -1; result.pass_clears = 1; result.winner = -1;
    return result;
}
static State step(const State *s, Move m) {
    State child = *s;
    int actor = s->actor, rank, ordinary = 0, total = 0, equal, stack;
    if (m.kind == DECLINE) {
        --child.window_count;
        child.window[0] = child.window_count ? s->window[1] : 0; child.window[1] = 0;
        if (child.window_count) { child.actor = child.window[0]; return child; }
        if (s->clear_after) return cleared(s,s->next_actor);
        child.actor = s->next_actor; child.clear_after = 0; child.next_actor = 0; return child;
    }
    if (m.kind == PASS) {
        child.window[0] = actor; child.window[1] = 1-actor; child.window_count = 2;
        child.clear_after = s->pass_clears; child.next_actor = s->pass_actor;
        child.pass_clears = 1; child.pass_actor = s->pass_clears ? s->pass_actor : actor;
        return child;
    }
    child.hands[actor][m.rank] -= m.count;
    for (rank = 0; rank < 13; ++rank) {
        total += child.hands[actor][rank];
        if (rank < 12) ordinary += child.hands[actor][rank];
    }
    if (!total) { child.winner = m.kind == BOMB ? 1-actor : actor; return child; }
    if (!ordinary) { child.winner = 1-actor; return child; }
    if (m.kind == BOMB || m.kind == COMPLETE) return cleared(&child,actor);
    equal = m.rank == s->rank; stack = equal ? s->stack+m.count : m.count;
    if (stack == 4) return cleared(&child,actor);
    child = cleared(&child,actor);
    child.rank = m.rank; child.count = m.count; child.stack = stack;
    child.window[0] = actor; child.window[1] = 1-actor; child.window_count = 2;
    child.clear_after = equal; child.next_actor = equal ? actor : 1-actor; child.pass_actor = actor;
    return child;
}
static int winner(Table *t, const State *s) {
    uint64_t hash = hash_state(s);
    size_t index = slot(t,s,hash);
    int result, count, i;
    Move options[50];
    if (t->entries[index].used) { ++t->hits; return t->entries[index].winner; }
    ++t->misses;
    result = s->winner;
    if (result < 0) {
        result = 1-s->actor; count = moves(s,options);
        for (i = 0; i < count; ++i) {
            State child = step(s,options[i]);
            int child_winner = winner(t,&child);
            if (t->failed) return -1;
            if (child_winner == s->actor) { result = s->actor; break; }
        }
    }
    if (t->size >= t->capacity-t->capacity/3 && !grow(t)) { t->failed = 1; return -1; }
    /* Recursion may have resized the table. Never retain an entry pointer. */
    index = slot(t,s,hash);
    t->entries[index].state = *s; t->entries[index].hash = hash;
    t->entries[index].used = 1; t->entries[index].winner = result; ++t->size;
    return result;
}
int president_solver_version(void) { return 1; }
/* Return 1 success, -1 invalid state, -2 allocation failure. */
int president_solve(const State *input, Result *out) {
    State root;
    Table table = {0};
    Move options[50], best;
    int p, rank, i, count, outcome;
    if (!input || !out) return -1;
    memset(&root,0,sizeof root);
    memcpy(&root,input,sizeof root);
    if (root.actor < 0 || root.actor > 1 || root.rank < -1 || root.rank > 11 ||
        root.count < 0 || root.count > 4 || root.stack < 0 || root.stack > 4 ||
        root.window_count < 0 || root.window_count > 2 || root.winner != -1 ||
        root.next_actor < 0 || root.next_actor > 1 || root.pass_actor < 0 || root.pass_actor > 1 ||
        (root.clear_after != 0 && root.clear_after != 1) || (root.pass_clears != 0 && root.pass_clears != 1)) return -1;
    if ((root.rank < 0 && (root.count || root.stack || root.window_count)) ||
        (root.rank >= 0 && (root.count < 1 || root.stack < root.count))) return -1;
    if (root.window_count && (root.stack >= 4 || root.window[0] != root.actor)) return -1;
    for (i = 0; i < root.window_count; ++i) if (root.window[i] < 0 || root.window[i] > 1) return -1;
    if (root.window_count == 2 && root.window[0] == root.window[1]) return -1;
    for (i = root.window_count; i < 2; ++i) root.window[i] = 0;
    for (p = 0; p < 2; ++p) {
        int ordinary = 0;
        for (rank = 0; rank < 13; ++rank) {
            if (root.hands[p][rank] < 0 || root.hands[p][rank] > 4) return -1;
            if (rank < 12) ordinary += root.hands[p][rank];
        }
        if (!ordinary) return -1;
    }
    count = moves(&root,options);
    if (!count) return -1;
    if (!grow(&table)) return -2;
    best = options[0]; outcome = 1-root.actor;
    for (i = 0; i < count; ++i) {
        State child = step(&root,options[i]);
        int result = winner(&table,&child);
        if (table.failed) { free(table.entries); return -2; }
        if (result == root.actor) { best = options[i]; outcome = root.actor; break; }
    }
    out->kind = best.kind; out->rank = best.rank; out->count = best.count; out->winner = outcome;
    out->states = table.misses; out->hits = table.hits;
    free(table.entries); return 1;
}
