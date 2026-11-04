/* C11 port of president.py. No external dependencies. */
#include "president.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The search build adds an optional native evaluator in this translation unit
 * so it can reuse the engine's private state without exposing it as an ABI. */
#ifdef PRESIDENT_SEARCH
#include <math.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif
#endif

static const char *const ranks[] = {
    "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2"
};
static const char *const suits[] = {"C", "D", "H", "S"};
static const char *const roles[] = {"President", "Vice President", "Vice Scum", "Scum"};
static const char *const actions[] = {
    "choose_pile", "trade_request", "trade_return", "play", "pass", "bomb",
    "complete", "decline_completion"
};
enum Phase { DRAFT, TRADE_REQUEST, TRADE_RETURN, PLAY, COMPLETION, FINISHED };
static const char *const phases[] = {
    "draft", "trade_request", "trade_return", "play", "completion", "finished"
};
enum EventType {
    ROUND_STARTED, PILE_CHOSEN, DRAFT_COMPLETE, TRADE_REQUESTED,
    TRADE_COMPLETED, TRADES_COMPLETE, OPENING_PLAY, PLAYED, PASSED,
    BOMBED, COMPLETED, BOARD_CLEARED, PLAYER_DONE, TWOS_ANNOUNCED, ROUND_FINISHED
};
static const char *const event_names[] = {
    "round_started", "pile_chosen", "draft_complete", "trade_requested",
    "trade_completed", "trades_complete", "opening_play", "played", "passed",
    "bombed", "completed", "board_cleared", "player_done", "twos_announced",
    "round_finished"
};
typedef struct {
    enum EventType type;
    int player, receiver, value, flag, count, cards[4], list[4];
} Event;
typedef struct { int player; PresidentAction action; } History;
typedef struct { int cards[52], count; } Hand;
struct PresidentRound {
    char *players[4];
    int first_round, roles[4], active[4], rankings[4];
    Hand hands[4], piles[4];
    int finish_order[4], finish_count, bomb_finishers[4], bomb_count;
    int top_rank, top_count, top_stack_count, last_player, turn, opened;
    enum Phase phase;
    int draft_order[4], draft_index, trade_index;
    int completion_order[4], completion_count, completion_cursor;
    int pending_player, pending_skip;
    Event *events;
    size_t event_count, event_capacity;
    History *history;
    size_t history_count, history_capacity;
};
struct PresidentGame {
    char *players[4];
    uint64_t rng;
    size_t round_number;
    PresidentRound *round;
};

const char *president_status_string(PresidentStatus status) {
    static const char *const messages[] = {
        "ok", "invalid argument", "illegal action", "out of memory",
        "round has not finished", "action limit reached", "client error"
    };
    return (unsigned)status < sizeof messages / sizeof *messages ? messages[status] : "unknown status";
}
const char *president_rank_name(int rank) { return rank >= 0 && rank < 13 ? ranks[rank] : NULL; }
const char *president_suit_name(int suit) { return suit >= 0 && suit < 4 ? suits[suit] : NULL; }
const char *president_role_name(int placement) { return placement >= 1 && placement <= 4 ? roles[placement - 1] : NULL; }
PresidentCard president_card(const char *rank, const char *suit) {
    int i, j;
    if (!rank || !suit) return -1;
    for (i = 0; i < 13; ++i) for (j = 0; j < 4; ++j)
        if (!strcmp(rank, ranks[i]) && !strcmp(suit, suits[j])) return i * 4 + j;
    return -1;
}
void president_standard_deck(PresidentCard deck[52]) {
    int i;
    if (deck) for (i = 0; i < 52; ++i) deck[i] = i;
}
void president_free(void *memory) { free(memory); }
static int valid_player(int p) { return p >= 0 && p < 4; }
static int valid_card(int c) { return c >= 0 && c < 52; }
static uint64_t random_next(uint64_t *state) {
    uint64_t x = *state ? *state : UINT64_C(0x9e3779b97f4a7c15);
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *state = x;
    return x * UINT64_C(2685821657736338717);
}
static void shuffle(int deck[52], uint64_t *state) {
    int i;
    for (i = 51; i > 0; --i) {
        uint64_t bound = (uint64_t)i + 1, threshold = (UINT64_C(0) - bound) % bound, x;
        int j, t;
        do { x = random_next(state); } while (x < threshold);
        j = (int)(x % bound); t = deck[i]; deck[i] = deck[j]; deck[j] = t;
    }
}
static PresidentStatus copy_players(char *out[4], const char *const in[4]) {
    int i, j;
    if (!in) return PRESIDENT_INVALID_ARGUMENT;
    for (i = 0; i < 4; ++i) {
        if (!in[i]) return PRESIDENT_INVALID_ARGUMENT;
        for (j = 0; j < i; ++j) if (!strcmp(in[i], in[j])) return PRESIDENT_INVALID_ARGUMENT;
    }
    for (i = 0; i < 4; ++i) {
        size_t n = strlen(in[i]) + 1;
        out[i] = malloc(n);
        if (!out[i]) return PRESIDENT_OUT_OF_MEMORY;
        memcpy(out[i], in[i], n);
    }
    return PRESIDENT_OK;
}
static void *reserve(void *ptr, size_t *capacity, size_t needed, size_t item_size) {
    size_t n;
    void *next;
    if (needed <= *capacity) return ptr;
    if (needed > SIZE_MAX / item_size) return NULL;
    n = *capacity ? *capacity : 32;
    while (n < needed) {
        if (n > SIZE_MAX / 2) { n = needed; break; }
        n *= 2;
    }
    if (n > SIZE_MAX / item_size) n = needed;
    next = realloc(ptr, n * item_size);
    if (!next) return NULL;
    *capacity = n;
    return next;
}
/* Capacity is reserved before any state transition. At most eight events can
 * arise from one action; sixteen slots leave room for all terminal paths. */
static Event *event(PresidentRound *r, enum EventType type, int player) {
    Event *e = &r->events[r->event_count++];
    memset(e, 0, sizeof *e); e->type = type; e->player = player;
    return e;
}
static int by_role(const PresidentRound *r, int role) {
    int p;
    for (p = 0; p < 4; ++p) if (r->roles[p] == role) return p;
    return -1;
}
static int trade_initiator(const PresidentRound *r) { return by_role(r, r->trade_index < 2 ? 1 : 2); }
static int trade_receiver(const PresidentRound *r) { return by_role(r, r->trade_index < 2 ? 4 : 3); }
PresidentStatus president_round_create(PresidentRound **out,
    const char *const players[4], int first_round, const int previous_roles[4],
    const PresidentCard *deck, size_t deck_count, uint64_t seed) {
    PresidentRound *r;
    PresidentStatus status;
    int cards[52], seen[52] = {0}, i;
    Event *e;
    if (!out) return PRESIDENT_INVALID_ARGUMENT;
    *out = NULL;
    if (!!first_round == (previous_roles != NULL)) return PRESIDENT_INVALID_ARGUMENT;
    if (previous_roles) {
        int mask = 0;
        for (i = 0; i < 4; ++i) {
            if (previous_roles[i] < 1 || previous_roles[i] > 4) return PRESIDENT_INVALID_ARGUMENT;
            mask |= 1 << previous_roles[i];
        }
        if (mask != 30) return PRESIDENT_INVALID_ARGUMENT;
    }
    if (deck) {
        if (deck_count != 52) return PRESIDENT_INVALID_ARGUMENT;
        for (i = 0; i < 52; ++i) {
            if (!valid_card(deck[i]) || seen[deck[i]]++) return PRESIDENT_INVALID_ARGUMENT;
            cards[i] = deck[i];
        }
    } else {
        if (deck_count) return PRESIDENT_INVALID_ARGUMENT;
        president_standard_deck(cards); shuffle(cards, &seed);
    }
    r = calloc(1, sizeof *r);
    if (!r) return PRESIDENT_OUT_OF_MEMORY;
    status = copy_players(r->players, players);
    if (status != PRESIDENT_OK) { president_round_destroy(r); return status; }
    r->events = reserve(NULL, &r->event_capacity, 1, sizeof *r->events);
    if (!r->events) {
        president_round_destroy(r); return PRESIDENT_OUT_OF_MEMORY;
    }
    r->first_round = !!first_round;
    r->top_rank = r->last_player = r->turn = r->pending_player = -1;
    r->phase = first_round ? PLAY : DRAFT;
    for (i = 0; i < 4; ++i) { r->active[i] = 1; r->roles[i] = previous_roles ? previous_roles[i] : 0; }
    e = event(r, ROUND_STARTED, -1); e->flag = !!first_round;
    if (first_round) {
        for (i = 0; i < 52; ++i) {
            Hand *h = &r->hands[i % 4]; h->cards[h->count++] = cards[i];
            if (cards[i] == 0) r->turn = i % 4;
        }
        e->player = r->turn;
    } else {
        for (i = 0; i < 4; ++i) {
            memcpy(r->piles[i].cards, cards + i * 13, 13 * sizeof(int));
            r->piles[i].count = 13;
            e->cards[i] = cards[i * 13 + 12];
            e->list[i] = r->draft_order[i] = by_role(r, i + 1);
        }
    }
    *out = r; return PRESIDENT_OK;
}
void president_round_destroy(PresidentRound *r) {
    int i;
    if (!r) return;
    for (i = 0; i < 4; ++i) free(r->players[i]);
    free(r->events); free(r->history); free(r);
}
int president_round_done(const PresidentRound *r) { return r && r->phase == FINISHED; }
int president_round_actor(const PresidentRound *r) {
    if (!r) return -1;
    switch (r->phase) {
    case DRAFT: return r->draft_order[r->draft_index];
    case TRADE_REQUEST: case TRADE_RETURN: return trade_initiator(r);
    case COMPLETION: return r->completion_order[r->completion_cursor];
    default: return r->turn;
    }
}
static void offer(PresidentAction *out, size_t capacity, size_t *n, PresidentAction a) {
    if (out && *n < capacity) out[*n] = a;
    ++*n;
}
static void combinations(const int *cards, int length, int needed, int offset,
    PresidentAction *a, PresidentAction *out, size_t capacity, size_t *n) {
    int i;
    if (a->count == needed) { offer(out, capacity, n, *a); return; }
    for (i = offset; i <= length - (needed - a->count); ++i) {
        a->cards[a->count++] = cards[i];
        combinations(cards, length, needed, i + 1, a, out, capacity, n);
        --a->count;
    }
}
static int compare_card(const void *a, const void *b) { return *(const int *)a - *(const int *)b; }
static Hand sorted_hand(const PresidentRound *r, int p) {
    Hand h = r->hands[p]; qsort(h.cards, (size_t)h.count, sizeof(int), compare_card); return h;
}
size_t president_round_legal_actions(const PresidentRound *r, int p, PresidentAction *out, size_t capacity) {
    PresidentAction a = {0};
    size_t n = 0;
    int i, rank;
    if (!r || !valid_player(p) || president_round_done(r) || president_round_actor(r) != p) return 0;
    if (r->phase == DRAFT) {
        a.type = PRESIDENT_CHOOSE_PILE;
        for (i = 0; i < 4; ++i) if (r->piles[i].count) { a.pile = i; offer(out, capacity, &n, a); }
    } else if (r->phase == TRADE_REQUEST) {
        a.type = PRESIDENT_TRADE_REQUEST;
        for (i = 0; i < 13; ++i) { a.rank = i; offer(out, capacity, &n, a); }
    } else if (r->phase == TRADE_RETURN) {
        Hand h = sorted_hand(r, p); a.type = PRESIDENT_TRADE_RETURN;
        for (i = 0; i < h.count; ++i) { a.cards[0] = h.cards[i]; offer(out, capacity, &n, a); }
    } else if (r->phase == COMPLETION) {
        int cards[4], count = 0, needed = 4 - r->top_stack_count;
        for (i = 0; i < r->hands[p].count; ++i)
            if (r->hands[p].cards[i] / 4 == r->top_rank) cards[count++] = r->hands[p].cards[i];
        a.type = PRESIDENT_COMPLETE;
        if (needed > 0 && count >= needed) combinations(cards, count, needed, 0, &a, out, capacity, &n);
        a.type = PRESIDENT_DECLINE_COMPLETION; offer(out, capacity, &n, a);
    } else {
        if (r->top_rank >= 0) {
            a.type = PRESIDENT_PASS; offer(out, capacity, &n, a);
            a.type = PRESIDENT_BOMB;
            for (i = 0; i < r->hands[p].count; ++i) if (r->hands[p].cards[i] / 4 == 12) {
                a.cards[0] = r->hands[p].cards[i]; offer(out, capacity, &n, a);
            }
        }
        for (rank = 0; rank < 12; ++rank) {
            int cards[4], count = 0, k;
            if (r->top_rank < 0 && r->first_round && !r->opened && rank != 0) continue;
            if (r->top_rank >= 0 && rank < r->top_rank) continue;
            for (i = 0; i < r->hands[p].count; ++i)
                if (r->hands[p].cards[i] / 4 == rank) cards[count++] = r->hands[p].cards[i];
            a.type = PRESIDENT_PLAY; a.count = 0;
            if (r->top_rank < 0) {
                for (k = 1; k <= count; ++k) combinations(cards, count, k, 0, &a, out, capacity, &n);
            } else if (count >= r->top_count)
                combinations(cards, count, r->top_count, 0, &a, out, capacity, &n);
        }
    }
    return n;
}
static int remove_cards(Hand *h, const int *cards, int count) {
    Hand next = *h;
    int i, j;
    for (i = 0; i < count; ++i) {
        for (j = 0; j < next.count && next.cards[j] != cards[i]; ++j) {}
        if (j == next.count) return 0;
        --next.count;
        memmove(next.cards + j, next.cards + j + 1, (size_t)(next.count - j) * sizeof(int));
    }
    *h = next; return 1;
}
static int next_active(const PresidentRound *r, int p, int skip) {
    int i;
    for (i = 0; i < 4; ++i) {
        p = (p + 1) % 4;
        if (r->active[p]) { if (skip) { skip = 0; continue; } return p; }
    }
    return -1;
}
static void clear_to(PresidentRound *r, int p) {
    r->top_rank = r->last_player = -1; r->top_count = r->top_stack_count = 0;
    r->phase = PLAY; r->completion_count = r->completion_cursor = 0; r->pending_player = -1;
    r->turn = r->active[p] ? p : next_active(r, p, 0);
    event(r, BOARD_CLEARED, r->turn);
}
static void advance(PresidentRound *r, int p, int skip) {
    int i, candidate = p, remaining_skip = skip;
    if (r->last_player >= 0) for (i = 0; i < 4; ++i) {
        candidate = (candidate + 1) % 4;
        if (candidate == r->last_player) { clear_to(r, r->last_player); return; }
        if (r->active[candidate]) { if (remaining_skip) { remaining_skip = 0; continue; } break; }
    }
    r->turn = next_active(r, p, skip);
}
static void check_end(PresidentRound *r) {
    int p, count = 0, last = -1, position = 1;
    Event *e;
    if (r->phase == FINISHED) return;
    for (p = 0; p < 4; ++p) if (r->active[p]) { ++count; last = p; }
    if (count > 1) return;
    if (last >= 0) {
        r->active[last] = 0; r->finish_order[r->finish_count++] = last;
        event(r, PLAYER_DONE, last)->flag = 2; /* last_active, no bomb_finish field */
    }
    for (p = 0; p < r->finish_count; ++p) r->rankings[r->finish_order[p]] = position++;
    for (p = 0; p < r->bomb_count; ++p) r->rankings[r->bomb_finishers[p]] = position++;
    r->phase = FINISHED; r->turn = -1;
    e = event(r, ROUND_FINISHED, -1); memcpy(e->list, r->rankings, sizeof e->list);
}
static void finish_if_empty(PresidentRound *r, int p, int bomb) {
    if (r->hands[p].count) return;
    r->active[p] = 0;
    if (bomb) r->bomb_finishers[r->bomb_count++] = p;
    else r->finish_order[r->finish_count++] = p;
    event(r, PLAYER_DONE, p)->flag = bomb;
    check_end(r);
}
static void check_twos(PresidentRound *r) {
    int p, i;
    if (r->phase == FINISHED) return;
    for (p = 0; p < 4; ++p) if (r->active[p] && r->hands[p].count) {
        for (i = 0; i < r->hands[p].count && r->hands[p].cards[i] / 4 == 12; ++i) {}
        if (i == r->hands[p].count) {
            r->active[p] = 0; r->bomb_finishers[r->bomb_count++] = p; event(r, TWOS_ANNOUNCED, p);
        }
    }
    check_end(r);
}
static void offer_completions(PresidentRound *r, int p, int skip) {
    int i, candidate;
    r->completion_count = r->completion_cursor = 0;
    for (i = 0; i < 4; ++i) {
        candidate = (p + i) % 4;
        if (r->active[candidate]) r->completion_order[r->completion_count++] = candidate;
    }
    r->pending_player = p; r->pending_skip = skip; r->phase = COMPLETION;
}
/* Validate everything before reserving memory or modifying state. */
static int legal(const PresidentRound *r, int p, const PresidentAction *a) {
    int i, count;
    Hand h;
    if (!a || !valid_player(p) || r->phase == FINISHED || president_round_actor(r) != p) return 0;
    if (r->phase == DRAFT) return a->type == PRESIDENT_CHOOSE_PILE && valid_player(a->pile) && r->piles[a->pile].count;
    if (r->phase == TRADE_REQUEST) return a->type == PRESIDENT_TRADE_REQUEST && a->rank >= 0 && a->rank < 13;
    if (r->phase == TRADE_RETURN) {
        if (a->type != PRESIDENT_TRADE_RETURN) return 0;
    } else if (r->phase == COMPLETION) {
        if (a->type == PRESIDENT_DECLINE_COMPLETION) return 1;
        if (a->type != PRESIDENT_COMPLETE || a->count != 4 - r->top_stack_count) return 0;
    } else {
        if (a->type == PRESIDENT_PASS) return r->top_rank >= 0;
        if (a->type == PRESIDENT_BOMB) {
            if (r->top_rank < 0 || (r->first_round && !r->opened) || !valid_card(a->cards[0]) || a->cards[0] / 4 != 12) return 0;
        } else if (a->type != PRESIDENT_PLAY) return 0;
    }
    count = (a->type == PRESIDENT_TRADE_RETURN || a->type == PRESIDENT_BOMB) ? 1 : a->count;
    if (count < 1 || count > 4) return 0;
    for (i = 0; i < count; ++i) if (!valid_card(a->cards[i])) return 0;
    if (a->type == PRESIDENT_PLAY || a->type == PRESIDENT_COMPLETE) {
        int rank = a->cards[0] / 4;
        for (i = 1; i < count; ++i) if (a->cards[i] / 4 != rank) return 0;
        if (a->type == PRESIDENT_COMPLETE) { if (rank != r->top_rank) return 0; }
        else {
            if (rank == 12) return 0;
            if (r->top_rank < 0) { if (r->first_round && !r->opened && rank != 0) return 0; }
            else if (count != r->top_count || rank < r->top_rank) return 0;
        }
    }
    h = r->hands[p]; return remove_cards(&h, a->cards, count);
}
PresidentStatus president_round_apply(PresidentRound *r, int p, const PresidentAction *input, size_t *first_event) {
    PresidentAction a;
    Event *e;
    Event *next_events;
    History *next_history;
    size_t start;
    int receiver, i, chosen, equal, rank;
    if (!r) return PRESIDENT_INVALID_ARGUMENT;
    if (!legal(r, p, input)) return PRESIDENT_ILLEGAL_ACTION;
    a = *input; /* input may refer to a history entry invalidated by realloc */
    if (r->event_count > SIZE_MAX - 16 || r->history_count == SIZE_MAX)
        return PRESIDENT_OUT_OF_MEMORY;
    next_events = reserve(r->events, &r->event_capacity, r->event_count + 16, sizeof *r->events);
    if (!next_events) return PRESIDENT_OUT_OF_MEMORY;
    r->events = next_events;
    next_history = reserve(r->history, &r->history_capacity, r->history_count + 1, sizeof *r->history);
    if (!next_history) return PRESIDENT_OUT_OF_MEMORY;
    r->history = next_history;
    start = r->event_count;
    switch (a.type) {
    case PRESIDENT_CHOOSE_PILE:
        r->hands[p] = r->piles[a.pile]; r->piles[a.pile].count = 0;
        event(r, PILE_CHOSEN, p)->value = a.pile;
        if (++r->draft_index == 4) { r->phase = TRADE_REQUEST; event(r, DRAFT_COMPLETE, -1); }
        break;
    case PRESIDENT_TRADE_REQUEST:
        receiver = trade_receiver(r); chosen = -1;
        for (i = 0; i < r->hands[receiver].count; ++i) {
            int c = r->hands[receiver].cards[i];
            if (c / 4 == a.rank && (chosen < 0 || c < chosen)) chosen = c;
        }
        e = event(r, TRADE_REQUESTED, p); e->receiver = receiver; e->value = a.rank; e->flag = chosen >= 0;
        if (chosen >= 0) {
            remove_cards(&r->hands[receiver], &chosen, 1);
            r->hands[p].cards[r->hands[p].count++] = chosen; r->phase = TRADE_RETURN;
        }
        break;
    case PRESIDENT_TRADE_RETURN:
        receiver = trade_receiver(r); remove_cards(&r->hands[p], a.cards, 1);
        r->hands[receiver].cards[r->hands[receiver].count++] = a.cards[0];
        event(r, TRADE_COMPLETED, p)->receiver = receiver;
        if (++r->trade_index == 3) {
            r->phase = PLAY; r->turn = by_role(r, 4); event(r, TRADES_COMPLETE, r->turn); check_twos(r);
        } else r->phase = TRADE_REQUEST;
        break;
    case PRESIDENT_PLAY:
        rank = a.cards[0] / 4; equal = rank == r->top_rank;
        remove_cards(&r->hands[p], a.cards, a.count);
        r->top_stack_count = equal ? r->top_stack_count + a.count : a.count;
        r->top_rank = rank; r->top_count = a.count; r->last_player = p;
        e = event(r, r->opened ? PLAYED : OPENING_PLAY, p); r->opened = 1;
        e->count = a.count; memcpy(e->cards, a.cards, (size_t)a.count * sizeof(int));
        finish_if_empty(r, p, 0); check_twos(r);
        if (r->phase != FINISHED) {
            if (r->top_stack_count == 4) clear_to(r, p);
            else offer_completions(r, p, equal);
        }
        break;
    case PRESIDENT_PASS:
        event(r, PASSED, p); offer_completions(r, p, 0); break;
    case PRESIDENT_BOMB:
        remove_cards(&r->hands[p], a.cards, 1); event(r, BOMBED, p)->cards[0] = a.cards[0];
        finish_if_empty(r, p, 1); check_twos(r);
        if (r->phase != FINISHED) clear_to(r, p);
        break;
    case PRESIDENT_COMPLETE:
        remove_cards(&r->hands[p], a.cards, a.count);
        e = event(r, COMPLETED, p); e->count = a.count; memcpy(e->cards, a.cards, (size_t)a.count * sizeof(int));
        finish_if_empty(r, p, 0); check_twos(r);
        if (r->phase != FINISHED) clear_to(r, p);
        break;
    case PRESIDENT_DECLINE_COMPLETION:
        if (++r->completion_cursor == r->completion_count) {
            int previous = r->pending_player, skip = r->pending_skip;
            r->pending_player = -1; r->phase = PLAY; advance(r, previous, skip);
        }
        break;
    }
    r->history[r->history_count].player = p;
    r->history[r->history_count++].action = a;
    if (first_event) *first_event = start;
    return PRESIDENT_OK;
}
size_t president_round_event_count(const PresidentRound *r) { return r ? r->event_count : 0; }
size_t president_round_history_count(const PresidentRound *r) { return r ? r->history_count : 0; }
PresidentStatus president_round_history_at(const PresidentRound *r, size_t index, int *p, PresidentAction *a) {
    if (!r || index >= r->history_count || !p || !a) return PRESIDENT_INVALID_ARGUMENT;
    *p = r->history[index].player; *a = r->history[index].action; return PRESIDENT_OK;
}
PresidentStatus president_round_rankings(const PresidentRound *r, int out[4]) {
    if (!r || !out) return PRESIDENT_INVALID_ARGUMENT;
    if (r->phase != FINISHED) return PRESIDENT_UNFINISHED;
    memcpy(out, r->rankings, sizeof r->rankings); return PRESIDENT_OK;
}

/* JSON output: append-only buffer, with sticky allocation failure. */
typedef struct { char *data; size_t length, capacity; int failed; } Json;
static void append(Json *j, const char *text) {
    size_t n;
    char *next;
    if (j->failed) return;
    n = strlen(text);
    if (n >= SIZE_MAX - j->length) {
        j->failed = 1; return;
    }
    next = reserve(j->data, &j->capacity, j->length + n + 1, 1);
    if (!next) { j->failed = 1; return; }
    j->data = next;
    memcpy(j->data + j->length, text, n + 1); j->length += n;
}
static void number(Json *j, int n) { char buffer[32]; snprintf(buffer, sizeof buffer, "%d", n); append(j, buffer); }
static void string(Json *j, const char *s) {
    const unsigned char *p = (const unsigned char *)s;
    append(j, "\"");
    for (; *p; ++p) {
        char buffer[7];
        if (*p == '"' || *p == '\\') { buffer[0] = '\\'; buffer[1] = (char)*p; buffer[2] = 0; }
        else if (*p < 32) snprintf(buffer, sizeof buffer, "\\u%04x", (unsigned)*p);
        else { buffer[0] = (char)*p; buffer[1] = 0; }
        append(j, buffer);
    }
    append(j, "\"");
}
static void player_json(Json *j, const PresidentRound *r, int p) {
    if (p < 0) append(j, "null"); else string(j, r->players[p]);
}
static void card_json(Json *j, int c) {
    append(j, "{\"rank\":"); string(j, ranks[c / 4]); append(j, ",\"suit\":"); string(j, suits[c % 4]); append(j, "}");
}
static void cards_json(Json *j, const int *cards, int count) {
    int i; append(j, "[");
    for (i = 0; i < count; ++i) { if (i) append(j, ","); card_json(j, cards[i]); }
    append(j, "]");
}
static void placements_json(Json *j, const PresidentRound *r, const int values[4]) {
    int i, comma = 0; append(j, "{");
    for (i = 0; i < 4; ++i) if (values[i]) {
        if (comma++) append(j, ",");
        player_json(j, r, i); append(j, ":"); number(j, values[i]);
    }
    append(j, "}");
}
static void action_json(Json *j, const PresidentAction *a) {
    append(j, "{\"type\":"); string(j, actions[a->type]);
    switch (a->type) {
    case PRESIDENT_CHOOSE_PILE: append(j, ",\"pile\":"); number(j, a->pile); break;
    case PRESIDENT_TRADE_REQUEST: append(j, ",\"rank\":"); string(j, ranks[a->rank]); break;
    case PRESIDENT_TRADE_RETURN: case PRESIDENT_BOMB: append(j, ",\"card\":"); card_json(j, a->cards[0]); break;
    case PRESIDENT_PLAY: case PRESIDENT_COMPLETE: append(j, ",\"cards\":"); cards_json(j, a->cards, a->count); break;
    default: break;
    }
    append(j, "}");
}
static void event_json(Json *j, const PresidentRound *r, const Event *e) {
    int i;
    append(j, "{\"type\":"); string(j, event_names[e->type]);
    switch (e->type) {
    case ROUND_STARTED:
        append(j, ",\"first_round\":"); append(j, e->flag ? "true" : "false");
        if (e->flag) { append(j, ",\"first_player\":"); player_json(j, r, e->player); }
        else {
            append(j, ",\"face_up\":"); cards_json(j, e->cards, 4);
            append(j, ",\"draft_order\":[");
            for (i = 0; i < 4; ++i) { if (i) append(j, ","); player_json(j, r, e->list[i]); }
            append(j, "]");
        }
        break;
    case TRADE_REQUESTED: case TRADE_COMPLETED:
        append(j, ",\"initiator\":"); player_json(j, r, e->player);
        append(j, ",\"receiver\":"); player_json(j, r, e->receiver);
        if (e->type == TRADE_REQUESTED) {
            append(j, ",\"rank\":"); string(j, ranks[e->value]);
            append(j, ",\"success\":"); append(j, e->flag ? "true" : "false");
        }
        break;
    case TRADES_COMPLETE: append(j, ",\"first_player\":"); player_json(j, r, e->player); break;
    case ROUND_FINISHED: append(j, ",\"rankings\":"); placements_json(j, r, e->list); break;
    case DRAFT_COMPLETE: break;
    default:
        append(j, ",\"player\":"); player_json(j, r, e->player);
        if (e->type == PILE_CHOSEN) { append(j, ",\"pile\":"); number(j, e->value); }
        else if (e->type == BOMBED) { append(j, ",\"card\":"); card_json(j, e->cards[0]); }
        else if (e->type == OPENING_PLAY || e->type == PLAYED || e->type == COMPLETED) {
            append(j, ",\"cards\":"); cards_json(j, e->cards, e->count);
        } else if (e->type == PLAYER_DONE) {
            if (e->flag == 2) append(j, ",\"last_active\":true");
            else { append(j, ",\"bomb_finish\":"); append(j, e->flag ? "true" : "false"); }
        }
        break;
    }
    append(j, "}");
}
static void events_json(Json *j, const PresidentRound *r, size_t start) {
    size_t i; append(j, "[");
    for (i = start; i < r->event_count; ++i) { if (i != start) append(j, ","); event_json(j, r, &r->events[i]); }
    append(j, "]");
}
static char *json_result(Json *j) { if (j->failed) { free(j->data); return NULL; } return j->data; }
char *president_round_events_json(const PresidentRound *r, size_t start) {
    Json j = {0};
    if (!r || start > r->event_count) return NULL;
    events_json(&j, r, start); return json_result(&j);
}
char *president_round_message_json(const PresidentRound *r, int p) {
    Json j = {0}; Hand hand;
    int i, comma = 0, actor;
    if (!r || !valid_player(p)) return NULL;
    actor = president_round_actor(r); hand = sorted_hand(r, p);
    append(&j, "{\"type\":\"state\",\"phase\":"); string(&j, phases[r->phase]);
    append(&j, ",\"turn\":"); player_json(&j, r, actor);
    append(&j, ",\"top\":");
    if (r->top_rank < 0) append(&j, "null");
    else {
        append(&j, "{\"rank\":"); string(&j, ranks[r->top_rank]);
        append(&j, ",\"count\":"); number(&j, r->top_count);
        append(&j, ",\"stack_count\":"); number(&j, r->top_stack_count); append(&j, "}");
    }
    append(&j, ",\"players\":[");
    for (i = 0; i < 4; ++i) { if (i) append(&j, ","); player_json(&j, r, i); }
    append(&j, "],\"roles\":"); placements_json(&j, r, r->roles);
    append(&j, ",\"active_players\":[");
    for (i = 0; i < 4; ++i) if (r->active[i]) { if (comma++) append(&j, ","); player_json(&j, r, i); }
    append(&j, "],\"events\":"); events_json(&j, r, 0);
    append(&j, ",\"rankings\":");
    if (r->phase == FINISHED) placements_json(&j, r, r->rankings); else append(&j, "null");
    append(&j, ",\"player\":"); player_json(&j, r, p);
    append(&j, ",\"hand\":"); cards_json(&j, hand.cards, hand.count);
    append(&j, ",\"request\":");
    if (actor == p && r->phase != FINISHED) {
        /* At most 12 * (2^4 - 1) non-bomb plays plus pass and four bombs. */
        PresidentAction legal_actions[185];
        size_t k, n = president_round_legal_actions(r, p, legal_actions, 185);
        append(&j, "{\"type\":"); string(&j, phases[r->phase]); append(&j, ",\"legal_actions\":[");
        for (k = 0; k < n; ++k) { if (k) append(&j, ","); action_json(&j, &legal_actions[k]); }
        append(&j, "]}");
    } else append(&j, "null");
    append(&j, "}"); return json_result(&j);
}

PresidentStatus president_game_create(PresidentGame **out, const char *const players[4], uint64_t seed) {
    PresidentGame *g;
    PresidentStatus status;
    if (!out) return PRESIDENT_INVALID_ARGUMENT;
    *out = NULL; g = calloc(1, sizeof *g);
    if (!g) return PRESIDENT_OUT_OF_MEMORY;
    status = copy_players(g->players, players);
    if (status != PRESIDENT_OK) { president_game_destroy(g); return status; }
    g->rng = seed; *out = g; return PRESIDENT_OK;
}
void president_game_destroy(PresidentGame *g) {
    int i;
    if (!g) return;
    president_round_destroy(g->round);
    for (i = 0; i < 4; ++i) free(g->players[i]);
    free(g);
}
PresidentStatus president_game_start_round(PresidentGame *g, const PresidentCard *deck, size_t count) {
    PresidentRound *next;
    PresidentStatus status;
    uint64_t rng;
    int cards[52];
    if (!g) return PRESIDENT_INVALID_ARGUMENT;
    if (g->round && !president_round_done(g->round)) return PRESIDENT_UNFINISHED;
    rng = g->rng;
    if (!deck && !count) { president_standard_deck(cards); shuffle(cards, &rng); deck = cards; count = 52; }
    status = president_round_create(&next, (const char *const *)g->players, g->round_number == 0,
        g->round ? g->round->rankings : NULL, deck, count, rng);
    if (status != PRESIDENT_OK) return status;
    president_round_destroy(g->round); g->round = next; g->rng = rng; ++g->round_number;
    return PRESIDENT_OK;
}
PresidentRound *president_game_round(PresidentGame *g) { return g ? g->round : NULL; }
size_t president_game_round_number(const PresidentGame *g) { return g ? g->round_number : 0; }
PresidentStatus president_game_finish_round(PresidentGame *g, int out[4]) {
    if (!g || !out) return PRESIDENT_INVALID_ARGUMENT;
    if (!g->round) return PRESIDENT_UNFINISHED;
    return president_round_rankings(g->round, out);
}
static PresidentStatus notify_clients(PresidentGame *g, const PresidentClient clients[4], char **request) {
    char *messages[4] = {0};
    int p, actor = president_round_actor(g->round);
    *request = NULL;
    /* Allocate every snapshot before invoking observers. The response snapshot
     * is separate even from the actor's observation, as in the Python adapter. */
    for (p = 0; p < 4; ++p) {
        messages[p] = president_round_message_json(g->round, p);
        if (!messages[p]) goto failed;
    }
    if (actor >= 0) {
        *request = president_round_message_json(g->round, actor);
        if (!*request) goto failed;
    }
    for (p = 0; p < 4; ++p) {
        if (clients[p].observe) clients[p].observe(clients[p].context, messages[p]);
        free(messages[p]);
    }
    return PRESIDENT_OK;
failed:
    for (p = 0; p < 4; ++p) free(messages[p]);
    free(*request); *request = NULL; return PRESIDENT_OUT_OF_MEMORY;
}
PresidentStatus president_game_run(PresidentGame *g, const PresidentClient clients[4], size_t max_actions) {
    PresidentStatus status;
    char *request = NULL;
    size_t i;
    if (!g || !clients) return PRESIDENT_INVALID_ARGUMENT;
    if (!g->round) { status = president_game_start_round(g, NULL, 0); if (status != PRESIDENT_OK) return status; }
    status = notify_clients(g, clients, &request);
    if (status != PRESIDENT_OK) return status;
    for (i = 0; i < max_actions && !president_round_done(g->round); ++i) {
        PresidentAction a = {0};
        int p = president_round_actor(g->round);
        if (p < 0 || !clients[p].respond) { free(request); return PRESIDENT_CLIENT_ERROR; }
        status = clients[p].respond(clients[p].context, request, &a);
        free(request); request = NULL;
        if (status != PRESIDENT_OK) return status;
        status = president_round_apply(g->round, p, &a, NULL);
        if (status != PRESIDENT_OK) return status;
        status = notify_clients(g, clients, &request);
        if (status != PRESIDENT_OK) return status;
    }
    free(request);
    return president_round_done(g->round) ? PRESIDENT_OK : PRESIDENT_ACTION_LIMIT;
}

#ifdef PRESIDENT_SEARCH
#include "agent_search_native.inc"
#endif
