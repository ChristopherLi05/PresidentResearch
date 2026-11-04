/* Included only by president.c with PRESIDENT_SEARCH defined. Sharing this
 * translation unit deliberately reuses the authoritative engine transitions.
 * The normal C engine build has no dependency on this optional evaluator. */
#include "agent_search_native.h"

enum Weight { RESERVE, GROUP, CARD_COST, SINGLE, HAND_CONTROL, BOMB_GROUP,
    BOMB_COVER, BOMB_VALUE, LEAD_CONTROL, STRANDING, PASS_COST, PASS_DANGER,
    CLEAR_COST, FINISH_COST, NEXT_RESPONSE, NEXT_FINISH, NEXT_COMPLETION, NEXT_BOMB };
typedef struct { double choose[53][53]; const double *w; } SearchContext;
typedef struct {
    int hand[13], unseen[13], sizes[3], order[13], opponents, pool, top, width, stack;
    double holding[3][5][5], control[13][5];
    const SearchContext *search;
} Evaluation;

static double search_now(void) {
#ifdef _WIN32
    LARGE_INTEGER value, frequency;
    QueryPerformanceCounter(&value); QueryPerformanceFrequency(&frequency);
    return (double)value.QuadPart / (double)frequency.QuadPart;
#else
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + value.tv_nsec * 1e-9;
#endif
}
static double smaller(double a, double b) { return a < b ? a : b; }
static double larger(double a, double b) { return a > b ? a : b; }
/* Match modern Python sum's compensated accumulation. Tiny rounding changes
 * otherwise break ties between strategically identical low-card plays. */
typedef struct { double hi, lo; } Sum;
static void sum_add(Sum *s, double x) {
    double t = s->hi+x;
    s->lo += fabs(s->hi) >= fabs(x) ? (s->hi-t)+x : (x-t)+s->hi;
    s->hi = t;
}
static double sum_value(Sum s) { return s.hi+s.lo; }
static void search_context(SearchContext *s, const double *w) {
    int n, k;
    memset(s, 0, sizeof *s); s->w = w;
    for (n = 0; n <= 52; ++n) {
        s->choose[n][0] = s->choose[n][n] = 1;
        for (k = 1; k < n; ++k) s->choose[n][k] = s->choose[n-1][k-1] + s->choose[n-1][k];
    }
}
static double holding(const SearchContext *s, int pool, int copies, int size, int needed) {
    double sum = 0;
    int k;
    if (size > pool) size = pool;
    if (pool <= 0 || copies > pool || needed > copies || needed > size) return 0;
    for (k = needed; k <= copies && k <= size; ++k)
        if (size-k <= pool-copies) sum += s->choose[copies][k] * s->choose[pool-copies][size-k];
    return sum / s->choose[pool][size];
}
static void evaluation(Evaluation *e, const SearchContext *s, const PresidentRound *r, int p) {
    int i, j, q, rank, copies, needed;
    memset(e, 0, sizeof *e); e->search = s;
    for (i = 0; i < 13; ++i) e->order[i] = i;
    e->top = r->top_rank; e->width = r->top_count; e->stack = r->top_stack_count;
    for (q = 0; q < 4; ++q) for (i = 0; i < r->hands[q].count; ++i) {
        rank = r->hands[q].cards[i] / 4;
        if (q == p) ++e->hand[rank]; else ++e->unseen[rank];
    }
    for (rank = 0; rank < 13; ++rank) e->pool += e->unseen[rank];
    for (i = 1; i < 4; ++i) {
        q = (p+i)%4;
        if (r->active[q] && r->hands[q].count) e->sizes[e->opponents++] = r->hands[q].count;
    }
    for (j = 0; j < e->opponents; ++j) for (copies = 0; copies <= 4; ++copies)
        for (needed = 1; needed <= 4; ++needed)
            e->holding[j][copies][needed] = holding(s, e->pool, copies, e->sizes[j], needed);
    for (rank = 0; rank < 13; ++rank) for (i = 0; i <= 4; ++i) e->control[rank][i] = -1;
}
static double control(Evaluation *e, int rank, int width) {
    double value = 1;
    int j, other;
    if (width == 4) return 1;
    if (e->control[rank][width] >= 0) return e->control[rank][width];
    for (j = 0; j < e->opponents; ++j) {
        double response = e->holding[j][e->unseen[12]][1];
        Sum sum = {0};
        for (other = rank; other < 12; ++other) sum_add(&sum,e->holding[j][e->unseen[other]][width]);
        response += sum_value(sum);
        response += e->holding[j][e->unseen[rank]][4-width];
        value *= 1 - smaller(1, response);
    }
    e->control[rank][width] = value; return value;
}
static int card_total(const int *hand) {
    int r, n = 0; for (r = 0; r < 13; ++r) n += hand[r]; return n;
}
static double hand_cost(Evaluation *e, const int *hand) {
    const double *w = e->search->w;
    double probabilities[12], cost = 0, reserve = 0, support = 0;
    Sum costs = {0}, reserves = {0}, supports = {0};
    int r, j, index, groups = 0, cover = 0, twos = hand[12];
    for (index = 0; index < 13; ++index) { r = e->order[index]; if (r != 12 && hand[r]) {
        double c = control(e, r, hand[r]);
        probabilities[groups++] = c; cover += hand[r];
        sum_add(&costs,w[GROUP] + w[CARD_COST]*hand[r] + (hand[r] == 1 ? w[SINGLE] : 0) - w[HAND_CONTROL]*c);
    } }
    if (!groups) return twos ? 60.0 + twos : 0;
    cost = sum_value(costs);
    if (w[RESERVE]) {
        double patience = 0;
        for (r = 0; r < groups; ++r) {
            double square = probabilities[r]*probabilities[r];
            Sum sum = {0};
            sum_add(&reserves,square);
            for (j = 0; j < groups; ++j) if (r != j) sum_add(&sum,1-probabilities[j]);
            sum_add(&supports,square*sum_value(sum));
        }
        reserve = sum_value(reserves); support = sum_value(supports);
        if (e->opponents) {
            int closest = 13, total = 0;
            for (j = 0; j < e->opponents; ++j) {
                int size = e->sizes[j] > 13 ? 13 : e->sizes[j];
                total += size; if (size < closest) closest = size;
            }
            patience = ((double)total/(13*e->opponents)) * (16.0*closest/(13*(closest+3)));
        }
        cost -= w[RESERVE]*patience*support/((1+reserve)*groups);
    }
    cost += w[BOMB_GROUP]*larger(0, twos-(groups-1));
    cost += w[BOMB_COVER]*larger(0, twos-(cover-1));
    return cost - w[BOMB_VALUE]*twos;
}
static double lead_cost(Evaluation *e, const int *hand) {
    const double *w = e->search->w;
    int r, width, rest[13], total = card_total(hand);
    double best = HUGE_VAL;
    memcpy(rest, hand, sizeof rest);
    for (r = 0; r < 12; ++r) for (width = 1; width <= hand[r]; ++width) {
        double value;
        rest[r] = hand[r]-width;
        if (total == width) value = -20;
        else {
            double c = control(e, r, width);
            value = hand_cost(e, rest) - w[LEAD_CONTROL]*c;
            if (rest[12] && total-width-rest[12] <= rest[12]) value += w[STRANDING]*c;
        }
        best = smaller(best, value); rest[r] = hand[r];
    }
    return best == HUGE_VAL ? hand_cost(e, hand) : best;
}
static double turn_danger(Evaluation *e, const PresidentAction *a) {
    const double *w = e->search->w;
    int rank, width, stack, normal, size, other, needed;
    double response = 0, bomb = 0, finish = 0, completion;
    if (!e->opponents || a->type == PRESIDENT_BOMB || a->type == PRESIDENT_COMPLETE) return 0;
    if (a->type == PRESIDENT_PASS || a->type == PRESIDENT_DECLINE_COMPLETION) {
        if (e->top < 0) return 0;
        rank = e->top; width = e->width; stack = e->stack; normal = 1;
    } else {
        rank = a->cards[0]/4; width = a->count;
        stack = width + (rank == e->top ? e->stack : 0);
        if (width == 4 || stack == 4) return 0;
        normal = rank != e->top;
    }
    size = e->sizes[0];
    if (!size || !e->pool) return 0;
    if (normal) {
        Sum sum = {0};
        for (other = rank; other < 12; ++other) sum_add(&sum,e->holding[0][e->unseen[other]][width]);
        response = sum_value(sum);
        if (size > 1) bomb = e->holding[0][e->unseen[12]][1];
        if (size == width) finish = response;
    }
    needed = 4-stack;
    completion = needed > 0 ? e->holding[0][e->unseen[rank]][needed] : 0;
    if (size == needed) finish = smaller(1, finish+completion);
    return w[NEXT_RESPONSE]*smaller(1,response) + w[NEXT_BOMB]*bomb +
        w[NEXT_FINISH]*smaller(1,finish) + w[NEXT_COMPLETION]*completion;
}
static double action_score(Evaluation *e, const PresidentAction *a) {
    const double *w = e->search->w;
    int rank, width, rest[13], total, i, other, stack;
    double value;
    if (a->type == PRESIDENT_PASS || a->type == PRESIDENT_DECLINE_COMPLETION) {
        Sum danger = {0};
        for (i = 0; i < e->opponents; ++i) if (e->sizes[i] <= 3) sum_add(&danger,1.0/e->sizes[i]);
        return hand_cost(e,e->hand)+w[PASS_COST]+w[PASS_DANGER]*sum_value(danger)+turn_danger(e,a);
    }
    rank = a->cards[0]/4; width = a->type == PRESIDENT_BOMB ? 1 : a->count;
    memcpy(rest,e->hand,sizeof rest); rest[rank] -= width;
    total = card_total(rest);
    if (!total) return a->type == PRESIDENT_BOMB ? 100 : -100;
    stack = width + (rank == e->top ? e->stack : 0);
    if (a->type == PRESIDENT_BOMB || a->type == PRESIDENT_COMPLETE || width == 4 || stack == 4) {
        value = w[CLEAR_COST]+lead_cost(e,rest);
        if (a->type == PRESIDENT_BOMB && e->hand[12] > card_total(e->hand)-e->hand[12])
            value = smaller(value,hand_cost(e,rest));
    } else {
        double c = control(e,rank,width);
        value = hand_cost(e,rest)-w[LEAD_CONTROL]*c;
        for (i = 0; i < e->opponents; ++i) {
            int size = e->sizes[i];
            double finish = 0;
            if (size == width && !(i == 0 && rank == e->top)) {
                Sum sum = {0};
                for (other = rank; other < 12; ++other) sum_add(&sum,e->holding[i][e->unseen[other]][width]);
                finish = sum_value(sum);
            }
            if (size == 4-stack) finish += e->holding[i][e->unseen[rank]][size];
            value += w[FINISH_COST]*smaller(1,finish);
        }
        if (rest[12] && total-rest[12] <= rest[12]) value += w[STRANDING]*c;
    }
    return value+turn_danger(e,a);
}

static int load_position(PresidentRound *r, const SearchPosition *p) {
    int seat, i, seen[52] = {0}, retired = 0, respondents = 0, rank_mask = 0;
    memset(r,0,sizeof *r);
    if (!p || p->phase < PLAY || p->phase > FINISHED || p->finish_count < 0 || p->finish_count > 4 ||
        p->bomb_count < 0 || p->bomb_count+p->finish_count > 4 || p->completion_count < 0 || p->completion_count > 4 ||
        p->top_rank < -1 || p->top_rank > 11 || p->top_count < 0 || p->top_count > 4 ||
        p->stack < 0 || p->stack > 4 || p->last_player < -1 || p->last_player > 3 ||
        (p->phase == PLAY && !valid_player(p->turn)) ||
        (p->phase == COMPLETION && (p->cursor < 0 || p->cursor >= p->completion_count || !valid_player(p->pending_player)))) return 0;
    if ((p->top_rank < 0 && (p->top_count || p->stack || p->phase == COMPLETION)) ||
        (p->top_rank >= 0 && (p->top_count < 1 || p->stack < p->top_count)) ||
        (p->phase == COMPLETION && p->stack >= 4)) return 0;
    for (seat = 0; seat < 4; ++seat) {
        if (p->sizes[seat] < 0 || p->sizes[seat] > 52 || (p->active[seat] != 0 && p->active[seat] != 1)) return 0;
        if (p->active[seat] && (!p->sizes[seat] || p->phase == FINISHED)) return 0;
        r->hands[seat].count = p->sizes[seat]; r->active[seat] = p->active[seat];
        for (i = 0; i < p->sizes[seat]; ++i) {
            int c = p->cards[seat][i];
            if (!valid_card(c) || seen[c]++) return 0;
            r->hands[seat].cards[i] = c;
        }
    }
    for (i = 0; i < p->finish_count+p->bomb_count; ++i) {
        int seat_ = i < p->finish_count ? p->finish[i] : p->bombs[i-p->finish_count];
        if (!valid_player(seat_) || r->active[seat_] || (retired & (1<<seat_))) return 0;
        retired |= 1<<seat_;
    }
    for (seat = 0; seat < 4; ++seat) {
        if (!r->active[seat] && !(retired & (1<<seat))) return 0;
        if (p->phase == FINISHED) {
            if (p->rankings[seat] < 1 || p->rankings[seat] > 4 || (rank_mask & (1<<p->rankings[seat]))) return 0;
            rank_mask |= 1<<p->rankings[seat];
        }
    }
    if (p->phase == PLAY && !r->active[p->turn]) return 0;
    for (i = 0; i < p->completion_count; ++i) {
        int respondent = p->completion[i];
        if (!valid_player(respondent) || (respondents & (1<<respondent))) return 0;
        if (p->phase == COMPLETION && !r->active[respondent]) return 0;
        respondents |= 1<<respondent;
    }
    memcpy(r->finish_order,p->finish,sizeof p->finish); r->finish_count = p->finish_count;
    memcpy(r->bomb_finishers,p->bombs,sizeof p->bombs); r->bomb_count = p->bomb_count;
    memcpy(r->rankings,p->rankings,sizeof p->rankings);
    r->top_rank = p->top_rank; r->top_count = p->top_count; r->top_stack_count = p->stack;
    r->last_player = p->last_player; r->turn = p->turn; r->opened = p->opened;
    r->first_round = p->first_round; r->phase = (enum Phase)p->phase;
    memcpy(r->completion_order,p->completion,sizeof p->completion);
    r->completion_count = p->completion_count; r->completion_cursor = p->cursor;
    r->pending_player = p->pending_player; r->pending_skip = p->pending_skip;
    return 1;
}
static void save_position(SearchPosition *p, const PresidentRound *r) {
    int i;
    memset(p,0,sizeof *p);
    for (i = 0; i < 4; ++i) {
        p->sizes[i] = r->hands[i].count; p->active[i] = r->active[i];
        memcpy(p->cards[i],r->hands[i].cards,(size_t)p->sizes[i]*sizeof(int));
    }
    memcpy(p->finish,r->finish_order,sizeof p->finish); p->finish_count = r->finish_count;
    memcpy(p->bombs,r->bomb_finishers,sizeof p->bombs); p->bomb_count = r->bomb_count;
    memcpy(p->rankings,r->rankings,sizeof p->rankings);
    p->top_rank = r->top_rank; p->top_count = r->top_count; p->stack = r->top_stack_count;
    p->last_player = r->last_player; p->turn = r->turn; p->opened = r->opened;
    p->first_round = r->first_round; p->phase = r->phase;
    memcpy(p->completion,r->completion_order,sizeof p->completion);
    p->completion_count = r->completion_count; p->cursor = r->completion_cursor;
    p->pending_player = r->pending_player; p->pending_skip = r->pending_skip;
}
static int same_move(const PresidentAction *a, const PresidentAction *b) {
    if (a->type != b->type) return 0;
    if (a->type == PRESIDENT_PASS || a->type == PRESIDENT_DECLINE_COMPLETION) return 1;
    return a->cards[0]/4 == b->cards[0]/4 && a->count == b->count;
}
static PresidentAction policy(const SearchContext *s, const PresidentRound *r, int p) {
    PresidentAction all[200], best;
    size_t count = president_round_legal_actions(r,p,all,200), i, j;
    double value = HUGE_VAL;
    Evaluation e;
    /* At most 12*15 ordinary combinations + four bombs + pass. */
    memset(&best,0,sizeof best); best.type = (PresidentActionType)-1;
    if (!count || count > 200) return best;
    best = all[0];
    if (count == 1) return best;
    evaluation(&e,s,r,p);
    for (i = 0; i < count; ++i) {
        double score;
        if ((all[i].type == PRESIDENT_PLAY || all[i].type == PRESIDENT_COMPLETE) && all[i].count == r->hands[p].count) return all[i];
        for (j = 0; j < i; ++j) if (same_move(&all[i],&all[j])) break;
        if (j < i) continue;
        score = action_score(&e,&all[i]);
        if (score < value) { value = score; best = all[i]; }
    }
    return best;
}
static int placement_order(const PresidentRound *r, int p) {
    int i;
    for (i = 0; i < r->finish_count; ++i) if (r->finish_order[i] == p) return i;
    for (i = 0; i < r->bomb_count; ++i) if (r->bomb_finishers[i] == p) return 8+i;
    return 4;
}
static double leaf(const SearchContext *s, const PresidentRound *r, int p) {
    int q, own_order = placement_order(r,p), has_own = 0;
    double total = 0, own = 0;
    for (q = 0; q < 4; ++q) if (q != p) {
        int other_order = placement_order(r,q);
        if (r->phase == FINISHED) total += r->rankings[p] < r->rankings[q];
        else if (own_order != 4 || other_order != 4) total += own_order < other_order;
        else {
            Evaluation e;
            double other, difference;
            if (!has_own) {
                evaluation(&e,s,r,p);
                own = r->top_rank < 0 && president_round_actor(r) == p ? lead_cost(&e,e.hand) : hand_cost(&e,e.hand);
                has_own = 1;
            }
            evaluation(&e,s,r,q);
            other = r->top_rank < 0 && president_round_actor(r) == q ? lead_cost(&e,e.hand) : hand_cost(&e,e.hand);
            difference = larger(-60,smaller(60,own-other));
            total += 1/(1+exp(difference));
        }
    }
    return total/3;
}
static int apply_search(PresidentRound *r, int actor, const PresidentAction *a) {
    /* Retain only current-action scratch logs. opened carries the sole
     * history-dependent rule. Fixed buffers satisfy apply's reserve calls. */
    r->event_count = r->history_count = 0;
    return president_round_apply(r,actor,a,NULL) == PRESIDENT_OK;
}
static int simulate(const SearchContext *s, PresidentRound *r, int p, const PresidentAction *a, double deadline) {
    int normal = r->phase == PLAY;
    if (!apply_search(r,p,a)) return -1;
    while (r->phase != FINISHED && r->active[p]) {
        int actor;
        PresidentAction chosen;
        if (search_now() >= deadline) return 0;
        actor = president_round_actor(r);
        if (r->phase == PLAY && (actor == p || normal >= 16)) break;
        chosen = policy(s,r,actor);
        if (search_now() >= deadline) return 0;
        normal += r->phase == PLAY;
        if (!apply_search(r,actor,&chosen)) return -1;
    }
    return 1;
}
static int valid_weights(const double *w) {
    int i; if (!w) return 0;
    for (i = 0; i < 18; ++i) if (!isfinite(w[i])) return 0;
    return 1;
}
int president_hand_value(const int hand[13], const int unseen[13], const int order[13],
        const int sizes[3], int opponents, const double weights[18], int lead, double *value) {
    SearchContext s; Evaluation e;
    int r, j, copies, needed, mask = 0;
    if (!hand || !unseen || !order || !sizes || !value || opponents < 0 || opponents > 3 ||
        (lead != 0 && lead != 1) || !valid_weights(weights)) return -1;
    memset(&e,0,sizeof e);
    for (r = 0; r < 13; ++r) {
        if (hand[r] < 0 || hand[r] > 4 || unseen[r] < 0 || unseen[r] > 4 ||
            order[r] < 0 || order[r] >= 13 || (mask & (1<<order[r]))) return -1;
        mask |= 1<<order[r]; e.pool += unseen[r];
        e.hand[r] = hand[r]; e.unseen[r] = unseen[r]; e.order[r] = order[r];
        for (j = 0; j <= 4; ++j) e.control[r][j] = -1;
    }
    for (j = 0; j < opponents; ++j) {
        if (sizes[j] < 0 || sizes[j] > 52) return -1;
        e.sizes[j] = sizes[j];
    }
    search_context(&s,weights); e.search = &s; e.opponents = opponents;
    for (j = 0; j < opponents; ++j) for (copies = 0; copies <= 4; ++copies)
        for (needed = 1; needed <= 4; ++needed)
            e.holding[j][copies][needed] = holding(&s,e.pool,copies,e.sizes[j],needed);
    *value = lead ? lead_cost(&e,e.hand) : hand_cost(&e,e.hand);
    return 1;
}
int president_search_version(void) { return 1; }
int president_search_policy(const SearchPosition *position, const double w[18], PresidentAction *out) {
    PresidentRound r; SearchContext s;
    if (!out || !valid_weights(w) || !load_position(&r,position) || r.phase == FINISHED) return -1;
    search_context(&s,w); *out = policy(&s,&r,president_round_actor(&r));
    return (int)out->type < 0 ? -1 : 1;
}
int president_search_batch(const SearchPosition *position, int p, const PresidentAction *a,
        int count, const double w[18], double budget_ms, double *scores) {
    PresidentRound base;
    SearchContext s;
    double deadline = budget_ms < 0 ? HUGE_VAL : search_now()+budget_ms/1000;
    int i;
    if (!valid_player(p) || !a || !scores || count < 1 || count > 200 ||
        !isfinite(budget_ms) || !valid_weights(w) || !load_position(&base,position) || president_round_actor(&base) != p) return -1;
    search_context(&s,w);
    for (i = 0; i < count; ++i) {
        PresidentRound r = base;
        Event events[16]; History history[1];
        int result;
        r.events = events; r.event_capacity = 16; r.history = history; r.history_capacity = 1;
        if (search_now() >= deadline) return 0;
        result = simulate(&s,&r,p,&a[i],deadline);
        if (result != 1) return result;
        scores[i] = leaf(&s,&r,p);
        if (search_now() >= deadline) return 0;
    }
    return 1;
}
int president_search_scores(const SearchPosition *position, int p, const PresidentAction *a,
        int count, const double w[18], double *scores) {
    PresidentRound r; SearchContext s; Evaluation e; int i;
    if (!valid_player(p) || !a || !scores || count < 1 || count > 200 || !valid_weights(w) ||
        !load_position(&r,position)) return -1;
    search_context(&s,w); evaluation(&e,&s,&r,p);
    for (i = 0; i < count; ++i) {
        if (!legal(&r,p,&a[i])) return -1;
        scores[i] = action_score(&e,&a[i]);
    }
    return 1;
}
int president_search_rollout(const SearchPosition *position, int p, const PresidentAction *a,
        const double w[18], SearchPosition *out, double *score) {
    PresidentRound r; SearchContext s; Event events[16]; History history[1]; int status;
    if (!valid_player(p) || !a || !out || !score || !valid_weights(w) || !load_position(&r,position)) return -1;
    search_context(&s,w);
    r.events = events; r.event_capacity = 16; r.history = history; r.history_capacity = 1;
    status = simulate(&s,&r,p,a,HUGE_VAL);
    if (status != 1) return status;
    save_position(out,&r); *score = leaf(&s,&r,p); return 1;
}
