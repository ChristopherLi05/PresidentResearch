#ifndef AGENT_SEARCH_NATIVE_H
#define AGENT_SEARCH_NATIVE_H
#include "president.h"
/* Fixed-width value snapshot. No pointers or host-private information. */
typedef struct {
    int cards[4][52], sizes[4], active[4];
    int finish[4], finish_count, bombs[4], bomb_count;
    int top_rank, top_count, stack, last_player, turn, opened, first_round, phase;
    int completion[4], completion_count, cursor, pending_player, pending_skip;
    int rankings[4];
} SearchPosition;
/* Rollout controls. A NULL pointer selects the defaults {16, 0, 1.0}.
 * depth: maximum normal (play-phase) actions in a rollout, counting the root
 *   action when it is a play-phase action; negative means unlimited. Pending
 *   completion windows are always resolved before the cutoff applies.
 * through_own_turns: 0 stops at our own next normal turn; 1 lets our later
 *   decisions use the same heuristic policy, so with unlimited depth the
 *   rollout ends only at our exit or round end (leaf fully resolved).
 * leaf_scale: finite, >= 0; multiplies the hand-cost difference inside the
 *   leaf sigmoid for unresolved pairs. */
typedef struct {
    int depth, through_own_turns;
    double leaf_scale;
} SearchOptions;
/* One piece of opponent evidence for the modeled sampler (agent_belief.py).
 * Counts at that moment are current counts + offset (floored at 0).
 * kind/data: 0 play {top rank or -1, width, smallest, observed (-1 pass,
 * -2 bomb, rank*8+width)}; 1 no completion {rank, needed}; 2 trade request
 * {failed-rank bitmask, requested rank}; 3 none {rank, later returns};
 * 4 some {rank}; 5 first-round opening {count of 3s}. */
typedef struct {
    int kind, offset[13], data[4];
} BeliefObservation;
/* ABI version 3; weights order is defined in agent_search_c.py. */
int president_search_version(void);
int president_hand_value(const int hand[13], const int unseen[13],
    const int order[13], const int sizes[3], int opponents,
    const double weights[18], int lead, double *value);
/* Returns 1 for a complete batch, 0 on deadline, -1 on invalid input.
 * All outputs of an incomplete batch must be discarded. Negative budget
 * disables the deadline. No global mutable state, heap allocations, JSON,
 * Python callbacks, or engine event history are needed by the search. */
int president_search_batch(const SearchPosition *, int player,
    const PresidentAction *, int count, const double weights[18],
    const SearchOptions *, double budget_ms, double *scores);
/* Diagnostic parity interfaces for tests and profiling. */
int president_search_scores(const SearchPosition *, int player,
    const PresidentAction *, int count, const double weights[18], double *scores);
int president_search_rollout(const SearchPosition *, int player,
    const PresidentAction *, const double weights[18], const SearchOptions *,
    SearchPosition *, double *score);
int president_search_policy(const SearchPosition *, const double weights[18], PresidentAction *);
/* Modeled opponent hands. Runs `steps` Metropolis swaps in place on `cards`
 * (card codes rank*4+suit; opponent p owns slots[p] consecutive entries),
 * scoring each opponent's counts (cards plus pinned[p]) against their own
 * consecutive obs_count[p] observations. rng is splitmix64 state, updated.
 * stats[0] += steps, stats[1] += accepted swaps. Returns 1, or -1 if invalid. */
int president_belief_walk(int opponents, const int slots[3], int *cards, const int pinned[3][13],
    const BeliefObservation *obs, const int obs_count[3], double noise,
    int steps, uint64_t *rng, int *stats);
/* Diagnostic: log weight of one hand (agent_belief.log_likelihood). */
int president_belief_loglik(const int counts[13], const BeliefObservation *obs, int count,
    double noise, double *out);
#endif
