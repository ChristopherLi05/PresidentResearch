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
/* ABI version 1; weights order is defined in agent_search_c.py. */
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
    double budget_ms, double *scores);
/* Diagnostic parity interfaces for tests and profiling. */
int president_search_scores(const SearchPosition *, int player,
    const PresidentAction *, int count, const double weights[18], double *scores);
int president_search_rollout(const SearchPosition *, int player,
    const PresidentAction *, const double weights[18], SearchPosition *, double *score);
int president_search_policy(const SearchPosition *, const double weights[18], PresidentAction *);
#endif
