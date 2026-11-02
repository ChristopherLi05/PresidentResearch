#ifndef PRESIDENT_H
#define PRESIDENT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Seats are 0..3 in counterclockwise order. Cards are rank * 4 + suit,
 * with ranks 3,4,5,6,7,8,9,10,J,Q,K,A,2 and suits C,D,H,S. */
typedef int PresidentCard;
typedef enum {
    PRESIDENT_CHOOSE_PILE, PRESIDENT_TRADE_REQUEST, PRESIDENT_TRADE_RETURN,
    PRESIDENT_PLAY, PRESIDENT_PASS, PRESIDENT_BOMB, PRESIDENT_COMPLETE,
    PRESIDENT_DECLINE_COMPLETION
} PresidentActionType;
typedef struct {
    PresidentActionType type;
    int pile;
    int rank;
    int count;
    PresidentCard cards[4]; /* card actions use cards[0]; count is ignored */
} PresidentAction;
typedef enum {
    PRESIDENT_OK, PRESIDENT_INVALID_ARGUMENT, PRESIDENT_ILLEGAL_ACTION,
    PRESIDENT_OUT_OF_MEMORY, PRESIDENT_UNFINISHED, PRESIDENT_ACTION_LIMIT,
    PRESIDENT_CLIENT_ERROR
} PresidentStatus;
typedef struct PresidentRound PresidentRound;
typedef struct PresidentGame PresidentGame;

const char *president_status_string(PresidentStatus status);
const char *president_rank_name(int rank);
const char *president_suit_name(int suit);
const char *president_role_name(int placement);
PresidentCard president_card(const char *rank, const char *suit); /* -1 invalid */
void president_standard_deck(PresidentCard deck[52]);

/* Create copies player names. roles is NULL exactly for first_round != 0;
 * otherwise roles[seat] must be a permutation of 1..4. deck is either NULL
 * with deck_count == 0 (shuffle using seed), or 52 unique cards. Output is
 * NULL on failure. The shuffle is reproducible but differs from Python's. */
PresidentStatus president_round_create(PresidentRound **out,
    const char *const players[4], int first_round, const int roles[4],
    const PresidentCard *deck, size_t deck_count, uint64_t seed);
void president_round_destroy(PresidentRound *round);
int president_round_done(const PresidentRound *round);
int president_round_actor(const PresidentRound *round); /* -1 when finished */
/* Return total action count, writing up to capacity actions. NULL/0 queries
 * the size. Generated combinations preserve Python's hand insertion order. */
size_t president_round_legal_actions(const PresidentRound *round, int player,
    PresidentAction *out, size_t capacity);
/* On success first_event receives the starting offset of the new events.
 * Rejected actions leave game state and logical histories unchanged. */
PresidentStatus president_round_apply(PresidentRound *round, int player,
    const PresidentAction *action, size_t *first_event);
size_t president_round_event_count(const PresidentRound *round);
size_t president_round_history_count(const PresidentRound *round);
PresidentStatus president_round_history_at(const PresidentRound *round,
    size_t index, int *player, PresidentAction *action);
PresidentStatus president_round_rankings(const PresidentRound *round, int out[4]);

/* Independent, heap-owned UTF-8 JSON snapshots matching PROTOCOL.md.
 * NULL means invalid input or allocation failure. Release with president_free.
 * events_json returns events from start through the current end. Private turn
 * history is deliberately excluded from all public JSON. */
char *president_round_message_json(const PresidentRound *round, int player);
char *president_round_events_json(const PresidentRound *round, size_t start);
void president_free(void *memory);

PresidentStatus president_game_create(PresidentGame **out,
    const char *const players[4], uint64_t seed);
void president_game_destroy(PresidentGame *game);
/* A successful start invalidates the previous round pointer. Failed starts
 * preserve the current round, round number, roles, and RNG state. */
PresidentStatus president_game_start_round(PresidentGame *game,
    const PresidentCard *deck, size_t deck_count);
PresidentRound *president_game_round(PresidentGame *game); /* borrowed */
size_t president_game_round_number(const PresidentGame *game);
PresidentStatus president_game_finish_round(PresidentGame *game, int out[4]);
typedef struct {
    void *context;
    void (*observe)(void *context, const char *message_json);
    PresidentStatus (*respond)(void *context, const char *message_json,
                               PresidentAction *action);
} PresidentClient;
/* Callbacks receive independent snapshots valid only during the callback.
 * Clients must not mutate/destroy/reenter the game. Notifications happen
 * initially and after each action, including the terminal action. */
PresidentStatus president_game_run(PresidentGame *game,
    const PresidentClient clients[4], size_t max_actions);

#ifdef __cplusplus
}
#endif
#endif
