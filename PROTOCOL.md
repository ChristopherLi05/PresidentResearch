# President Engine Protocol

## Purpose

This project exposes a JSON-shaped protocol for a four-player President game engine. The engine does not require a specific network transport: the same message format works for in-process clients, subprocesses, WebSockets, or another host.

A host owns the `Game` and `Round` objects. It gives the active player a private state snapshot, receives one action, applies it, and broadcasts the newly generated public events.

## Protocol directions

| Direction | Object | Meaning |
|---|---|---|
| Engine to a player | `state` | Player-specific snapshot, including that player's hand and an optional request. |
| Player to engine | `action` | A single object selected from `request.legal_actions`. |
| Engine to observers | `event` | A public event newly created by the action. |

## Shared JSON values

### Card

```json
{ "rank": "A", "suit": "S" }
```

- Ranks, lowest to highest: `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `J`, `Q`, `K`, `A`, `2`.
- Suits: `C`, `D`, `H`, `S`.
- A `2` is a bomb. Suits identify cards but do not affect normal play legality.

### Roles and placements

| Value | Role |
|---:|---|
| 1 | President |
| 2 | Vice President |
| 3 | Vice Scum |
| 4 | Scum |

## State snapshot JSON

The host calls `Round.message_for(player)`. It returns an independent snapshot, with private fields scoped to that player.

```json
{
  "type": "state",
  "player": "a",
  "phase": "play",
  "hand": [{ "rank": "Q", "suit": "S" }],
  "turn": "b",
  "top": { "rank": "Q", "count": 2, "stack_count": 2 },
  "active_players": ["a", "b", "c"],
  "events": [],
  "request": {
    "type": "play",
    "legal_actions": []
  },
  "rankings": null
}
```

| Field | Meaning |
|---|---|
| `phase` | `draft`, `trade_request`, `trade_return`, `play`, `completion`, or `finished`. |
| `hand` | Private, sorted hand of the named player. |
| `turn` | Player who is currently expected to respond; `null` when finished. |
| `top` | `null` on a cleared board. Otherwise, the current rank, required counter-play count, and contiguous matching `stack_count`. |
| `active_players` | Players still participating in the round. |
| `events` | Complete public event history. |
| `request` | Present only when the named player is the current actor; otherwise `null`. |
| `rankings` | `null` until the round has finished; then a player-to-placement map. |

## Action JSON

Only the player whose state contains a non-null `request` may respond. A client should return one object from `request.legal_actions`.

| Phase | Action | Meaning |
|---|---|---|
| `draft` | `{"type":"choose_pile","pile":0}` | Claim an available pile, indexed 0 through 3. |
| `trade_request` | `{"type":"trade_request","rank":"A"}` | Request one rank from the paired lower role. |
| `trade_return` | `{"type":"trade_return","card":CARD}` | Return one card after a successful trade request. |
| `play` | `{"type":"play","cards":[CARD,...]}` | Play one or more equal-rank, non-2 cards. |
| `play` | `{"type":"pass"}` | Pass on a non-cleared board. |
| `play` | `{"type":"bomb","card":CARD}` | Play exactly one 2 on a non-cleared board. |
| `completion` | `{"type":"complete","cards":[CARD,...]}` | Supply the cards needed to create a four-of-a-kind at the top. |
| `completion` | `{"type":"decline_completion"}` | Decline only the current completion window. |

The engine rejects an action if:

- the round is already finished;
- the sender is not the current actor;
- the action is not an object;
- the action is malformed; or
- the action is illegal in the current state.

## Public event JSON

`Round.apply_action(player, action)` returns only the new public events created by that action. The next state snapshot retains the full history in `events`.

| Event type | Fields beyond `type` | Meaning |
|---|---|---|
| `round_started` | `first_round`; either `first_player`, or `face_up` and `draft_order` | A round begins. |
| `pile_chosen` | `player`, `pile` | A player drafts a pile. |
| `draft_complete` | none | All piles have been claimed. |
| `trade_requested` | `initiator`, `receiver`, `rank`, `success` | A trade rank is requested. Rank and success are public. |
| `trade_completed` | `initiator`, `receiver` | A successful exchange completes; the returned card remains private. |
| `trades_complete` | `first_player` | Later-round trades end and play begins. |
| `opening_play` | `player`, `cards` | The first-round opening play. |
| `played` | `player`, `cards` | A normal non-bomb play. |
| `passed` | `player` | A player passes. |
| `bombed` | `player`, `card` | A single 2 is played. |
| `completed` | `player`, `cards` | A player completes four of a kind. |
| `board_cleared` | `player` | The pile clears and this player gets the resulting turn. |
| `player_done` | `player`, `bomb_finish`; optionally `last_active` | A player is assigned a finishing position. |
| `twos_announced` | `player` | A player has only 2s and is removed with bomb-finish ordering. |
| `round_finished` | `rankings` | All final placements have been determined. |

## Handshake protocol

The host repeats this request-response cycle until the round finishes:

1. Create a game and call `game.start_round()`.
2. Identify the current actor.
3. Call `round.message_for(actor)`.
4. Send that state snapshot only to the actor.
5. The actor returns one action from `state.request.legal_actions`.
6. Call `round.apply_action(actor, action)`.
7. Broadcast the returned public events, then repeat at step 2.
8. When the round emits `round_finished`, use its `rankings`. Calling `start_round()` later carries those rankings forward as roles.

Minimal host loop:

```python
round_ = game.start_round()

while not round_.done:
    player = round_._actor()  # host-side implementation detail
    state = round_.message_for(player)
    action = clients[player].respond(state)
    new_events = round_.apply_action(player, action)
    broadcast(new_events)

rankings = game.finish_round()
```

## Phase lifecycle

### First round

```text
play -> completion windows as needed -> finished
```

- Cards are dealt directly to players.
- The holder of the 3 of clubs starts.
- The first play must contain rank 3.
- The board starts cleared.

### Later rounds

```text
draft -> trade_request -> trade_return -> play -> completion windows as needed -> finished
```

- Four 13-card piles are created, each with a public face-up card.
- Draft order is President, Vice President, Vice Scum, Scum.
- Trades are: President with Scum twice, then Vice President with Vice Scum once.
- A failed trade request produces `trade_requested` with `success: false`; the initiator remains in `trade_request` and can ask for another rank.
- After a successful request, the initiator alone receives a `trade_return` request.
- When trades end, Scum begins play.

## Play rules reflected by the protocol

- On a cleared board, a player may play one or more cards of one non-2 rank.
- On a non-cleared board, a normal play must use the same number of cards as the top play and a rank at least as high.
- Passing and bombs are illegal on a cleared board.
- A bomb is exactly one 2 and immediately clears the board to the bomber.
- Equal-rank counterplay creates a pending skip. Completion processing occurs before the skip takes effect.
- When play returns to the last player to play cards, the board clears to that player.
- A player who empties their hand is removed from active play. Finishing on a bomb, or having only 2s remaining, places the player after normal finishers.

## Completion window handshake

After every pass and every non-terminal play that does not immediately clear the board, the engine opens a completion window before ordinary play advances.

1. The player who just acted is polled first if still active.
2. The remaining active players are polled counterclockwise.
3. Every polled player receives `decline_completion`, even if completing is impossible.
4. A player able to make the top stack into four of a kind also receives one or more `complete` actions.
5. A `complete` immediately clears the board to that player.
6. If all players decline, the pending normal advance, skip, or clear-to-last-player takes effect.

A decline applies only to that individual window; it does not prevent a completion in a later window.

## Integration notes

- `Game.run(...)` is an in-process convenience adapter; it uses the same state/action contract.
- A network adapter should preserve private delivery of `hand` and `request`.
- The public `events` stream is sufficient for clients such as `BasicClient` to infer failed trade ranks without retaining private state.
- Treat the state and event objects as snapshots: the engine returns independent copies, so mutation by a client does not change engine state.

