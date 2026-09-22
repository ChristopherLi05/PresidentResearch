President (also called Scum) is a trick-based shedding card game much like the Japanese Daifugo. At its most basic,
President is played with a standard 52 card deck with anywhere from 3-6 players. In this report, we will focus on the
4-player variant with some local rules. The rules are as follows:

- The rank of the cards are 3 (lowest) up to A (highest), and 2 (bomb)
- The roles are named President (1), Vice President (2), Vice Scum (3), and Scum (4) depending on a player’s rank.
  Players who deplete their hands earlier will earn a higher rank.
- Card dealings:
    - First round: cards are dealt to each player
    - Later rounds: cards are dealt into 4 piles, with the last card facing up. The President will choose first,
      followed by Vice President, Vice Scum, and Scum.
- The cards they have are considered their “hand”
- First to play:
    - First round: the player with the 3 of clubs will start the game. The opening play must contain a 3; suits have no gameplay effect.
    - Later rounds: the scum will start the game.
    - The board is considered “cleared” at the start of the game.
    - The player who plays on a cleared board can play any number of cards of the same rank other than a 2 (bomb)
- In a counterclockwise order, players may play an equal number of matching cards with an equal or higher rank as the
  last card(s) played.
    - Players are allowed to pass even if they have a possible play.
    - Players are allowed to play even if they pass.
    - If a player plays a card with equal value to the previous card played, then the next player’s turn is skipped.
    - If play returns to the player who most recently played cards, then the board is considered cleared, and it is
      their turn. This is considering “clearing the board to them”
- If at any time, a player has cards that when added will make 4 of a kind at the top of the pile, then they may play
  them even if it isn’t their turn. This will clear the board to them. This is considered a “completion”
    - Completions take priority over skipping.
    - Players are allowed to complete themselves (e.g. if they have a 4 of a kind).
    - A completion will occur before the next player’s action.
- A player may choose to play a single 2 (bomb) during their turn no matter how many card(s) were played. This will
  clear the board to them.
- If a player empties their hand, then they are done with the round. Earlier players who are done will place higher.
    - If a player who is done is entitled to a cleared board, then the next player to play will inherit it.
    - Done players are not considered for the next player to play
- A player cannot end on a 2 (bomb). If this happens, then they are automatically last.
    - Completions are ok to end on.
    - If there are multiple players who fulfill this condition, then earlier players will rank higher.
    - If a player only has 2’s remaining in their hand, then they must announce this fact. They are considered done, and
      the remaining 2’s are not played. They receive bottom placements under the same penalty, with earlier announcements
      ranking higher than later announcements.
- When only one active player remains, the round ends immediately and that player receives the lowest remaining rank.
- At the start of every round after dealing, the President will trade 2 times with the Scum, and the Vice President will
  trade 1 time with the Vice Scum.
    - In a trade, the initiator (President/Vice President) may request any card rank, and the receiver (Scum/Vice Scum)
      must provide the card if present in their hand. The initiator will return a card of their choosing to the
      receiver. If the trade fails, the initiator may request a different card.
    - The requests that the President and Vice President asked are considered public knowledge. The success or failure
      of the trade is public knowledge. The cards they return are not public.
    - Since there are no roles in the first round, this rule does not apply in the first round.

Engine protocol and tests:

- `Round.message_for(player)` returns an independent JSON-shaped state snapshot and, when it is that player's turn
  to respond, a private request containing their legal actions. `Round.apply_action(player, action)` returns independent
  copies of newly broadcast public events.
- After each play that does not immediately clear or end the round, and after each pass, the engine opens a completion
  window before advancing play. Every active player is prompted in a fixed order: the player who just acted, if still
  active, followed by the other active players counterclockwise. Players can `decline_completion`; their private legal
  actions also include `complete` if their cards permit it. Being prompted does not reveal whether someone can complete.
  A decline applies only to that window, so a player can complete after a later pass. If everyone declines, the pending
  turn advance, skip, or board clear takes effect.
- `Game.run(..., max_actions=...)` counts all responses, including completion declines, toward the action limit.
- `Game.start_round()` rejects starting another round while the current one is unfinished. Once the current round
  finishes, starting the next round automatically carries forward its rankings as roles. `Game.finish_round()` remains
  available to retrieve and save the completed round's rankings explicitly.
- Run the dependency-free test suite from the repository root with `python -m unittest -v` (or
  `python -m unittest discover -s tests -v`). The suite includes deterministic multi-round simulations checking card
  conservation, independently checked gameplay legality and turn/board transitions, terminal rankings, and penalty ordering.
