# ExperiencedAgent3

`ExperiencedAgent3` uses `ExperiencedAgent2` for drafting, trading, and its
base action scores. Its hand-cost heuristic gives a small additional discount
to reliable control groups when they support other, vulnerable groups. It uses
the existing response probabilities, with diminishing returns for additional
control, rather than K/A rules or Switch+ pattern thresholds. A lone group
receives no support discount. The adjustment is bounded below 1.5 cost points
and naturally affects plays, next-lead evaluation, and trades through their
remaining-hand evaluations. Immediate finishes and the exact solver retain
priority. The weight is heuristic and has not been benchmark-calibrated.

Conservation also adapts continuously to the public game state. The support
discount is multiplied by opponents' average hand size divided by 13, then by
`16*m / (13*(m+3))`, where `m` is the smallest opponent hand size. Sizes are
clamped to 0..13 for temporary trade hands. Thus the full discount applies
against initial hands, fades as cards are shed, and fades faster when any
opponent is close to finishing. Existing immediate-finish risk scoring still
applies. This borrows Switch+'s state-dependent conservation idea without its
board-clear stage thresholds or learned leader predictor.

When exactly two players remain, it reconstructs the
other active hand from the full deck: cards are either in its private hand,
already public through play, or in that opponent's hand. It then asks
`src.solver.solve_round` for the exact minimax action.

The policy also works with compact request messages that omit `turn`: the
request recipient is necessarily the current actor.

If a caller supplies an incomplete or malformed state snapshot, it safely
falls back to the `ExperiencedAgent2` heuristic. It also falls back after a
previous player announces unplayed 2s, because the protocol does not disclose
how many such cards were removed from that player's hand.
