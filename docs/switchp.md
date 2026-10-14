# SwitchPAgent port notes

This policy ports the shared `kou2` play routines and `keep_kou2` behavior from
the supplied `switch+` C client (2021 distribution). It inherits directly from
`ExperiencedAgent` for drafting, trading and protocol helpers. 

https://flute.u-shizuoka-ken.ac.jp/daihinmin/2022/download.html

## Source mapping

| Original source | President implementation |
| --- | --- |
| `daihinmin.c:value_strong` | `_strength`: start x at 100, subtract 30 per possible single response rank; grouped responses use the original 4/9/15/24 penalty table, or 30 heads-up. Minimum x is 1. |
| `daihinmin.c:pat_make`, `value_strong` | `_pattern`: group by ordinary rank; compute weakest/second-weakest x and z buckets. One remaining group uses the original second-minimum sentinel 201. |
| `daihinmin.c:pat_make`, `kou_lead` | `_lead_priority`: use original y formulas, same-width bonuses, and STRONG=86 branches. Near-finish hands lead strong groups first. |
| `daihinmin.c:kou_followgroup` | `_accept_follow`, `_play`: inspect legal responses in ascending rank order; apply the original ordered acceptance tests and adjusted z comparison. Preserve strong control candidates when the remainder has a near-finish shape. |
| `client.c:senjutu == 5` | `_keep`: initial z threshold 12; post-second-clear threshold 8; conserve K+ in weak hands; exempt strong A pairs/triples. |
| `client.c:strong`, `top_ranks` | `_predict_leader`, `observe`: predict after three clears using original clear/card-shedding points, then record prediction success from completed-round observations. |

Strength is an integer heuristic, not a probability. Hand z adds 2 for x=1–30,
1 for x=31–60, 0 for x=61–90, and subtracts 1 for x=91–200 (or the group width
when its rank is at least the highest unseen ordinary rank). The source omits
quads from this z sum; the port preserves that. Lower z is better.

Follow selection preserves the source's branch order: prefer strong plays with
a strong second-weakest remaining group; accept clears that do not add groups;
avoid spending sole control with four or more weak groups; reject weak responses
leaving a single weak group; accept near-finish shapes; preserve medium-strength
singles in expensive hands; otherwise reject worsening z. This is not a global
weighted action-score minimization.

## Necessary rule changes

- Our game has rank groups, no sequences, Joker, suit locks, revolution or 8-clears.
  Those source branches are omitted. Ordinary groups stay whole when leading
  where legal; quads are not split to avoid revolution, which does not exist here.
- Equal-rank replies count as responses. Opponent hand size excludes impossible
  widths. Out-of-turn completions and unseen 2-bombs each subtract 30 strength
  points when possible; these are explicit local-rule additions.
- A quad or matching play completing a stack gets x=200 as an immediate clear.
  Completion requests use the clear branch of the follow rules.
- 2s are separate bombs, excluded from ordinary rank grouping. Immediate ordinary
  finishes take precedence; bomb-cover guards prevent voluntarily leaving only
  2s. Bomb fallback uses near-finish structure or an ace board. There are no soft
  2-cost penalties or one-lead search from ExperiencedAgent2.
- Four-player counts replace five-player counts. The source group-penalty formula
  retains its constant 5; it uses the actual number of active players.
- The initial/post-second-clear hands are reconstructed from public events and
  our current hand. Prediction accuracy is learned only when a host supplies
  finished-round observations. `Game.run` does; the independent-round arena
  resets agents and does not provide final observations, so it cannot train
  this cross-round statistic. Early conservation still works there.

## Deliberate scope boundaries

This is the `kou2`/`keep_kou2` policy port, not all six policies in the tournament
wrapper. The 1,500/3,000-game strategy schedule, seat-order policy, fixed stage
variants, and dynamic `tes` adjustment variant are not ported. Neither the prior
SwitchPAgent's stage weights nor its probabilistic control bonus remain.
Trading remains the original ExperiencedAgent policy as requested.

Tests cover source-derived integer values and branch decisions, local-rule
interruptions, trade-method identity, prediction observation deduplication, and
multiple complete self-play/mixed-policy rounds. No relative-strength claim is
made without an arena benchmark.
