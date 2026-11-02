# ExperiencedAgent2

`ExperiencedAgent2` replaces the original play policy with a numerical action
evaluation. It reconstructs unseen rank counts by subtracting its hand and all
publicly played cards from four copies of each rank. Opponent hand sizes come
from the public event history.

For a candidate play, it uses a hypergeometric calculation to estimate each
opponent's chance of holding a response: a 2-bomb, a same-or-higher group of the
needed width, or enough matching cards for a completion. Multiplying opponents'
estimated failure-to-respond probabilities gives a control estimate.

It scores the remaining hand by charging each ordinary group, adding a singleton
cost, discounting groups likely to retain control, and penalizing too many 2s
for the remaining ordinary groups/cards. A non-clearing action also receives an
opponent-finish risk penalty. Clearing actions evaluate the best next lead after
the clear. The lowest action score wins.

Unlike `ExperiencedAgent`, trades are scored numerically: it simulates receiving
each requested rank and returning each possible card, then chooses the exchange
with the best resulting hand score plus a modest price for giving away strength.
The estimates assume unseen cards are uniformly distributed; passes do not
reveal private-hand information.

Probability, control, whole-hand, and next-lead calculations use bounded LRU caches (8,192 entries
each), shared across decisions. Control keys snapshot unseen counts and ordered
opponent hand sizes, so changed public information and hypothetical trades use
their own entries. Caching does not change the scoring formulas.
Hand cache keys also preserve rank iteration order to keep floating-point ties
unchanged. Trade returns evaluate only the first legal card of each rank, and
plays or completions that empty the hand return immediately without scoring
other actions. Bombs retain their special final-card penalty.

Trade scoring also has a decision-local lookup path for normal, physically
consistent snapshots. It precomputes the existing hypergeometric holding
probabilities by opponent size, unseen-copy count, and required width, then
rebuilds the original response and hand-cost expressions in their original
order. Candidate hands and unseen pools are mutable rank-count arrays, updated
and restored in place for each request/return pair; the original Counter order
is retained separately for floating-point behavior and ties. Holding values use
indexed tables, fixed group costs and reserve patience are computed once per
decision, and equal-size opponents reuse a response factor before multiplying
the factors in their original seat order. It does not use response deltas,
aggregate group totals, or reordered reserve arithmetic.

The direct evaluator remains the fallback for partial or inconsistent
snapshots, non-integer/out-of-range counts, failed requests with no remaining
candidate, and any subclass or instance replacement of trade-scoring helpers
(`_holding`, `_control`, `_hand_cost`, or `_trade_value`, including their cache
helpers). This keeps weight-only policy subclasses eligible while preserving
the behavior of custom scoring policies and test monkeypatches.
