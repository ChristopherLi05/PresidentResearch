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

## Native exact solver

Build once with `python -m src.solver_c --build` (GCC or Clang), then start a
new arena/comparison process. Agent3, Agent3Tuned, and Agent4 automatically use
the native solver when the library is present and current. There is no runtime
compilation. A running comparison will keep its already-loaded backend.

`solve_round(round_, backend="python")` selects the reference implementation;
`backend="c"` requires the library and reports absence instead of falling back.
The default `"auto"` falls back to Python when native code is unavailable.
Library availability is cached per process; restart after rebuilding.

The C implementation in `src/solver_native.c` uses rank-count hands, exact
minimax, and a growing hash table owned by each call. It preserves Python move
ordering and returns the same optimal action, win/loss outcome, state count,
and cache-hit count. Python still reconstructs the two-player position and
maps the selected rank move back to the original legal action with suits.
There is no depth/time cutoff, heuristic approximation, or shared mutable
search state. Allocation failure raises an error rather than inventing a
search result. The native solver uses a separate DLL from the C game engine
and hand evaluator.

Run `python -m unittest tests.test_solver tests.test_solver_c -q` for engine
oracle and native/reference comparisons, including all active seat pairs,
completion windows, inherited boards, bomb penalties, table growth, and
concurrent isolated calls. Build the native library before running its tests.

Benchmark with `python benchmark_solver.py --trials 40 --positions 300 --repeats 3`.
This samples actual solver calls from Agent3Tuned versus BasicAgent on balanced
seats and identical deals, checks complete result equality, then alternates
both backends through the same C-engine arena workload. Hand evaluation stays
native in both runs. Complete-game records must match exactly. Solver timings
include Python position conversion and legal-action mapping; they are not
just timings of the inner C recursion.

A local Windows/Python 3.13 run of that command sampled 300 actual endgames
(up to 36,938 states). Aggregate solver time was 1,252.3 ms in Python versus
62.6 ms in C: **20.0x faster**, with zero result mismatches. Median call latency
was 0.093 versus 0.044 ms; the aggregate gain is dominated by harder positions.
Across three repetitions of 240 full games, median elapsed time was 10.476 s
versus 9.182 s (about **14% higher throughput**), with identical game records.
Host load varied substantially, so the whole-game improvement is indicative,
not a guaranteed rate. Trade scoring, copying, and normal policy work remain
outside this exact-solver speedup.

The policy also works with compact request messages that omit `turn`: the
request recipient is necessarily the current actor.

If a caller supplies an incomplete or malformed state snapshot, it safely
falls back to the `ExperiencedAgent2` heuristic. It also falls back after a
previous player announces unplayed 2s, because the protocol does not disclose
how many such cards were removed from that player's hand.
