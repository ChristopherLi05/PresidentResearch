# ExperiencedAgent4

Agent4 adds turn-order danger and shallow Monte Carlo action evaluation to
ExperiencedAgent3Tuned. The Ninja-inspired parts are shared sampled worlds,
heuristic-guided continuations, and separation of tactical/exact decisions
from statistical evaluation. This is root action evaluation, not a full MCTS
tree or Ninja's outer choice bandit.

## Controls

```python
from src.agent import ExperiencedAgent4
import random

agent = ExperiencedAgent4()  # 50 ms cooperative budget, at most 8 worlds
baseline = ExperiencedAgent4(search_worlds=0)  # same heuristic, no sampling
repeatable = ExperiencedAgent4(random.Random(42), search_time_ms=None,
                              search_worlds=8)  # deterministic fixed work
python_reference = ExperiencedAgent4(search_backend="python")
native_required = ExperiencedAgent4(search_backend="c")
```

### Shared hand evaluation for earlier agents

`ExperiencedAgent2`, `ExperiencedAgent3`, `ExperiencedAgent3Tuned`, and Agent4
also use the native hand/lead evaluator on cache misses by default. Select it
independently of Monte Carlo search:

```python
from src.agent import ExperiencedAgent3Tuned
agent = ExperiencedAgent3Tuned(evaluation_backend="auto")
reference = ExperiencedAgent3Tuned(evaluation_backend="python")
required = ExperiencedAgent3Tuned(evaluation_backend="c")
```

The existing bounded Python caches remain in place. Each agent supplies its
own weights, Counter iteration order, unseen rank counts, and opponent sizes;
no sampled world or full engine state is required. Unsupported custom helper
overrides and missing/stale native libraries fall back to Python in auto mode.
Explicit C mode reports them instead. Use a new agent when switching backends
so its existing cached results do not mask the selection. Library availability
is resolved once per process for hand evaluation; restart after a rebuild.

BasicAgent, BasicAgentPlus, the original ExperiencedAgent, and SwitchPAgent
have different policies and do not use this hand-cost formula. The separate
optimized trade lookup path remains Python; direct trade scoring can reuse
the native hand evaluator. No search is added to earlier agents.

On 500 varied cache misses for Agent3Tuned, local timings were 48.4 ms Python
versus 10.4 ms C for hand costs, and 255.0 ms versus 11.3 ms for lead costs
(about 4.6x and 22.5x respectively). Cache hits still return directly in
Python; these are evaluator microbenchmarks, not whole-game speedups.
Differential tests cover shuffled rank orders and all four agents' weights.
For an entirely Python Agent4 reference, set both `search_backend="python"`
and `evaluation_backend="python"`.

Build the optional native backend once with
`python -m src.agent_search_c --build` (GCC or Clang required). The default
`search_backend="auto"` uses the compiled library when available and current,
otherwise Python. Explicit `"c"` reports missing/stale library or unsupported
policy overrides instead of silently falling back. Rebuild after native-source
changes, in a fresh process if Windows has the DLL loaded. No compilation or
dependency installation occurs during an agent decision.

The native implementation is `src/agent_search_native.inc`, compiled into a
separate search library through the optional `PRESIDENT_SEARCH` engine build.
It reuses the engine's private state and authoritative transitions without
changing the ordinary engine ABI or duplicating game rules. All simulated
policy scoring, hand/lead evaluation, and transitions stay in C, using fixed
scratch buffers and decision-local probability tables. Python retains public
history reconstruction, root candidate ranking, sampling, and batch selection.
Weights are passed at runtime; custom Python scoring-method overrides use the
Python backend. The two-player exact solver has its own optional C library,
built with `python -m src.solver_c --build`, and is selected independently.

The inherited exact two-player solver runs before this search and is not
limited by its deadline. Drafting and trading are unchanged. Immediate normal
finishes and forced actions skip sampling. `last_search` reports completed
worlds, candidate count, elapsed milliseconds, fallback status, backend, and reason
for the most recent heuristic/search decision; it is not refreshed by the
inherited exact solver or trade/draft paths.

## Decisions and simulated information

Suit-equivalent plays collapse to the first supplied legal representative.
The four best heuristic actions plus legal pass/decline and bomb options are
evaluated. Each sampled allocation is shared across every candidate. Only
complete batches contribute; timeout before the first batch returns the
original heuristic choice. Final ties retain heuristic order.

Sampling is uniform over unseen cards with public hand sizes. It does not
infer holdings from voluntary passes or implement history-weighted trade or
draft inference. Missing/inconsistent histories and previously announced
unplayed 2s use the heuristic fallback. No host-private hands are inspected.

Engine transitions resolve completion windows, skips, clears, and
penalties. Continuations use Agent4's heuristic from each acting player's
private hand plus public history, without recursive search or exact solving.
They stop at our next normal turn, our exit, round end, or 16 normal actions;
pending completion windows are resolved before the depth cutoff. A bomb can
therefore stop immediately before our required follow-up lead.

The leaf evaluator averages pairwise finishing preferences against the
other three seats. Known finishing order and bomb penalties override hand
costs. Otherwise it uses `1 / (1 + exp(our_cost - opponent_cost))`, with the
existing hand cost, or lead cost when the player owns an empty board.
The evaluator is heuristic, not a calibrated placement probability.

## Validation and measured limits

Native parity and timing checks:

```text
python -m unittest tests.test_agent_search_c tests.test_agent_search tests.test_experienced_agent2 tests.test_solver -q
python benchmark_agent_search.py --compare-backends --trials 0
```

On the local Windows/Python 3.13 host, the same 75 positions at eight fixed
worlds took 3,298 ms total in Python versus 126 ms in C: approximately **26x**
faster including Python setup and sampling. Median decision time was 30.95 ms
versus 1.48 ms; p95 was 122.23 ms versus 2.77 ms. All 75 decisions matched.
These fixed-work comparisons disable the deadline, so Python p95 can exceed
50 ms. With the normal deadline, native p95 was about 2–4 ms across game stages.
The eight-world default remains unchanged; increase `search_worlds` explicitly
to spend the saved time on more samples. No playing-strength gain is claimed
from the native port itself.

Differential tests compare scores, complete rollout states, and leaf values
against Python, including completion, skip, bomb and finished-player cases.
Compensated sums match modern Python's float summation to preserve tied move
choices on the tested host. Platform/library rounding and Python version can
still affect close ties. The parity work also fixed duplicate completion
respondents in public-state reconstruction (including the exact-solver path).

Run `python -m unittest tests.test_agent_search tests.test_experienced_agent2
tests.test_solver -q` for focused correctness checks (put the command on one
line). Run `python benchmark_agent_search.py --trials 5` for snapshot timing
and two balanced 30-game comparisons. Increase trials for strength research.

Before the native port, a local sample of 25 positions per stage measured median search
times of approximately 50 ms early, 33 ms midgame, and 13 ms late, with p95
around 50–51 ms. Average completed worlds were 5.0, 7.24, and 6.56. This is a
cooperative deadline checked between operations; one scoring/evaluation call
can overrun it. Hardware, cache state, and scheduling affect timings and
decisions. Fixed-work mode removes deadline-dependent sampling differences.

The initial 30-game comparisons produced mean placement differences
(Agent4 search minus opponent, negative is better) of -0.167 against Agent4
without search (95% interval -0.656 to 0.322), and +0.333 against Agent3Tuned
(-0.080 to 0.747). These smoke tests do not establish a strength improvement.
Both comparisons used seed 42 and all six two-versus-two seat configurations.

Invalid sampled worlds (including active players holding only 2s) are discarded
before simulation. A native position rejection also discards the entire batch
and retries a fresh world, rather than aborting the match. Each decision allows
at most `search_worlds + 3` total attempts under its original deadline. Only
complete successful batches contribute; if none succeed, the legal heuristic
choice is returned. `last_search` exposes `attempts`, `rejected_worlds`, and
`native_errors` so repeated failures remain visible. Configuration and unrelated
programming errors are not swallowed.
