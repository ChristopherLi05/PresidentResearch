# Search sampling in ExperiencedAgent4

ExperiencedAgent4 picks some of its moves by Monte Carlo search. It cannot see
the other hands, so it imagines several complete deals ("worlds") that agree
with what it knows. It plays each candidate move forward in every world and
keeps the move that does best on average. This writeup covers how those worlds
are dealt. That step changed from a uniform shuffle to deals weighted by an
opponent model built from the table-side rules of `HumanRulesAgent3`.

Code: `src/agent_search.py` (search loop and uniform sampler),
`src/agent_belief.py` (opponent model and sampler), `src/agent_belief_native.inc`
(the walk in C). Tests:
`tests/test_agent_belief.py`. Diagnostic: `belief_diagnostic.py`.

## 1. The search loop

For each normal play or completion decision, `agent_search.choose`:

1. **Skips trivial cases.** A forced action or an action that empties our hand
   is returned directly. With exactly two players left, the exact solver
   inherited from ExperiencedAgent3 runs instead and the search is not used.
2. **Picks candidates.** It takes the heuristic's top `search_candidates`
   actions (default 4) and always adds legal pass, decline and bomb options.
   Candidate 0 is the heuristic's own choice.
3. **Rebuilds the public position** (`prepare`). From our hand and the public
   event history it rebuilds an engine `Round`. It works out which cards are
   unseen (not in our hand and not yet played) and each opponent's hand size.
   Histories it cannot account for, such as announced 2s, fall back to the
   heuristic.
4. **Samples a world and scores every candidate in it.** Each world is shared
   by all candidates, so they are compared on the same hidden cards. The rest
   of the game is played with Agent4's heuristic for every seat, in Python or
   in the native C library. The result is scored as the share of the other
   three seats we beat.
5. **Picks the move.** After `search_worlds` complete worlds (at most three
   rejected samples are replaced), the best average wins. With
   `search_margin > 0`, it overrides the heuristic only if a paired test shows
   the gain is that many standard errors above zero.

Only step 4's *sampling* changed. The rollouts, scoring, native code interface
and move selection are untouched.

## 2. The old sampler: uniform

`sample_world` shuffles the unseen cards and deals each active opponent as many
as they hold. Every deal that matches the public hand sizes is equally likely.

This ignores almost everything the history says. An opponent who passed on a
single 5 while holding a 2 is a rare event under the rules, yet uniform worlds
give them a 2 as often as any other card. Worlds dealt this way have opponents
hold cards that their own earlier play makes unlikely. The search then plans
against those hands.

## 3. The new sampler: modeled

`search_sampling="modeled"` treats dealing as inference. Each candidate deal is
weighted by how well it explains what the opponents did. Deals are then drawn
in proportion to that weight.

### 3.1 The opponent model

The model is `HumanRulesAgent3`: the short rule list in
[experiencedagent3-human.md](interpretation/experiencedagent3-human.md),
distilled from ExperiencedAgent3Tuned. `agent_belief.py` contains a rank-level
port of it:

- `rules_play`: rules 0, L1–L3 and F1–F3, given rank counts, the board, and
  the smallest opponent's hand size;
- `rules_trade`: T1/T2/T2b requests;
- `rules_completes`: C1.

Suits never affect play, so rank counts are enough. That makes the port fast.
It is tested to make exactly the same choice as the full agent on 25,000
random positions.

### 3.2 Turning history into evidence

The key trick is that played cards are public. An opponent's hand at any
earlier moment is their current hand plus every card they have played since.
So each past decision becomes a check on the *current* hand being sampled:

```
counts_then = counts_now + offset        # offset: cards they played since then
```

`extract()` walks the public events once per decision and records, for each
active opponent, a list of `(offset, kind, data)` observations:

| Evidence | What it tells us | Weight if violated |
| --- | --- | --- |
| Drafted a pile with a face-up card | The drafter holds that card. It is pinned to them unless a trade could have moved it (they were a trade initiator, or later gave that rank away). In that case only "held that rank before trading" is checked. | pinned (hard) or 0.02 |
| Our own returned card | The receiver holds it. Pinned unless they later gave that rank back. Agent4 remembers the card from its own `trade_return` decision. | pinned |
| Failed trade request | The receiver had none of that rank then. A later returned card could be that rank, so a count up to the number of later returns is only mildly penalised. | 0.3, else 0.02 |
| Successful trade request | The initiator holds that rank after the exchange. | 0.02 |
| Rank requested | Should match T1/T2/T2b on the approximate pre-trade hand. | `noise` |
| Every lead, follow, pass and bomb | Should match `rules_play` given the board and the smallest opponent at that moment. | `noise` |
| Completion window closed without a completion | Each active opponent either could not complete or C1 said decline. | `noise` |
| Active hand of only 2s | Impossible: it would have been announced. | 1e-9 |

Trades use the same trick with trade offsets. "Before exchange k" means current
hand + cards played + ranks given away in exchanges ≥ k − ranks received in
exchanges ≥ k. Returned cards are not public, so the pre-trade hand is
approximate. That is why those checks are soft.

Decisions made with only two players left are skipped, since those players use
the exact solver rather than the rules.

### 3.3 Likelihood

A deal's weight is the product, over opponents and observations, of 1 for a
match or the factor above for a mismatch. In log form this is a sum of
penalties (`log_likelihood`). `search_model_noise` (default 0.2) says how much
we trust the rules. At 1.0 the model says nothing and sampling is uniform
again. Mismatches are not ruled out, because real opponents deviate from the
rules (Agent3Tuned agrees with them 71–99% of the time, depending on the rule).

This is a noisy-rational opponent model. It is not an exact posterior: the
rules are a heuristic description, and the trade checks are approximate.

### 3.4 Drawing deals: a Metropolis chain

Most uniform deals have tiny weight, so drawing many and resampling wastes
work. Instead, `BeliefSampler` runs one Markov chain per decision:

1. **Start.** Place pinned cards with their owners (dropped for a player if
   the pins exceed their hand size). Shuffle the remaining unseen cards into
   the open slots.
2. **Propose.** Pick two opponents and swap one unpinned card between them.
   Hand sizes and pins stay valid automatically. Swapping two cards of the
   same rank changes nothing and is skipped.
3. **Accept** with probability `min(1, exp(Δ))`, where Δ is the change in
   the two affected players' log weight. Each player's weight depends only on
   their rank counts, so it is cached by `(player, counts)`. Rejected proposals
   and revisited hands cost nothing.
4. **Sample.** After a warm-up (burn-in) of steps, take a snapshot every
   so often; each snapshot is a world. Both counts scale with the number of
   free cards: 40 and 0.5 per card in C, 10 and 0.25 per card in the Python
   fallback (see §3.6).

### 3.6 The native walk and step counts

The walk normally runs in C (`src/agent_belief_native.inc`, compiled into the
search library). Python still reads the history into evidence, about 0.16 ms.
It packs the evidence into a flat `BeliefObservation` array and calls
`president_belief_walk(..., steps)`, which swaps cards in place in a
card-code array. The first call runs the warm-up and each later world runs one
snapshot gap. The C scorer is the same rank-level rules port.
`president_belief_loglik` exposes it so the tests can check every score
against Python's.

The C walk uses its own splitmix64 generator, seeded from the agent's RNG. Its
worlds are reproducible but differ deal for deal from the Python walk's. With
`search_backend="python"` (or no compiled library) the Python walk runs.

`belief_diagnostic.py --burn ... --thin ...` sweeps the step counts on the same
decisions (928 decisions, 48 worlds each). L1 is the error from §5.

| Walk | Warm-up / card | Gap / card | Steps | ms per decision | L1 |
| --- | --- | --- | --- | --- | --- |
| Python | 20 | 0.5 | 846 | 25.2 | 4.371 |
| Python | 10 | 0.25 | 423 | 14.6 | 4.418 |
| Python | 2 | 0.1 | 135 | 5.9 | 4.661 |
| C | 20 | 0.5 | 846 | 1.2 | 4.373 |
| C | **40** | **0.5** | 1,236 | **1.5** | **4.297** |
| C | 80 | 2 | 3,389 | 2.2 | 4.308 |

Accuracy flattens out beyond about 1,000 steps. In C the steps are nearly
free; what remains is Python setup and building the worlds. Hence the two
defaults: 40/0.5 in C, and 10/0.25 in Python, where steps are expensive.

### 3.5 Keeping some worlds random

`search_random_fraction` (default 0.25) of worlds are still dealt uniformly.
They are spread evenly through the batch (attempts 3, 7, 11, ...) rather than
by coin flip, so every decision gets its share. This keeps the search sound
when an opponent does not play like the rules, or when the model is wrong
about a particular hand.

If the evidence extractor cannot read a history, the whole decision falls back
to uniform (`last_search["sampling"] == "uniform_fallback"`).

## 4. Controls

| Keyword | Default | Meaning |
| --- | --- | --- |
| `search_sampling` | `"uniform"` | `"uniform"`: the original sampler, bit-identical decisions. `"modeled"`: the sampler above. |
| `search_random_fraction` | `0.25` | Share of uniform worlds in modeled mode, in `[0, 1]`. |
| `search_model_noise` | `0.2` | Weight of a rules mismatch, in `(0, 1]`. |

`last_search` reports `sampling`, `modeled_worlds`, `uniform_worlds`,
`belief_ms` and `belief_acceptance`. The tuned configuration is pinned as
`search_variants:ExperiencedAgent4Full64M2Modeled`. In `tune_search.py`, add
`search_sampling=modeled` to a variant to compare it with the same deals
dealt uniformly.

## 5. Validation

**The model matches its source.** `rules_play` and `rules_trade` agree with
`HumanRulesAgent3` on every one of 20,000 play and 5,000 trade positions.

**The evidence is correct.** At a table of four `HumanRulesAgent3` players,
the true hands should satisfy the evidence. Across 150 games:

| Observation | Violated by true hands |
| --- | --- |
| play / pass / bomb | 0 of 13,303 |
| no completion | 0 of 28,693 |
| draft and returned-card pins | 0 of 5,029 |
| trade request choice | 129 of 4,427 (approximate pre-trade hands) |
| successful request | 45 of 4,253 |
| failed request | 136 of 2,065 before the return allowance (§3.2) was added |

This check found two bugs. The engine names the first lead of *every* round
`opening_play`, and later-round leads were wrongly read as "lead all your 3s".
And the failed-request check had ignored later returns.

**The guesses are closer to the truth.** `belief_diagnostic.py` deals 32
worlds both ways at each Agent4 decision and compares them with the real hands.
L1 is the summed per-rank count error per opponent; lower is better. Results
against ExperiencedAgent3Tuned (12 games, 364 decisions):

| | Uniform | Modeled |
| --- | --- | --- |
| L1 error, all decisions | 5.87 | 4.04 |
| L1 error, early (< 40 events) | 9.04 | 7.08 |
| L1 error, midgame | 3.36 | 1.64 |
| (opponent, rank) cells exactly right | 65% | 77% |

The result held for noise from 0.1 to 0.35, and against `HumanRulesAgent3`
opponents (6.52 → 4.53).

## 6. Results

These are `tune_search.py` runs on identical deals across all six 2v2
seatings. The search was the `full64m2` configuration: 64 worlds, rollouts to
the end of the round, override margin 2, danger weights 0, no deadline.
Differences are paired per game (modeled minus uniform); lower placement is
better.

| Opponents | Sampling | Mean placement | Difference (95% CI) |
| --- | --- | --- | --- |
| ExperiencedAgent3Tuned, 9,000 games | uniform | 2.311 | |
| | modeled, 25% random | 2.193 | **−0.117** (−0.131 to −0.103) |
| | modeled, 0% random | 2.183 | −0.127 (−0.141 to −0.113) |
| | modeled, 50% random | 2.209 | −0.102 (−0.115 to −0.088) |
| SwitchPAgent, 6,000 games | uniform | 2.179 | |
| | modeled, 25% random | 2.155 | **−0.024** (−0.041 to −0.008) |
| | modeled, 0% random | 2.162 | −0.017 (−0.034 to +0.000) |

- The gain is large against the agent the rules describe and smaller, but
  still significant, against one they do not. That is the expected pattern
  for a model of opponent behaviour.
- Dropping the random worlds helps slightly against Agent3Tuned and hurts
  slightly against SwitchP. Neither difference is significant. 25% is the
  robust default.
- Raw data: `results/search_tuning/belief_ab.json` and
  `results/search_tuning/belief_ab_switchp.json`.

### With the C walk

The same comparison was rerun after the port (seed 2029, 6,000 games each,
against ExperiencedAgent3Tuned). It also tested the shipped Agent4 default:
8 worlds under the 50 ms deadline.

| Configuration | Uniform | Modeled | Difference (95% CI) | ms per decision |
| --- | --- | --- | --- | --- |
| full64m2, no deadline | 2.299 | 2.200 | **-0.099** (-0.116 to -0.082) | 24.2 -> 25.0 |
| 8 worlds, 50 ms deadline | 2.615 | 2.565 | **-0.050** (-0.070 to -0.030) | 5.2 -> 7.0 |

Modeling now costs under 1 ms per decision in the strong configuration, down
from about 59 ms with the Python walk. It also helps the shipped default under
its real deadline: the budget ran out in only 3 of about 100,000 decisions.
Raw data: `results/search_tuning/belief_native_ab.json`.

## 7. Limitations and next steps

- **Cost.** Solved by the C walk (§3.6): sampling costs about 1.5 ms per
  decision instead of about 25 ms.
- **One model for everyone.** All opponents, and our teammate, are assumed to
  play like `HumanRulesAgent3`. Per-player noise, learned from how often each
  player's past actions matched the rules, would adapt to unfamiliar
  opponents.
- **All-or-nothing mismatches.** A near miss (leading a 6 when the rules say
  5) costs as much as a wild one. A graded likelihood, such as a softmax over
  the heuristic's scores, would keep more information.
- **Trade evidence is approximate.** Returned cards are unknown, and trades
  are scored on an estimated pre-trade hand.
