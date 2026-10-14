# SwitchPAgent

`SwitchPAgent` uses the switch+ `kou2`/`keep_kou2` decision model rather than
`ExperiencedAgent2`'s probability-weighted action score. It retains
`ExperiencedAgent`'s drafting and trade policy. Its integer formulas and
President-specific adaptations are described below.

For each ordinary rank group it computes switch+'s integer strength `x`: start
at 100 and subtract fixed penalties for every publicly possible response. The
hand evaluation `z` buckets those strengths, while the weakest and
second-weakest groups determine whether a controlled finish sequence exists.
Lower `z` is better; `STRONG` is 86.

For a single, each possible higher-or-equal response rank subtracts 30 from
`x`. For pairs and triples, the penalty is 4, 9, 15, or 24 depending on the
remaining unseen copies; a possible response costs 30 in a two-player position.
Possible 2-bombs and out-of-turn completions also subtract 30. A quad or a
matching-stack clear gets `x = 200`.

`z` adds 2 for a group with `x = 1–30`, 1 for `31–60`, zero for `61–90`, and
subtracts value for `91–200`; quads are omitted. The source's sentinel for a
missing second group is 201. This preserves the source's distinction between a
hand with one weak group and one whose remaining groups can control the board.

On a clear board it assigns switch+'s lead priority `y`, which combines group
strength, rank, group size, and remaining same-sized groups. When the
second-weakest group is strong, the agent leads a control group to start its
finish sequence. When following, it considers ranks from low to high and applies
the source's ordered accept/reject tests rather than minimizing one global score.

`keep_kou2` can pass instead of spending K or higher when the reconstructed
initial hand, or post-second-clear hand, has poor `z`. After three board clears,
it predicts the leading opponent from board clears won and cards shed. Completed
round observations track how accurate those predictions were and gate later
conservation decisions.

The port maps source-specific concepts to this game: 2-bombs and possible
completions reduce `x`; quads and matching-stack clears receive maximum control;
there are no source branches for Jokers, sequences, suit locks, revolution, or
8-clears. The source's multi-strategy tournament selector is also omitted;
this is the `kou2`/`keep_kou2` policy only.
