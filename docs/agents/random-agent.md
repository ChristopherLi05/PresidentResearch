# RandomAgent

`RandomAgent` is a protocol smoke-test policy, not a competitive baseline. It
collects the engine-provided legal actions. If a pass or completion decline is
legal, it selects one with probability 0.5; otherwise it selects uniformly from
the remaining legal actions. If no pass-like action exists, every legal action
has equal probability.

It retains no game state and makes no distinction among ranks, group size,
hand shape, roles, or opponent information. Supplying a `random.Random` instance
at construction makes its choices reproducible.
