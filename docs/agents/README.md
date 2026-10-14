# Agent policies

Each policy receives only its private hand, the public event history, seating,
active players, the board, and the engine's legal actions. Policies return one
of those legal actions. They do not inspect opponents' private cards.

| Policy | Decision model |
| --- | --- |
| [RandomAgent](random-agent.md) | Uniformly random legal action, with a 50% preference for legal passes. |
| [BasicAgent](basic-agent.md) | Deterministic lowest-play and highest-request baseline. |
| [BasicAgentPlus](basic-agent-plus.md) | Basic policy with a high-card bomb substitution. |
| [ExperiencedAgent](experienced-agent.md) | Hand-shape and 2-cover heuristics using public card totals. |
| [ExperiencedAgent2](experienced-agent2.md) | Hypergeometric response estimates, soft hand costs, and one-lead lookahead. |
| [SwitchPAgent](switchp-agent.md) | Port of switch+'s integer `x`/`z`/`y` decision rules and `keep_kou2` behavior. |

`BasicAgent` and all experienced policies draft the highest face-up pile;
`RandomAgent` chooses randomly. `ExperiencedAgent`, `ExperiencedAgent2`, and
`SwitchPAgent` support the full later-round trade, completion, and 2-bomb rules.
Benchmark commands appear in the repository root README.
