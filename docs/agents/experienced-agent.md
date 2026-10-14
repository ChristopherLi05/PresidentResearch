# ExperiencedAgent

`ExperiencedAgent` uses deterministic hand-shape rules. It counts distinct
ordinary ranks as future group cover and counts 2s as bombs. A position is
considered safe when ordinary groups are at least as numerous as 2s; this avoids
ending with bombs that cannot be played without a penalty.

When leading or following, it prefers safe actions that remove an entire rank
group, avoids splitting pairs or triples, keeps J through A where possible, and
plays low ranks first. It finishes with an ordinary play whenever possible and
will not intentionally finish with a bomb. It spends a 2 when the current hand
is unsafe or when the number of 2s reaches the available ordinary group cover.

It reconstructs opponents' remaining card totals from public plays, bombs,
completion events, and done announcements. That information only affects its
pass rule: on a low board it can pass rather than split a group, unless an
opponent has two or fewer cards.

Trading is also structural. Requests prioritize completing triples, matching a
singleton Ace, acquiring a safe extra 2, then improving K/Q/J groups and lower
ranks. Returns prefer a low triple that becomes a pair, then a low singleton,
while avoiding 2s, quads, high cards, and moves that break bomb cover.
