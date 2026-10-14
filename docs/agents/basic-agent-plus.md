# BasicAgentPlus

`BasicAgentPlus` inherits every draft, trade, completion, and ordinary-play rule
from `BasicAgent`. Its only change applies to a normal play decision: when the
lowest available ordinary rank is J, Q, K, or A and a legal 2-bomb exists, it
uses the bomb instead.

This creates a simple threshold policy for spending 2s. It does not evaluate
whether a bomb leaves enough ordinary groups to dispose of later 2s, does not
consider opponent hand sizes, and does not alter a forced pass or completion.
