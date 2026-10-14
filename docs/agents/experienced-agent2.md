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
