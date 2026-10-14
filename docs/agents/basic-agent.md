# BasicAgent

`BasicAgent` is a deterministic rule baseline. It drafts the pile with the
highest exposed card. During a trade request it asks for the highest rank that
has not publicly failed in the current exchange; after a successful exchange,
that failure memory resets. It returns the lowest card, breaking ties by suit.

On the board, it always completes when a completion is legal. Otherwise it
chooses the lowest ordinary play or 2-bomb. For equal ranks, it chooses the
largest legal group, which tends to remove more cards without evaluating the
future hand. It passes or declines only when the engine offers no playable or
completion action.

The policy does not estimate opponents' cards, value hand structure, or protect
2s. Its only public-history use is avoiding duplicate failed trade requests.
