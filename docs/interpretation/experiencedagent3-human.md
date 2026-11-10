# ExperiencedAgent3Tuned: table-side rules

These are the playing rules taken from
[experiencedagent3-claude-revised.md](experiencedagent3-claude-revised.md) and
[experiencedagent3-codex-revised.md](experiencedagent3-codex-revised.md). They
are written so a person can follow them at the table. `HumanRulesAgent3` in
`src/agent.py` implements them literally.

The rules describe what `ExperiencedAgent3Tuned` usually does. They are not a
proof of good play. The percentage after a rule is how often the tuned agent
agreed with it in the Claude revision's logs.

## Terms

- **Group**: all the cards you hold of one ordinary rank (3 through A). A
  single card is a group of one.
- **G**: how many groups you hold.
- **B**: how many 2s you hold.
- **Clean follow**: a legal play that uses a whole group of exactly the
  board's width, so no pair or triple is broken.
- **Smallest opponent**: the fewest cards held by any opponent still playing.

## 0. Always first

1. **Finish now if you can.** Take any normal play or completion that empties
   your hand.
2. **Never go out on a 2.** A bomb finish places you last.
3. **Never leave yourself holding only 2s.** If another play exists, use it.

## 1. Trading

- **Draft:** take the pile with the highest face-up card.
- **Request:**
  1. **T1.** If you hold a triple, ask for its fourth card. If you hold two
     triples, ask for the higher one. (≈100%)
  2. **T2.** Otherwise ask for a 2. **Exception:** if you already hold two or
     more 2s and a lone Ace, ask for the Ace. (94–95%)
  3. **T2b.** If the request for the fourth card fails, ask for a 2. If the
     request for a 2 fails, ask for the card that pairs your highest single.
  4. Never repeat a failed rank in the same exchange. Never ask for a rank you
     already hold four of.
- **Return (T3):** give back your lowest single. If you have no singles, take
  one card off your lowest triple. Otherwise give the lowest ordinary card.
  Do not give back a 2. (100% in sampled exchanges)

## 2. Leading (3+ players)

Check these in order:

1. **L3. An opponent has 1 card left.** If you hold a pair or larger, lead
   your lowest such group whole. (71–89%)
2. **L2 split. You have no singles and B ≥ G.** Lead one card from your lowest
   group. This keeps an ordinary card to follow each 2.
3. **L1. You have a single.** Lead your lowest single, even if a lower pair
   exists. (93–96%)
4. **L2. Otherwise** lead your lowest whole group. Do not open a full 13-card
   hand with a quad if another group exists. (71–86%)

On the opening lead, only plays with the 3 are legal. Lead all your 3s.

## 3. Following (3+ players)

Check these in order:

1. **F2. Bomb rule.** If you hold a 2, bomb when any of these is true
   (98–99.7%):
   - no clean follow exists;
   - G − B ≤ 1, so your 2s plus one last group take you out;
   - an opponent has exactly 1 card and the board is a single.
2. **F1. Clean follow.** Play the lowest clean follow. Do not pass. (≥99.5%)
3. **F3. No 2 and no clean follow.** (81–91%)
   - On a single, break a triple: play one card from your lowest triple that
     beats the board. This leaves a pair.
   - Otherwise break a group only if the smallest opponent has 2 or 3 cards.
     Play the lowest such option.
   - If every opponent has 4+ cards, or one has exactly 1 card, pass.

## 4. Completions

- **C1.** Complete, unless afterwards B ≥ G with B > 0. That would leave
  your 2s without enough groups to follow them. (97.5–99%)
- The closest call is two groups and two 2s after completing. The agent
  declines it 57% of the time, so the rule says decline.

## 5. Heads-up (2 players left)

Every unseen card is now in the last opponent's hand. Count them and work out
the line. The singles-first and keep-groups-intact habits no longer apply.

`HumanRulesAgent3` uses the same exact solver as `ExperiencedAgent3` for this.
It falls back to the rules above only if the position cannot be rebuilt, for
example after a player announced their 2s.

## Cheat sheet

0. Finish if you can. Never go out on a 2. Never leave only 2s.
1. **Trade:** triple → ask for its fourth card; else ask for a 2 (Ace if you
   have 2+ twos and a lone Ace). After a failed 2, pair your highest single.
   Return your lowest single.
2. **Lead:** opponent on 1 card → lead a pair. Otherwise lead your lowest
   single, then your lowest group. Split a group if your 2s match or
   outnumber your groups.
3. **Bomb** when you would have to break a group, when your 2s plus one group
   take you out, or when an opponent is on their last card and the board is
   a single.
4. **Follow** with the lowest whole group that fits. Never pass when one
   exists.
5. **No 2:** break a triple to beat a single. Break a pair only if someone
   has 2–3 cards. Otherwise pass.
6. **Complete** unless it leaves at least as many 2s as groups.
7. **Heads-up:** count the remaining cards and play it out exactly.


