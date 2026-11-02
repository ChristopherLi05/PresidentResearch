# C engine and Python interface

`src/president.c` is a dependency-free C11 implementation of the engine in
`src/president.py`. Its public C interface is in `src/president.h`.
`src/president_c.py` provides a standard-library `ctypes` interface for Python.
The existing Python engine remains available as the reference implementation.

## Build and call from Python

From the repository root, with GCC or Clang on PATH:

```sh
python -m src.president_c --build
```

This creates `build/president.dll` on Windows, `build/libpresident.so` on Linux,
or `build/libpresident.dylib` on macOS. The compiler and Python must target the
same architecture. No Python development headers or third-party packages are
needed. Use `--compiler PATH` or set `CC` to select a compiler executable.
Rebuild after changing `president.c` or `president.h`, before loading the library
in Python; restart Python before rebuilding a DLL it has already loaded.

Call the engine with the same dictionary action/message format:

```python
import random
from src.president_c import Round

with Round(["a", "b", "c", "d"], first_round=True,
           rng=random.Random(42)) as round_:
    while not round_.done:
        player = round_._actor()
        actions = round_.legal_actions(player)
        events = round_.apply_action(player, random.choice(actions))
    print(round_.rankings)
```

Existing Python agents can run through `Game`:

```python
import random
from src.agent import BasicAgent
from src.president_c import Game

players = ["a", "b", "c", "d"]
clients = {player: BasicAgent() for player in players}
with Game(players, rng=random.Random(42)) as game:
    for _ in range(3):
        game.start_round()
        game.run(clients)
        print(game.finish_round())
```

Python `Round` supports `message_for`, `legal_actions`, `apply_action`, `done`,
`rankings`, `events`, `turn_history`, `hands`, `roles`, `players`, and `phase`.
`Game` supports `start_round`, `run`, and `finish_round`. It retains the existing
Python client adapter and immutable public snapshots; C performs round rules,
validation, legal-action generation, and transitions. Python's `random.Random`
shuffles the deck at this boundary, so identical Python seeds and actions give
the same results as the reference engine, including later rounds.

Player IDs must be four distinct strings without NUL characters. State
properties return independent copies; assigning internal state to construct
artificial mid-round fixtures is not supported. The wrapper is compatible
with the public action/message workflow, not every private Python attribute.
Use a context manager or `close()` to release a round promptly; garbage
collection also releases it. Previously returned Python rounds remain valid
after `Game.start_round()` and can be closed separately when no longer needed.
A single round/game must not be called concurrently.

`PRESIDENT_LIBRARY` can point to a separately built shared library. Importing
the wrapper does not compile code; a missing library produces an error with
the build command. The JSON/dictionary boundary has conversion overhead, so
this interface does not promise faster end-to-end Python simulations.

## Native C interface

Link `src/president.c` into a C11 program and include `president.h`. There is no
`main` in the engine. For example, to create a deterministic round and play
its first legal action:

```c
#include "president.h"
#include <stdio.h>

int main(void) {
    const char *players[4] = {"a", "b", "c", "d"};
    PresidentCard deck[52];
    PresidentRound *round = NULL;
    PresidentAction action;
    size_t start;
    int player;
    char *events;
    president_standard_deck(deck);
    if (president_round_create(&round, players, 1, NULL, deck, 52, 0)
            != PRESIDENT_OK) return 1;
    player = president_round_actor(round);
    if (!president_round_legal_actions(round, player, &action, 1) ||
        president_round_apply(round, player, &action, &start) != PRESIDENT_OK) {
        president_round_destroy(round);
        return 1;
    }
    events = president_round_events_json(round, start);
    if (events) puts(events);
    president_free(events);
    president_round_destroy(round);
    return events ? 0 : 1;
}
```

Compile a host file named `example.c` with:

```sh
gcc -std=c11 -O2 -Wall -Wextra -Isrc example.c src/president.c -o example
```

- Player seats are 0 through 3. Cards are `rank * 4 + suit`, where ranks follow
  `RANKS` and suits follow `C,D,H,S`. `president_card("A", "S")` returns 47.
- C actions are `PresidentAction` values; no JSON input parser is required.
  Play/completion actions use `count` and `cards`; bomb/return actions use
  `cards[0]`; draft/request actions use `pile`/`rank` respectively.
- Query legal-action capacity by passing `NULL, 0`, or pass an output buffer.
  The function returns the total count even if the supplied buffer is smaller.
- `president_round_apply` returns a status and the offset of newly appended
  public events. Invalid actions leave state and logical histories unchanged.
- JSON messages and event arrays match `PROTOCOL.md`. They are allocated
  snapshots; release them with `president_free`. Returned trade cards appear
  only in the private history, never the public events.
- `president_round_history_at` copies an accepted typed action and actor.
  Native actions have no arbitrary extra JSON metadata; the Python wrapper
  retains extra action fields in its Python `turn_history`.
- The native `PresidentGame` carries placements into later rounds and offers
  observer/respond callbacks. A successful native `start_round` invalidates
  the previous borrowed C round pointer. Failed starts preserve the game.
- Native omitted-deck shuffling uses its own deterministic RNG, which differs
  from Python's. Supply the same explicit deck to compare the two C/Python
  interfaces. The Python wrapper already does this automatically.

## Verification

```sh
python -m unittest discover -s tests -p test_president_c.py -v
```

The tests compile a temporary shared library with strict compiler warnings,
compare whole games with the Python reference over 36 seeded rounds, check
exact legal-action ordering, messages, events and private history, exercise
invalid actions and snapshot ownership, test C callbacks and action limits,
and run the Python wrapper with existing agents. These tests skip when no
GCC/Clang compiler is available. Only Windows/GCC has been verified locally;
the source and build helper also support GCC/Clang on Linux/macOS.
