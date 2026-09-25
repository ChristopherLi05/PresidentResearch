"""Host a local four-human President game in four Tk windows.

Run ``python -m src.tk_four_human_game`` from the repository root. Each seat
has its own :class:`HumanClient`, so its hand and private requests remain local
to that window while all four clients use the normal engine protocol.
"""

from __future__ import annotations

from src.client import Client
from src.president import Game
from src.tk_client import HumanClient


PLAYERS = ("North", "West", "South", "East")


def main() -> None:
    game = Game(PLAYERS)
    humans = {
        player: HumanClient(title=f"President — {player}") for player in PLAYERS
    }
    clients: dict[str, Client] = humans
    try:
        while True:
            game.start_round()
            game.run(clients)
            # North acts as the local host between rounds. Placements become
            # the roles used for the next round's drafting and trading phase.
            if not humans["North"].play_again(game.finish_round()):
                return
    except RuntimeError as error:
        # Closing any one seat's window stops this local host cleanly.
        if str(error) != "human client window was closed":
            raise


if __name__ == "__main__":
    main()
