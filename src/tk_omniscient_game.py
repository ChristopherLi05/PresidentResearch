"""Run President with one omniscient human seat and three basic opponents.

This is a local debugging/learning host. It intentionally gives the human
client every player's hand and every undealt draft pile; do not use it for a
real private game. Run it with ``python -m src.tk_omniscient_game``.
"""

from __future__ import annotations

from typing import Any

from src.client import BasicClient, Client
from src.president import Game, RANK_VALUE, Round
from src.tk_client import OmniscientHumanClient


PLAYERS = ("you", "north", "west", "east")


def omniscient_message(round_: Round, player: str) -> dict[str, Any]:
    """Return *player*'s normal message plus every local card collection."""
    message = round_.message_for(player)
    collections = {
        name: [card.json() for card in sorted(hand, key=lambda card: (RANK_VALUE[card.rank], card.suit))]
        for name, hand in round_.hands.items()
    }
    if hasattr(round_, "piles"):
        collections.update({
            f"Pile {index + 1}": [card.json() for card in pile]
            for index, pile in enumerate(round_.piles)
        })
    message["all_hands"] = collections
    return message


def run(game: Game, human: OmniscientHumanClient, clients: dict[str, Client]) -> Round:
    """Run one round while continuously refreshing the omniscient display."""
    assert game.round is not None
    while not game.round.done:
        human.observe(omniscient_message(game.round, "you"))
        actor = game.round._actor()
        assert actor is not None
        if actor == "you":
            action = human.respond(omniscient_message(game.round, actor))
        else:
            action = clients[actor].respond(game.round.message_for(actor))
        game.round.apply_action(actor, action)
    human.observe(omniscient_message(game.round, "you"))
    return game.round


def main() -> None:
    game = Game(PLAYERS)
    human = OmniscientHumanClient(title="President — Omniscient view")
    clients: dict[str, Client] = {"you": human}
    clients.update({player: BasicClient() for player in PLAYERS if player != "you"})
    try:
        while True:
            game.start_round()
            run(game, human, clients)
            if not human.play_again(game.finish_round()):
                return
    except RuntimeError as error:
        if str(error) != "human client window was closed":
            raise


if __name__ == "__main__":
    main()
