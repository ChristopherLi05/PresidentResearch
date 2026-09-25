from typing import Any, Mapping, Optional
import random

from src.president import RANK_VALUE


class Client:
    """Base class for a JSON-protocol player. Override :meth:`respond`.

    Hosts may also call :meth:`observe` with state snapshots between turns.
    Stateless clients can ignore those nonblocking updates; interactive clients
    can use them to keep their display current while another player acts.
    """

    def observe(self, message: Mapping[str, Any]) -> None:
        """Receive a nonblocking state update. The default client ignores it."""

    def respond(self, message: Mapping[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class RandomClient(Client):
    """Example client: passes half the time when passing is legal, else acts randomly."""

    def __init__(self, rng: Optional[random.Random] = None) -> None: self.rng = rng or random.Random()

    def respond(self, message: Mapping[str, Any]) -> dict[str, Any]:
        actions = message["request"]["legal_actions"]
        passes = [a for a in actions if a["type"] in {"pass", "decline_completion"}]
        if passes and self.rng.random() < .5: return self.rng.choice(passes)
        return self.rng.choice([a for a in actions if a not in passes] or actions)


class BasicClient(Client):
    """Always act when possible, playing low and drafting/requesting high.

    Equal-rank plays prefer the largest group. Trade returns use the lowest
    card. Failed requests are remembered from the public history of the
    current exchange, so this client needs no private state between responses.
    """

    def respond(self, message: Mapping[str, Any]) -> dict[str, Any]:
        request = message.get("request")
        if not request or not request["legal_actions"]:
            raise ValueError("no legal action requested")
        actions = request["legal_actions"]

        if request["type"] == "draft":
            start = next(e for e in reversed(message["events"])
                         if e["type"] == "round_started")
            return max(actions, key=lambda a: RANK_VALUE[start["face_up"][a["pile"]]["rank"]])

        if request["type"] == "trade_request":
            failed = set()
            for event in reversed(message["events"]):
                if event["type"] in {"trade_completed", "round_started"}:
                    break
                if (event["type"] == "trade_requested"
                        and event["initiator"] == message["player"]
                        and not event["success"]):
                    failed.add(event["rank"])
            candidates = [a for a in actions if a["rank"] not in failed]
            return max(candidates, key=lambda a: RANK_VALUE[a["rank"]])

        if request["type"] == "trade_return":
            return min(actions, key=lambda a: (RANK_VALUE[a["card"]["rank"]], a["card"]["suit"]))

        completions = [a for a in actions if a["type"] == "complete"]
        if completions:
            return completions[0]

        plays = [a for a in actions if a["type"] in {"play", "bomb"}]
        if plays:
            def play_key(action):
                cards = action["cards"] if action["type"] == "play" else [action["card"]]
                return (RANK_VALUE[cards[0]["rank"]], -len(cards),
                        tuple(sorted(c["suit"] for c in cards)))

            return min(plays, key=play_key)

        # The engine also polls players who cannot complete, and a turn may
        # have no playable cards. Only then is a pass/decline necessary.
        return actions[0]
