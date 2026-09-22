from typing import Any, Iterable, Mapping, Optional, Sequence
import random


class Client:
    """Base class for a JSON-protocol player. Override :meth:`respond`."""

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
