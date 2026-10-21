"""Immutable public data for in-process observations; wire messages stay mutable."""

from collections.abc import Mapping
from copy import deepcopy
from types import MappingProxyType


class FrozenMapping(Mapping):
    """Read-only mapping whose deep copy is an ordinary, independent dictionary."""

    __slots__ = ("_data",)

    def __init__(self, values):
        object.__setattr__(self, "_data", MappingProxyType(dict(values)))

    def __setattr__(self, name, value):
        raise TypeError("snapshot is read-only")

    def __delattr__(self, name):
        raise TypeError("snapshot is read-only")

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __deepcopy__(self, memo):
        result = {}
        memo[id(self)] = result
        result.update((key, deepcopy(value, memo)) for key, value in self.items())
        return result


class FrozenSequence(tuple):
    """Immutable JSON array, restored to a list by deepcopy."""

    __slots__ = ()

    def __deepcopy__(self, memo):
        result = []
        memo[id(self)] = result
        result.extend(deepcopy(value, memo) for value in self)
        return result


def freeze(value):
    """Freeze a JSON-shaped value, copying all mutable containers once."""
    if isinstance(value, Mapping):
        return FrozenMapping((key, freeze(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return FrozenSequence(freeze(item) for item in value)
    return value


class SnapshotHistory:
    """Reuse frozen entries from the engine's append-only public event log."""

    def __init__(self):
        self._source = None
        self._events = FrozenSequence()

    def snapshot(self, events):
        if events is not self._source or len(events) < len(self._events):
            self._source = events
            self._events = FrozenSequence()
        if len(events) > len(self._events):
            self._events = FrozenSequence((*self._events,
                *(freeze(event) for event in events[len(self._events):])))
        return self._events
