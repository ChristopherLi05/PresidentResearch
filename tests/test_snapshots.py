"""Shared observations preserve privacy, snapshot isolation, and game decisions."""

from copy import deepcopy
import json
import random
import unittest
from unittest.mock import patch

from src.agent import BasicAgent, ExperiencedAgent2, SwitchPAgent
from src.president import Game
from src.snapshots import SnapshotHistory
from tests.test_president import PLAYERS, card, ready_round


def run_with_independent_messages(game, clients, max_actions=3000):
    """Original Game.run path, used as a behavior and timing reference."""
    round_ = game.round or game.start_round()
    for player, client in clients.items():
        client.observe(round_.message_for(player))
    for _ in range(max_actions):
        if round_.done:
            return round_
        player = round_._actor()
        round_.apply_action(player, clients[player].respond(round_.message_for(player)))
        for player, client in clients.items():
            client.observe(round_.message_for(player))
    if not round_.done:
        raise RuntimeError("action limit reached")
    return round_


class RecordingAgent(BasicAgent):
    def __init__(self):
        self.observations = []

    def observe(self, message):
        self.observations.append(message)


class SnapshotTests(unittest.TestCase):
    def test_shared_public_fields_private_hands_and_deepcopy_compatibility(self):
        game = Game(PLAYERS, rng=random.Random(3))
        round_ = game.start_round()
        clients = {p: RecordingAgent() for p in PLAYERS}
        request = game._notify_clients(clients)
        actor = round_._actor()
        for player, client in clients.items():
            message = client.observations[0]
            self.assertEqual(deepcopy(message), round_.message_for(player))
            self.assertEqual(json.loads(json.dumps(deepcopy(message))), round_.message_for(player))
            self.assertIs(message["events"], request["events"])
            self.assertIs(message["players"], request["players"])
            self.assertIsNot(message["hand"], request["hand"])
            self.assertEqual(message["request"] is not None, player == actor)
            self.assertNotIn("turn_history", message)
        observed = clients[actor].observations[0]
        observed["hand"][0]["rank"] = "invalid"
        observed["request"]["legal_actions"].clear()
        observed["phase"] = "invalid"
        self.assertEqual(deepcopy(request), round_.message_for(actor))

    def test_nested_public_state_is_immutable_and_old_snapshots_stay_stable(self):
        game = Game(PLAYERS, rng=random.Random(3))
        round_ = game.start_round()
        clients = {p: RecordingAgent() for p in PLAYERS}
        history = SnapshotHistory()
        request = game._notify_clients(clients, history)
        original = deepcopy(request)
        with self.assertRaises(TypeError):
            request["events"][0]["type"] = "corrupted"
        with self.assertRaises(TypeError):
            request["players"][0] = "corrupted"
        with self.assertRaises(TypeError):
            request["roles"]["a"] = 99
        actor = round_._actor()
        round_.apply_action(actor, clients[actor].respond(request))
        updated = game._notify_clients(clients, history)
        self.assertEqual(deepcopy(request), original)
        self.assertIs(request["events"][0], updated["events"][0])
        play = next(e for e in updated["events"] if e["type"] == "opening_play")
        with self.assertRaises(TypeError):
            play["cards"][0]["rank"] = "invalid"
        with self.assertRaises(TypeError):
            updated["top"]["rank"] = "invalid"

    def test_only_new_events_are_frozen_and_unchanged_history_is_reused(self):
        events = [{"type": "first", "cards": [card("3").json()]}]
        history = SnapshotHistory()
        first = history.snapshot(events)
        with patch("src.snapshots.freeze", side_effect=AssertionError("refroze history")):
            self.assertIs(history.snapshot(events), first)
        events.append({"type": "second"})
        second = history.snapshot(events)
        self.assertIs(first[0], second[0])
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 2)
        replacement = history.snapshot([{"type": "replacement"}])
        self.assertEqual(replacement[0]["type"], "replacement")

    def test_observation_mutations_do_not_corrupt_response_or_other_players(self):
        class EditingAgent(BasicAgent):
            def observe(self, message):
                message["hand"].clear()
                message["events"] = []
                if message["request"]:
                    message["request"]["legal_actions"].clear()

        game = Game(PLAYERS)
        game.round = ready_round({"a": [card("7")], "b": [card("8")]})
        observer = RecordingAgent()
        game.run({"a": EditingAgent(), "b": observer}, max_actions=1)
        self.assertEqual(game.round.rankings["a"], 3)  # Two fixture seats already finished.
        self.assertTrue(observer.observations[0]["events"])
        self.assertEqual(observer.observations[0]["hand"], [card("8").json()])
        self.assertIsNone(observer.observations[0]["rankings"])
        with self.assertRaises(TypeError):
            observer.observations[-1]["rankings"]["a"] = 99

    def test_game_matches_original_notifications_decisions_and_prediction_learning(self):
        for seed in range(3):
            before = Game(PLAYERS, rng=random.Random(seed))
            after = Game(PLAYERS, rng=random.Random(seed))
            policies = (BasicAgent, ExperiencedAgent2, SwitchPAgent, SwitchPAgent)
            old_clients = {p: cls() for p, cls in zip(PLAYERS, policies)}
            new_clients = {p: cls() for p, cls in zip(PLAYERS, policies)}
            for _ in range(3):
                before.start_round()
                after.start_round()
                run_with_independent_messages(before, old_clients)
                after.run(new_clients, max_actions=3000)
                self.assertEqual(after.round.turn_history, before.round.turn_history)
                self.assertEqual(after.round.events, before.round.events)
                self.assertEqual(after.round.rankings, before.round.rankings)
                for p in PLAYERS[2:]:
                    self.assertEqual(new_clients[p]._prediction_results,
                                     old_clients[p]._prediction_results)


if __name__ == "__main__":
    unittest.main()
