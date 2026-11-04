"""Differential tests; build with python -m src.agent_search_c --build first."""
import ctypes as ct
from copy import deepcopy
import random
import unittest
from unittest.mock import patch

from src.agent import ExperiencedAgent2, ExperiencedAgent3, ExperiencedAgent3Tuned, ExperiencedAgent4, BasicAgent
from collections import Counter
from src import agent_search_c as native
from src.agent_search import (snapshot, heuristic_action, rollout, evaluate,
                              unique_actions, prepare, sample_world)
from src.president import Round, Card
from src.president_c import _Action, _encode_action
from tests.test_president import ready_round


class NativeSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not native.library_path().exists():
            raise unittest.SkipTest("Build native search with python -m src.agent_search_c --build")
        cls.lib = native.load_library(str(native.library_path()))

    def setUp(self):
        self.agent = ExperiencedAgent4(random.Random(42), search_time_ms=None, search_worlds=2,
                                       search_backend="python", evaluation_backend="python")

    def test_earlier_agents_hand_and_lead_parity(self):
        rng = random.Random(18)
        for cls in (ExperiencedAgent2, ExperiencedAgent3, ExperiencedAgent3Tuned, ExperiencedAgent4):
            py, c = cls(evaluation_backend="python"), cls(evaluation_backend="c")
            for _ in range(100):
                ranks = list(native.RANKS)
                rng.shuffle(ranks)
                hand = Counter({r: rng.randrange(5) for r in ranks})
                unseen = Counter({r: rng.randrange(5-hand[r]) for r in ranks})
                opponents = [(str(i), rng.randrange(1, 14)) for i in range(rng.randrange(4))]
                self.assertEqual(py._hand_cost(hand, unseen, opponents), c._hand_cost(hand, unseen, opponents))
                self.assertEqual(py._lead_cost(hand, unseen, opponents), c._lead_cost(hand, unseen, opponents))

    def test_hand_evaluator_keeps_cache_and_custom_helpers(self):
        agent = ExperiencedAgent2(evaluation_backend="c")
        hand, unseen, opponents = Counter({"3": 2, "A": 1}), Counter({"4": 3}), [("b", 3)]
        agent._hand_cost_cached.cache_clear()
        with patch.object(self.lib, "president_hand_value", wraps=self.lib.president_hand_value) as call:
            first = agent._hand_cost(hand, unseen, opponents)
            self.assertEqual(first, agent._hand_cost(hand, unseen, opponents))
            self.assertEqual(call.call_count, 1)
        custom = ExperiencedAgent2()
        with patch.object(custom, "_control", return_value=.25) as control:
            custom._hand_cost(hand, unseen, opponents)
            self.assertTrue(control.called)

    def scores(self, state, actions):
        position = native.pack(state)
        encoded = (_Action * len(actions))(*(_encode_action(a) for a in actions))
        scores = (ct.c_double * len(actions))()
        status = self.lib.president_search_scores(ct.byref(position), state.players.index(state._actor()),
                                                 encoded, len(actions), native.weights(self.agent), scores)
        self.assertEqual(status, 1)
        return list(scores)

    def check_rollout(self, state, action):
        player = state._actor()
        expected = rollout(self.agent, state, player, action, float("inf"))
        position, result = native.pack(state), native.Position()
        score = ct.c_double()
        encoded = _encode_action(action)
        status = self.lib.president_search_rollout(ct.byref(position), state.players.index(player),
            ct.byref(encoded), native.weights(self.agent), ct.byref(result), ct.byref(score))
        self.assertEqual(status, 1)
        expected_position = native.pack(expected)
        for name, _ in native.Position._fields_:
            if name in {"cards", "finish", "bombs", "completion", "pending_skip"}:
                continue  # Ignore unused slots and stale skip after clearing.
            actual, wanted = getattr(result, name), getattr(expected_position, name)
            if isinstance(actual, ct.Array):
                actual, wanted = list(actual), list(wanted)
            self.assertEqual(actual, wanted, name)
        for p in range(4):
            n = result.sizes[p]
            self.assertEqual(list(result.cards[p][:n]), list(expected_position.cards[p][:n]))
        for name, count in (("finish", result.finish_count), ("bombs", result.bomb_count),
                            ("completion", result.completion_count)):
            self.assertEqual(list(getattr(result, name)[:count]), list(getattr(expected_position, name)[:count]))
        self.assertAlmostEqual(score.value, evaluate(self.agent, expected, player), places=11)

    def test_scores_and_rollouts_match_across_complete_games(self):
        checked = 0
        for seed in range(4):
            state = Round(list("abcd"), first_round=True, rng=random.Random(seed))
            step = 0
            while not state.done:
                player = state._actor()
                actions = unique_actions(state.legal_actions(player))
                expected = self.agent._play_scorer(snapshot(state, player))
                for action, score in zip(actions, self.scores(state, actions)):
                    self.assertAlmostEqual(score, expected(action), places=10,
                                           msg=f"seed={seed} step={step} action={action}")
                    checked += 1
                if step % 13 == 0:
                    for action in actions:
                        self.check_rollout(state, action)
                state.apply_action(player, heuristic_action(self.agent, state))
                step += 1
                self.assertLess(step, 2000)
        self.assertGreater(checked, 1000)

    def test_fixed_work_decisions_match_sampled_worlds(self):
        for seed in range(3):
            state = Round(list("abcd"), first_round=False, roles=dict(zip("abcd", (1, 2, 3, 4))),
                          rng=random.Random(seed))
            basic = BasicAgent()
            while state.phase in {"draft", "trade_request", "trade_return"}:
                p = state._actor()
                state.apply_action(p, basic.respond(state.message_for(p)))
            for step in range(100):
                if state.done:
                    break
                p = state._actor()
                message = state.message_for(p)
                actions = message["request"]["legal_actions"]
                if step % 5 == 0:
                    py = ExperiencedAgent4(random.Random(9), search_time_ms=None, search_worlds=3,
                                           search_backend="python", evaluation_backend="python")
                    c = ExperiencedAgent4(random.Random(9), search_time_ms=None, search_worlds=3, search_backend="c")
                    self.assertEqual(py._choose_play(message, actions), c._choose_play(message, actions))
                    self.assertEqual(py.last_search["worlds"], c.last_search["worlds"])
                state.apply_action(p, heuristic_action(self.agent, state))

    def test_special_transitions(self):
        state = ready_round({"a": [Card("7", "C"), Card("2", "C"), Card("3", "C")],
                             "b": [Card("7", s) for s in "DHS"],
                             "c": [Card("8", "C"), Card("4", "C")]})
        # Account for each seat's public card count in this small fixture.
        from src.president import standard_deck
        used = [c for c in standard_deck() if all(c not in h for h in state.hands.values())]
        state.events = []
        for seat in state.players:
            count = 13-len(state.hands[seat])
            state.events.append({"type": "played" if state.events else "opening_play", "player": seat,
                                 "cards": [c.json() for c in used[:count]]})
            used = used[count:]
        state.top_rank, state.top_count, state.top_stack_count = "7", 1, 1
        state.last_player = "c"
        completion = deepcopy(state)
        completion._offer_completions("a")
        for position in (state, completion):
            for action in position.legal_actions(position._actor()):
                self.check_rollout(position, action)

    def test_batch_deadline_and_input_immutability(self):
        state = Round(list("abcd"), first_round=True, rng=random.Random(42))
        message = state.message_for(state._actor())
        world = sample_world(prepare(self.agent, message), state._actor(), random.Random(7))
        before = deepcopy(world.__dict__)
        actions = unique_actions(world.legal_actions(world._actor()))
        self.assertIsNone(native.batch(self.lib, self.agent, world, world._actor(), actions, 0))
        scores = native.batch(self.lib, self.agent, world, world._actor(), actions, -1)
        expected = [evaluate(self.agent, rollout(self.agent, world, world._actor(), a, float("inf")), world._actor()) for a in actions]
        for got, want in zip(scores, expected):
            self.assertAlmostEqual(got, want, places=11)
        self.assertEqual(world.__dict__, before)

    def test_weight_changes_and_override_fallback(self):
        state = Round(list("abcd"), first_round=True, rng=random.Random(11))
        self.agent.GROUP_WEIGHT = 4.3
        self.agent.NEXT_RESPONSE_PENALTY = 2.1
        self.agent.CONTROL_RESERVE_WEIGHT = 0
        actions = unique_actions(state.legal_actions(state._actor()))
        scorer = self.agent._play_scorer(snapshot(state, state._actor()))
        for a, score in zip(actions, self.scores(state, actions)):
            self.assertAlmostEqual(scorer(a), score, places=11)
        agent = ExperiencedAgent4()
        with patch.object(agent, "_hand_cost", return_value=0):
            self.assertIsNone(native.backend(agent))
            agent.search_backend = "c"
            with self.assertRaises(RuntimeError):
                native.backend(agent)

    def test_invalid_snapshot_rejected(self):
        state = Round(list("abcd"), first_round=True, rng=random.Random(42))
        position = native.pack(state)
        position.cards[0][0] = 99
        action = _encode_action(state.legal_actions(state._actor())[0])
        scores = (ct.c_double * 1)()
        self.assertEqual(self.lib.president_search_batch(ct.byref(position), state.players.index(state._actor()),
                         ct.byref(action), 1, native.weights(self.agent), -1, scores), -1)


if __name__ == "__main__":
    unittest.main()
