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
                              unique_actions, prepare, sample_world, clone)
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

    def check_rollout(self, state, action, agent=None):
        agent = agent or self.agent
        player = state._actor()
        expected = rollout(agent, state, player, action, float("inf"))
        position, result = native.pack(state), native.Position()
        score = ct.c_double()
        encoded = _encode_action(action)
        status = self.lib.president_search_rollout(ct.byref(position), state.players.index(player),
            ct.byref(encoded), native.weights(agent), ct.byref(native.options(agent)),
            ct.byref(result), ct.byref(score))
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
        self.assertAlmostEqual(score.value, evaluate(agent, expected, player), places=11)
        return expected, score.value

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
                         ct.byref(action), 1, native.weights(self.agent), None, -1, scores), -1)

    def test_abi_version_and_option_validation(self):
        self.assertEqual(self.lib.president_search_version(), native.ABI_VERSION)
        state = Round(list("abcd"), first_round=True, rng=random.Random(42))
        position = native.pack(state)
        action = _encode_action(state.legal_actions(state._actor())[0])
        scores = (ct.c_double * 1)()
        seat = state.players.index(state._actor())
        for bad in (native.Options(16, 2, 1.0), native.Options(16, 0, -1.0),
                    native.Options(16, 0, float("nan")), native.Options(16, 0, float("inf"))):
            self.assertEqual(self.lib.president_search_batch(ct.byref(position), seat, ct.byref(action), 1,
                             native.weights(self.agent), ct.byref(bad), -1, scores), -1)
        # NULL options are the defaults.
        default = (ct.c_double * 1)()
        self.assertEqual(self.lib.president_search_batch(ct.byref(position), seat, ct.byref(action), 1,
                         native.weights(self.agent), None, -1, default), 1)
        self.assertEqual(self.lib.president_search_batch(ct.byref(position), seat, ct.byref(action), 1,
                         native.weights(self.agent), ct.byref(native.Options(16, 0, 1.0)), -1, scores), 1)
        self.assertEqual(list(default), list(scores))
        options = native.options(self.agent)
        self.assertEqual((options.depth, options.through_own_turns, options.leaf_scale), (16, 0, 1.0))
        options = native.options(self.option_agent(search_depth=None, search_through_own_turns=True))
        self.assertEqual((options.depth, options.through_own_turns), (-1, 1))

    OPTION_SETS = ({"search_depth": None, "search_through_own_turns": True},
                   {"search_depth": 4},
                   {"search_depth": 4, "search_through_own_turns": True},
                   {"search_depth": None},
                   {"search_leaf_scale": 0.35},
                   {"search_leaf_scale": 0.0, "search_depth": 7, "search_through_own_turns": True})

    def option_agent(self, **options):
        return ExperiencedAgent4(random.Random(42), search_time_ms=None, search_worlds=2,
                                 search_backend="python", evaluation_backend="python", **options)

    def test_rollout_options_match_across_complete_games(self):
        agents = [self.option_agent(**options) for options in self.OPTION_SETS]
        checked = 0
        for seed in range(3):
            state = Round(list("abcd"), first_round=seed != 1,
                          roles=None if seed != 1 else dict(zip("abcd", (1, 2, 3, 4))),
                          rng=random.Random(100 + seed))
            basic = BasicAgent()
            while state.phase in {"draft", "trade_request", "trade_return"}:
                p = state._actor()
                state.apply_action(p, basic.respond(state.message_for(p)))
            step = 0
            while not state.done:
                player = state._actor()
                if step % 7 == 0:
                    actions = unique_actions(state.legal_actions(player))
                    for index, agent in enumerate(agents):
                        with self.subTest(seed=seed, step=step, options=self.OPTION_SETS[index]):
                            for action in actions:
                                self.check_rollout(state, action, agent)
                                checked += 1
                state.apply_action(player, heuristic_action(self.agent, state))
                step += 1
        self.assertGreater(checked, 500)

    def test_batch_options_match_python_on_sampled_worlds(self):
        for options in self.OPTION_SETS:
            agent = self.option_agent(**options)
            state = Round(list("abcd"), first_round=True, rng=random.Random(3))
            for step in range(60):
                if state.done:
                    break
                player = state._actor()
                if step % 6 == 0:
                    message = state.message_for(player)
                    world = sample_world(prepare(agent, message), player, random.Random(step))
                    if not any(all(c.rank == "2" for c in world.hands[p]) for p in world.active):
                        actions = unique_actions(world.legal_actions(player))
                        self.assertIsNone(native.batch(self.lib, agent, world, player, actions, 0))
                        scores = native.batch(self.lib, agent, world, player, actions, -1)
                        expected = [evaluate(agent, rollout(agent, world, player, a, float("inf")), player)
                                    for a in actions]
                        for got, want in zip(scores, expected):
                            self.assertAlmostEqual(got, want, places=11, msg=f"{options} step={step}")
                state.apply_action(player, heuristic_action(self.agent, state))

    def test_full_rollout_value_is_fraction_of_seats_beaten(self):
        agent = self.option_agent(search_depth=None, search_through_own_turns=True)
        seen = {"exited": 0, "finished": 0, "bomb": 0}
        for seed in range(6):
            state = Round(list("abcd"), first_round=True, rng=random.Random(200 + seed))
            step = 0
            while not state.done:
                player = state._actor()
                if step % 5 == 0:
                    for action in unique_actions(state.legal_actions(player)):
                        final, value = self.check_rollout(state, action, agent)
                        self.assertTrue(final.done or player not in final.active)
                        # Play the remainder out to compute our true placement.
                        rest = clone(final)
                        while not rest.done:
                            rest.apply_action(rest._actor(), heuristic_action(agent, rest))
                        beaten = sum(rest.rankings[player] < rest.rankings[q]
                                     for q in rest.players if q != player)
                        self.assertEqual(value, beaten / 3)
                        self.assertEqual(evaluate(agent, final, player), beaten / 3)
                        seen["finished" if final.done else "exited"] += 1
                        seen["bomb"] += player in final.bomb_finishers
                state.apply_action(player, heuristic_action(self.agent, state))
                step += 1
        self.assertTrue(seen["exited"] and seen["finished"] and seen["bomb"], seen)

    def test_fixed_work_decisions_match_with_options(self):
        state = Round(list("abcd"), first_round=True, rng=random.Random(5))
        options = {"search_depth": None, "search_through_own_turns": True, "search_leaf_scale": 0.5,
                   "search_candidates": 2, "search_margin": 0.5}
        compared = searched = 0
        for step in range(80):
            if state.done:
                break
            p = state._actor()
            message = state.message_for(p)
            actions = message["request"]["legal_actions"]
            if step % 3 == 0 and state.phase == "play":
                py = ExperiencedAgent4(random.Random(9), search_time_ms=None, search_worlds=3,
                                       search_backend="python", evaluation_backend="python", **options)
                c = ExperiencedAgent4(random.Random(9), search_time_ms=None, search_worlds=3,
                                      search_backend="c", **options)
                self.assertEqual(py._choose_play(message, actions), c._choose_play(message, actions))
                for key in ("worlds", "overrode", "candidates", "reason"):
                    self.assertEqual(py.last_search[key], c.last_search[key])
                if c.last_search["reason"] == "search":
                    self.assertEqual(c.last_search["backend"], "c")
                    searched += 1
                for key in ("heuristic_score", "chosen_score"):
                    if py.last_search[key] is None:
                        self.assertIsNone(c.last_search[key])
                    else:
                        self.assertAlmostEqual(py.last_search[key], c.last_search[key], places=11)
                compared += 1
            state.apply_action(p, heuristic_action(self.agent, state))
        self.assertGreater(compared, 3)
        self.assertGreater(searched, 1)


if __name__ == "__main__":
    unittest.main()
