import random
import unittest
from copy import deepcopy
from unittest.mock import patch

from src.agent import ExperiencedAgent4
from src.agent_search import (prepare, sample_world, clone, rollout, evaluate,
                              heuristic_action, unique_actions, action_key)
from src.president import Round, standard_deck
from src.snapshots import freeze
from tests.test_president import ready_round
from src.president import Card


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.agent = ExperiencedAgent4(random.Random(5), search_time_ms=None, search_worlds=2, search_backend="python")
        self.state = Round(list("abcd"), first_round=True, rng=random.Random(42))

    def test_fixed_work_legal_immutable_and_repeatable(self):
        message = self.state.message_for(self.state._actor())
        before = deepcopy(message)
        other = ExperiencedAgent4(random.Random(5), search_time_ms=None, search_worlds=2, search_backend="python")
        with patch("src.agent.solve_round", side_effect=AssertionError("recursive solver")):
            result = self.agent._choose_play(freeze(message), message["request"]["legal_actions"])
            self.assertEqual(result, other._choose_play(message, message["request"]["legal_actions"]))
        self.assertIn(result, message["request"]["legal_actions"])
        self.assertEqual(message, before)
        self.assertEqual(self.agent.last_search["worlds"], 2)

    def test_sample_conservation_and_reconstruction_through_game(self):
        state = self.state
        for _ in range(1500):
            if state.done:
                break
            actor = state._actor()
            message = state.message_for(actor)
            prepared = prepare(self.agent, message)
            if not any(e["type"] == "twos_announced" for e in state.events):
                self.assertIsNotNone(prepared)
                world = sample_world(prepared, actor, self.agent.rng)
                self.assertEqual(world.completion_order, state.completion_order if state.phase == "completion" else [])
                self.assertEqual([action_key(a) for a in world.legal_actions(actor)],
                                 [action_key(a) for a in state.legal_actions(actor)])
                cards = [c for hand in world.hands.values() for c in hand]
                cards += self.agent._publicly_played(message)
                self.assertEqual(len(cards), 52)
                self.assertEqual(set(cards), set(standard_deck()))
                self.assertEqual({p: len(h) for p, h in world.hands.items()},
                                 {p: len(h) for p, h in state.hands.items()})
                self.assertEqual(world.hands[actor], state._sorted_hand(actor))
            state.apply_action(actor, heuristic_action(self.agent, state))
        self.assertTrue(state.done)

    def test_no_private_hand_leak_to_reply_policy(self):
        original = self.agent._play_scorer
        seen = []
        def scorer(message):
            self.assertNotIn("hands", message)
            self.assertEqual(message["hand"], [c.json() for c in self.state._sorted_hand(message["player"])])
            seen.append(message["player"])
            return original(message)
        with patch.object(self.agent, "_play_scorer", side_effect=scorer):
            heuristic_action(self.agent, self.state)
        self.assertTrue(seen)

    def test_budget_and_incomplete_batch_fallback(self):
        message = self.state.message_for(self.state._actor())
        actions = message["request"]["legal_actions"]
        expected = min(actions, key=self.agent._play_scorer(message))
        self.agent.search_time_ms = 0
        self.assertEqual(self.agent._choose_play(message, actions), expected)
        self.agent.search_time_ms = None
        with patch("src.agent_search.rollout", side_effect=[clone(self.state), None]):
            self.assertEqual(self.agent._choose_play(message, actions), expected)
        self.assertEqual(self.agent.last_search["worlds"], 0)

    def test_world_shared_and_original_action_returned(self):
        message = self.state.message_for(self.state._actor())
        actions = message["request"]["legal_actions"]
        worlds = []
        def simulate(agent, world, player, action, deadline):
            worlds.append(world)
            return clone(world)
        with patch("src.agent_search.rollout", side_effect=simulate):
            result = self.agent._choose_play(message, actions)
        n = self.agent.last_search["candidates"]
        self.assertTrue(all(w is worlds[0] for w in worlds[:n]))
        self.assertIsNot(worlds[0], worlds[n])
        self.assertTrue(any(result is a for a in actions))

    def test_fallback_invalid_history(self):
        message = self.state.message_for(self.state._actor())
        for events in ([], message["events"] + [{"type": "twos_announced", "player": "b"}]):
            message["events"] = events
            self.assertIsNone(prepare(self.agent, message))

    def test_all_twos_sample_is_retried_before_either_backend(self):
        message = self.state.message_for(self.state._actor())
        actions = message["request"]["legal_actions"]
        invalid = clone(self.state)
        opponent = next(p for p in invalid.active if p != message["player"])
        invalid.hands[opponent] = [Card("2", "H")]
        for native in (None, object()):
            with patch("src.agent_search_c.backend", return_value=native), \
                 patch("src.agent_search.sample_world", return_value=invalid) as sample, \
                 patch("src.agent_search.rollout") as simulate, \
                 patch("src.agent_search_c.batch") as batch:
                result = self.agent._choose_play(message, actions)
                self.assertIn(result, actions)
                self.assertTrue(self.agent.last_search["fallback"])
                self.assertEqual(sample.call_count, self.agent.search_worlds + 3)
                self.assertEqual(self.agent.last_search["rejected_worlds"], sample.call_count)
                simulate.assert_not_called()
                batch.assert_not_called()

    def test_native_rejection_retries_fresh_world_then_succeeds(self):
        from src.agent_search_c import NativeSearchRejected
        message = self.state.message_for(self.state._actor())
        actions = message["request"]["legal_actions"]
        seen = []
        def batch(lib, agent, world, player, candidates, remaining):
            seen.append(world)
            if len(seen) == 1:
                raise NativeSearchRejected("test rejection")
            return [0.5] * len(candidates)
        with patch("src.agent_search_c.backend", return_value=object()), \
             patch("src.agent_search_c.batch", side_effect=batch):
            self.assertIn(self.agent._choose_play(message, actions), actions)
        self.assertEqual(self.agent.last_search["worlds"], 2)
        self.assertEqual(self.agent.last_search["native_errors"], 1)
        self.assertFalse(self.agent.last_search["fallback"])
        self.assertEqual(len({id(w) for w in seen}), 3)

    def test_persistent_native_rejection_has_bounded_legal_fallback(self):
        from src.agent_search_c import NativeSearchRejected
        message = self.state.message_for(self.state._actor())
        actions = message["request"]["legal_actions"]
        expected = min(unique_actions(actions), key=self.agent._play_scorer(message))
        with patch("src.agent_search_c.backend", return_value=object()), \
             patch("src.agent_search_c.batch", side_effect=NativeSearchRejected("test")) as batch:
            self.assertEqual(self.agent._choose_play(message, actions), expected)
        self.assertEqual(batch.call_count, self.agent.search_worlds + 3)
        self.assertTrue(self.agent.last_search["fallback"])
        self.assertEqual(self.agent.last_search["worlds"], 0)

    def test_rollout_stops_at_normal_turn_after_completions(self):
        player = self.state._actor()
        action = self.state.legal_actions(player)[0]
        result = rollout(self.agent, self.state, player, action, float("inf"))
        self.assertTrue(result.done or player not in result.active or
                        (result.phase == "play" and result._actor() == player))
        self.assertEqual(len(self.state.events), 1)

    def test_finish_order_and_penalties_override_hand_scores(self):
        state = clone(self.state)
        state.finish_order = ["a"]
        state.bomb_finishers = ["d"]
        state.active = {"b", "c"}
        self.assertEqual(evaluate(self.agent, state, "a"), 1.0)
        self.assertEqual(evaluate(self.agent, state, "d"), 0.0)
        state.rankings = {"a": 1, "b": 2, "c": 3, "d": 4}
        self.assertAlmostEqual(evaluate(self.agent, state, "b"), 2 / 3)

    def test_suit_duplicates_keep_first(self):
        actions = [{"type": "play", "cards": [{"rank": "3", "suit": s}]} for s in "CDHS"]
        self.assertEqual(unique_actions(actions), actions[:1])

    def test_clone_transitions_match_engine_for_special_rules(self):
        state = ready_round({"a": [Card("7", "C"), Card("2", "C"), Card("3", "C")],
                             "b": [Card("7", s) for s in "DHS"],
                             "c": [Card("8", "C"), Card("4", "C")]})
        state.top_rank, state.top_count, state.top_stack_count = "7", 1, 1
        state.last_player = "c"
        completion = deepcopy(state)
        completion._offer_completions("a")
        quad = ready_round({"a": [Card("9", s) for s in "CDHS"] + [Card("3", "C")],
                            "b": [Card("4", "C")], "c": [Card("5", "C")]})
        bomb_finish = deepcopy(state)
        bomb_finish.hands["a"] = [Card("2", "C")]
        for position in (state, completion, quad, bomb_finish):
            actor = position._actor()
            for action in position.legal_actions(actor):
                expected, actual = deepcopy(position), clone(position)
                expected.apply_action(actor, action)
                actual.apply_action(actor, action)
                self.assertEqual(actual.__dict__, expected.__dict__)

    def test_bomb_returns_control_and_preserves_forced_lead(self):
        state = ready_round({"a": [Card("2", "C"), Card("3", "C")],
                             "b": [Card("4", "C")], "c": [Card("5", "C")]})
        state.top_rank, state.top_count, state.top_stack_count = "7", 1, 1
        state.last_player = "c"
        result = rollout(self.agent, state, "a", {"type": "bomb", "card": Card("2", "C").json()}, float("inf"))
        self.assertEqual(result.phase, "play")
        self.assertEqual(result._actor(), "a")
        self.assertIsNone(result.top_rank)
        self.assertTrue(all(a["type"] == "play" for a in result.legal_actions("a")))

    def test_option_validation(self):
        good = ExperiencedAgent4(search_depth=None, search_through_own_turns=True, search_leaf_scale=0,
                                 search_candidates=1, search_margin=2)
        self.assertEqual((good.search_depth, good.search_through_own_turns, good.search_leaf_scale,
                          good.search_candidates, good.search_margin), (None, True, 0.0, 1, 2.0))
        default = ExperiencedAgent4()
        self.assertEqual((default.search_depth, default.search_through_own_turns, default.search_leaf_scale,
                          default.search_candidates, default.search_margin), (16, False, 1.0, 4, 0.0))
        for name, values in {"search_depth": (-1, 1.5, True, "4"),
                             "search_through_own_turns": (1, None, "yes"),
                             "search_leaf_scale": (-0.1, float("nan"), float("inf"), True, "1"),
                             "search_candidates": (0, -2, 2.0, True, None),
                             "search_margin": (-1, float("nan"), float("inf"), False, None)}.items():
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    ExperiencedAgent4(**{name: value})

    def scripted_choice(self, rows, **options):
        """Run choose with native batches returning rows[w] for world w."""
        agent = ExperiencedAgent4(random.Random(5), search_time_ms=None, search_worlds=len(rows),
                                  search_backend="python", **options)
        message = self.state.message_for(self.state._actor())
        actions = message["request"]["legal_actions"]
        calls = []
        def batch(lib, agent_, world, player, candidates, remaining):
            row = rows[len(calls)]
            calls.append(candidates)
            return row + [0.0] * (len(candidates) - len(row))
        with patch("src.agent_search_c.backend", return_value=object()), \
             patch("src.agent_search_c.batch", side_effect=batch):
            result = agent._choose_play(message, actions)
        return result, calls[0], agent.last_search

    def test_margin_paired_standard_error_rule(self):
        # Candidate 1 beats the heuristic choice by 0.1 in every world: SE 0.
        steady = [[0.5, 0.6]] * 3
        for margin in (0, 1e9):
            result, candidates, stats = self.scripted_choice(steady, search_margin=margin)
            self.assertIs(result, candidates[1])
            self.assertTrue(stats["overrode"])
            self.assertAlmostEqual(stats["heuristic_score"], 0.5)
            self.assertAlmostEqual(stats["chosen_score"], 0.6)
        # d = (0.4, -0.2, 0.1): mean 0.1, SE = 0.3/sqrt(3) ~ 0.173.
        noisy = [[0.1, 0.5], [0.6, 0.4], [0.3, 0.4]]
        for margin, overrides in ((0, True), (0.5, True), (0.6, False), (5, False)):
            with self.subTest(margin=margin):
                result, candidates, stats = self.scripted_choice(noisy, search_margin=margin)
                self.assertIs(result, candidates[1] if overrides else candidates[0])
                self.assertEqual(stats["overrode"], overrides)
                self.assertAlmostEqual(stats["heuristic_score"], 1.0 / 3)
                self.assertAlmostEqual(stats["chosen_score"], 1.3 / 3 if overrides else 1.0 / 3)
        # One world cannot establish significance; ties keep heuristic order.
        result, candidates, stats = self.scripted_choice([[0.1, 0.9]], search_margin=0.01)
        self.assertIs(result, candidates[0])
        result, candidates, stats = self.scripted_choice([[0.5, 0.5]] * 2)
        self.assertIs(result, candidates[0])
        self.assertFalse(stats["overrode"])

    def test_margin_on_real_search(self):
        message = self.state.message_for(self.state._actor())
        actions = message["request"]["legal_actions"]
        heuristic = min(unique_actions(actions), key=self.agent._play_scorer(message))
        for seed in range(4):
            plain = ExperiencedAgent4(random.Random(seed), search_time_ms=None, search_worlds=4,
                                      search_backend="python")
            zero = ExperiencedAgent4(random.Random(seed), search_time_ms=None, search_worlds=4,
                                     search_backend="python", search_margin=0.0)
            strict = ExperiencedAgent4(random.Random(seed), search_time_ms=None, search_worlds=4,
                                       search_backend="python", search_margin=1e6)
            self.assertEqual(plain._choose_play(message, actions), zero._choose_play(message, actions))
            self.assertEqual(plain.last_search["overrode"], zero.last_search["overrode"])
            self.assertEqual(strict._choose_play(message, actions), heuristic)
            self.assertFalse(strict.last_search["overrode"])
            self.assertEqual(strict.last_search["reason"], "search")
            self.assertIsNotNone(strict.last_search["heuristic_score"])
            self.assertEqual(strict.last_search["heuristic_score"], strict.last_search["chosen_score"])

    def test_candidate_count(self):
        message = self.state.message_for(self.state._actor())
        actions = message["request"]["legal_actions"]
        ranked = sorted(unique_actions(actions), key=self.agent._play_scorer(message))
        for width in (1, 2, 6):
            agent = ExperiencedAgent4(random.Random(5), search_time_ms=None, search_worlds=1,
                                      search_backend="python", search_candidates=width)
            agent._choose_play(message, actions)
            expected = sum(i < width or a["type"] in {"pass", "decline_completion", "bomb"}
                           for i, a in enumerate(ranked))
            self.assertEqual(agent.last_search["candidates"], expected)

    def test_fallback_reports_no_scores(self):
        message = self.state.message_for(self.state._actor())
        actions = message["request"]["legal_actions"]
        self.agent.search_time_ms = 0
        self.agent._choose_play(message, actions)
        self.assertFalse(self.agent.last_search["overrode"])
        self.assertIsNone(self.agent.last_search["heuristic_score"])
        self.assertIsNone(self.agent.last_search["chosen_score"])

    def test_rollout_depth_and_own_turn_options(self):
        player = self.state._actor()
        action = self.state.legal_actions(player)[0]
        def normal_actions(result):
            return sum(e["type"] in {"played", "opening_play", "passed", "bombed"}
                       for e in result.events[len(self.state.events):])
        for depth in (1, 4, 30):
            agent = ExperiencedAgent4(search_backend="python", search_depth=depth,
                                      search_through_own_turns=True)
            result = rollout(agent, self.state, player, action, float("inf"))
            self.assertLessEqual(normal_actions(result), depth)
            self.assertTrue(result.done or player not in result.active or normal_actions(result) == depth)
        full = ExperiencedAgent4(search_backend="python", search_depth=None, search_through_own_turns=True)
        result = rollout(full, self.state, player, action, float("inf"))
        self.assertTrue(result.done or player not in result.active)
        self.assertEqual(len(self.state.events), 1)

    def test_leaf_scale(self):
        state = clone(self.state)
        flat = ExperiencedAgent4(search_backend="python", search_leaf_scale=0)
        self.assertEqual(evaluate(flat, state, "a"), 0.5)
        unit = ExperiencedAgent4(search_backend="python", search_leaf_scale=1)
        self.assertEqual(evaluate(unit, state, "a"), evaluate(self.agent, state, "a"))
        # A single unresolved pair: scaling sharpens the sigmoid around 0.5.
        state.finish_order, state.bomb_finishers, state.active = ["c"], ["d"], {"a", "b"}
        values = [evaluate(ExperiencedAgent4(search_backend="python", search_leaf_scale=s), state, "a")
                  for s in (0, 1, 3)]
        pair = [3 * v - 1 for v in values]  # beat c is 0, beat d is 1
        self.assertEqual(pair[0], 0.5)
        self.assertNotEqual(pair[1], 0.5)
        self.assertGreater(abs(pair[2] - 0.5), abs(pair[1] - 0.5))

    def test_immediate_finish_does_not_sample(self):
        message = {"hand": [{"rank": "3", "suit": "C"}]}
        actions = [{"type": "pass"}, {"type": "play", "cards": message["hand"]}]
        with patch("src.agent_search.prepare", side_effect=AssertionError("unnecessary sampling")):
            self.assertIs(self.agent._choose_play(message, actions), actions[1])


if __name__ == "__main__":
    unittest.main()
