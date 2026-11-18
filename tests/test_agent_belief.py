import random
import unittest
from collections import Counter

from src.agent import ExperiencedAgent3Tuned, ExperiencedAgent4, HumanRulesAgent3
from src.agent_belief import (BOMB, PASS, TWO, BeliefSampler, extract, log_likelihood,
                              rules_completes, rules_play, rules_trade)
from src.agent_search import prepare, uniform_slot
from src.president import RANK_VALUE, RANKS, Card, Round, standard_deck
from src.simulate import PLAYERS


def key(action):
    if action["type"] == "pass":
        return PASS
    if action["type"] == "bomb":
        return BOMB
    return "play", RANK_VALUE[action["cards"][0]["rank"]], len(action["cards"])


def later_round(seed, roles=None):
    roles = roles or dict(zip(PLAYERS, (1, 2, 3, 4)))
    return Round(PLAYERS, first_round=False, roles=roles, rng=random.Random(seed))


def play_until(state, clients, stop):
    """Advance with ``clients`` until ``stop(state, actor, message)`` is true."""
    while not state.done:
        actor = state._actor()
        message = state.message_for(actor)
        if stop(state, actor, message):
            return actor, message
        state.apply_action(actor, clients[actor].respond(message))
    return None, None


class RulesPortTests(unittest.TestCase):
    """The rank-level model must equal HumanRulesAgent3 exactly."""

    def test_play_matches_human_rules_agent(self):
        agent = HumanRulesAgent3(heads_up_solver=False)
        rng = random.Random(3)
        deck = standard_deck()
        for _ in range(3000):
            hand = rng.sample(deck, rng.choice([1, 2, 3, 5, 8, 13]))
            counts = Counter(c.rank for c in hand)
            smallest = rng.choice([1, 2, 3, 6, 13])
            top = None if rng.random() < 0.4 else {
                "rank": RANKS[rng.randrange(TWO)], "count": rng.choice([1, 1, 2, 3]), "stack_count": 1}
            actions = []
            for rank in counts:
                if rank == "2":
                    continue
                cards = [c.json() for c in hand if c.rank == rank]
                if top is None:
                    actions += [{"type": "play", "cards": cards[:k]} for k in range(1, len(cards) + 1)]
                elif RANK_VALUE[rank] >= RANK_VALUE[top["rank"]] and len(cards) >= top["count"]:
                    actions.append({"type": "play", "cards": cards[:top["count"]]})
            if top is not None:
                actions += [{"type": "bomb", "card": c.json()} for c in hand if c.rank == "2"][:1]
                actions.append({"type": "pass"})
            if not actions:
                continue
            agent._smallest_opponent = lambda message, s=smallest: s
            expected = key(agent._play({"hand": [c.json() for c in hand], "top": top}, actions))
            board = None if top is None else (RANK_VALUE[top["rank"]], top["count"])
            self.assertEqual(rules_play([counts[r] for r in RANKS], board, smallest), expected)

    def test_trade_matches_human_rules_agent(self):
        agent = HumanRulesAgent3()
        rng = random.Random(4)
        actions = [{"type": "trade_request", "rank": r} for r in RANKS]
        for _ in range(1000):
            hand = rng.sample(standard_deck(), 13)
            failed = set(rng.sample(RANKS, rng.choice([0, 1, 2])))
            message = {"hand": [c.json() for c in hand], "player": "a", "events": [{"type": "round_started"}] + [
                {"type": "trade_requested", "initiator": "a", "success": False, "rank": r} for r in failed]}
            counts = Counter(c.rank for c in hand)
            rank = rules_trade([counts[r] for r in RANKS], frozenset(RANK_VALUE[r] for r in failed))
            self.assertEqual(RANKS[rank], agent._trade_rank(message, actions)["rank"])

    def test_completion_rule(self):
        c = [0] * len(RANKS)
        c[RANK_VALUE["9"]], c[RANK_VALUE["4"]] = 2, 1
        self.assertTrue(rules_completes(c, RANK_VALUE["9"], 2))
        self.assertFalse(rules_completes(c, RANK_VALUE["9"], 1))
        c[TWO] = 1  # afterwards B=1 >= G=1: decline
        self.assertFalse(rules_completes(c, RANK_VALUE["9"], 2))


class EvidenceTests(unittest.TestCase):
    def test_failed_request_draft_pin_and_passes(self):
        state = later_round(11)
        clients = {p: HumanRulesAgent3() for p in PLAYERS}
        # Stop at our first play decision with some history behind it.
        actor, message = play_until(state, clients, lambda s, a, m: a == "a" and m["request"]["type"] == "play"
                                    and any(e["type"] == "passed" for e in m["events"]))
        self.assertIsNotNone(message)
        opponents = [p for p in message["active_players"] if p != "a"]
        observations, pins = extract(message, "a", opponents)
        events = message["events"]
        for e in events:
            if e["type"] == "trade_requested" and not e["success"] and e["receiver"] in opponents:
                self.assertIn((e["receiver"], RANK_VALUE[e["rank"]]),
                              {(p, d[0]) for p in opponents for _, k, d in observations[p] if k == "none"})
        # Pins name cards the drafter really holds (they are unplayed, unseen).
        for card, p in pins.items():
            if card not in {c for h in state.hands.values() for c in h}:
                continue
            self.assertIn(card, state.hands[p])
        # True hands explain all play evidence: the opponents follow the rules exactly.
        for p in opponents:
            counts = Counter(c.rank for c in state.hands[p])
            vec = [counts[r] for r in RANKS]
            plays = [(o, k, d) for o, k, d in observations[p] if k in {"play", "nocomplete", "opening"}]
            self.assertTrue(plays or not any(e.get("player") == p for e in events))
            self.assertEqual(log_likelihood(vec, plays, 0.2), 0.0)

    def test_returned_card_is_recorded_and_pinned(self):
        state = later_round(5)
        clients = {p: ExperiencedAgent4(random.Random(1), search_worlds=0) for p in PLAYERS}
        president = next(p for p, r in state.roles.items() if r == 1)
        actor, message = play_until(state, clients, lambda s, a, m: s.phase == "play" and a == president)
        returned = clients[president].returned_cards(message)
        self.assertEqual(len(returned), 2)
        scum = next(p for p, r in state.roles.items() if r == 4)
        self.assertTrue(all(receiver == scum for _, receiver in returned))
        _, pins = extract(message, president, [p for p in PLAYERS if p != president], returned)
        # A returned rank the receiver later gives back may have moved: no pin.
        gave = {e["rank"] for e in message["events"] if e["type"] == "trade_requested"
                and e["receiver"] == scum and e["success"]}
        for card, _ in returned:
            c = Card.from_json(card)
            if c.rank in gave:
                self.assertNotIn(c, pins)
            elif c in state.hands[scum]:
                self.assertEqual(pins[c], scum)

    def test_returned_card_given_back_is_not_pinned(self):
        # Seed 5: the President returns the 3C, then asks for (and receives) a 3.
        state = later_round(5)
        clients = {p: ExperiencedAgent4(random.Random(1), search_worlds=0) for p in PLAYERS}
        _, message = play_until(state, clients, lambda s, a, m: s.phase == "play" and a == "a")
        returned = clients["a"].returned_cards(message)
        self.assertIn(({"rank": "3", "suit": "C"}, "d"), returned)
        _, pins = extract(message, "a", ["b", "c", "d"], returned)
        self.assertNotIn(Card("3", "C"), pins)


class SamplerTests(unittest.TestCase):
    def test_worlds_are_consistent_deals(self):
        state = later_round(21)
        clients = {p: ExperiencedAgent3Tuned() for p in PLAYERS}
        agent = ExperiencedAgent4(random.Random(2), search_worlds=0)
        checked = 0
        while not state.done and checked < 6:
            actor = state._actor()
            message = state.message_for(actor)
            if (actor == "b" and message["request"]["type"] == "play" and len(message["active_players"]) > 2):
                prepared = prepare(agent, message)
                sampler = BeliefSampler(prepared, "b", message, random.Random(checked))
                for card, p in [(c, p) for p, cards in sampler.pinned.items() for c in cards]:
                    self.assertIn(card, prepared[1])
                for _ in range(5):
                    world = sampler.sample()
                    dealt = [c for p in sampler.opponents for c in world.hands[p]]
                    self.assertEqual(sorted(dealt, key=repr), sorted(prepared[1], key=repr))
                    for p in sampler.opponents:
                        self.assertEqual(len(world.hands[p]), prepared[2][p])
                        self.assertTrue(set(sampler.pinned[p]) <= set(world.hands[p]))
                    self.assertEqual(world.hands["b"], state._sorted_hand("b"))
                checked += 1
            state.apply_action(actor, (agent if actor == "b" else clients[actor]).respond(message))
        self.assertGreater(checked, 0)

    def test_uniform_slots(self):
        self.assertEqual([i for i in range(12) if uniform_slot(i, 0.25)], [3, 7, 11])
        self.assertFalse(any(uniform_slot(i, 0.0) for i in range(20)))
        self.assertTrue(all(uniform_slot(i, 1.0) for i in range(20)))


class AgentSwitchTests(unittest.TestCase):
    def game(self, **kwargs):
        state = later_round(9)
        agents = {p: ExperiencedAgent4(random.Random(i), search_time_ms=None, search_worlds=4, **kwargs)
                  for i, p in enumerate(PLAYERS)}
        choices, samplings = [], set()
        while not state.done:
            actor = state._actor()
            action = agents[actor].respond(state.message_for(actor))
            samplings.add(agents[actor].last_search.get("sampling"))
            choices.append(action)
            state.apply_action(actor, action)
        return choices, samplings

    def test_uniform_is_the_default_and_unchanged(self):
        default, _ = self.game()
        explicit, samplings = self.game(search_sampling="uniform")
        self.assertEqual(default, explicit)
        self.assertEqual(samplings - {None}, {"uniform"})

    def test_modeled_is_repeatable_on_both_backends(self):
        for backend in ("python", "auto"):
            first, samplings = self.game(search_sampling="modeled", search_backend=backend)
            self.assertEqual(first, self.game(search_sampling="modeled", search_backend=backend)[0])
            self.assertIn("modeled", samplings)

    def test_validation(self):
        for kwargs in ({"search_sampling": "smart"}, {"search_random_fraction": 1.5},
                       {"search_random_fraction": -0.1}, {"search_model_noise": 0},
                       {"search_model_noise": True}):
            with self.assertRaises(ValueError):
                ExperiencedAgent4(**kwargs)


if __name__ == "__main__":
    unittest.main()


class NativeBeliefTests(unittest.TestCase):
    def setUp(self):
        from src import agent_search_c
        try:
            self.lib = agent_search_c.backend(ExperiencedAgent4(search_backend="c"))
        except RuntimeError as error:
            self.skipTest(str(error))

    def c_loglik(self, counts, observations, noise):
        import ctypes as ct
        from src.agent_belief import NATIVE_KIND, native_data
        from src.agent_search_c import BeliefObservation
        array = (BeliefObservation * max(1, len(observations)))()
        for item, (offset, kind, data) in zip(array, observations):
            item.kind = NATIVE_KIND[kind]
            item.offset[:] = offset
            item.data[:] = native_data(kind, data)
        out = ct.c_double()
        status = self.lib.president_belief_loglik((ct.c_int * 13)(*counts), array, len(observations),
                                                  noise, ct.byref(out))
        self.assertEqual(status, 1)
        return out.value

    def test_rule_predictions_match_python(self):
        # The observed action is Python's prediction, so C must score a match.
        from src.agent_belief import rules_trade
        rng = random.Random(8)
        zero = (0,) * 13
        for _ in range(20000):
            counts = [0] * 13
            for card in rng.sample(range(52), rng.choice([1, 2, 3, 4, 6, 9, 13])):
                counts[card // 4] += 1
            if not any(counts[:TWO]):
                continue  # an all-2s hand scores ALL_TWOS in both, not a match
            top = None if rng.random() < 0.4 else (rng.randrange(TWO), rng.choice([1, 1, 2, 3]))
            smallest = rng.choice([1, 2, 3, 5, 13])
            failed = frozenset(rng.sample(range(13), rng.choice([0, 1, 2])))
            rank, needed = rng.randrange(TWO), rng.randint(1, 3)
            observations = [(zero, "play", (top, smallest, rules_play(counts, top, smallest))),
                            (zero, "trade", (failed, rules_trade(counts, failed)))]
            self.assertEqual(self.c_loglik(counts, observations, 0.2), 0.0)
            window = [(zero, "nocomplete", (rank, needed))]
            self.assertEqual(self.c_loglik(counts, window, 0.2), log_likelihood(counts, window, 0.2))

    def test_scores_match_python_on_game_evidence(self):
        state = later_round(31)
        clients = {p: ExperiencedAgent3Tuned() for p in PLAYERS}
        rng = random.Random(9)
        checked = 0
        while not state.done:
            actor = state._actor()
            message = state.message_for(actor)
            if message["request"]["type"] == "play" and len(message["active_players"]) > 2:
                opponents = [p for p in message["active_players"] if p != actor]
                observations, _ = extract(message, actor, opponents)
                prepared = prepare(ExperiencedAgent4(), message)
                for p in opponents:
                    hand = rng.sample(prepared[1], prepared[2][p])
                    counts = Counter(c.rank for c in hand)
                    vec = [counts[r] for r in RANKS]
                    self.assertAlmostEqual(self.c_loglik(vec, observations[p], 0.2),
                                           log_likelihood(vec, observations[p], 0.2), places=12)
                    checked += 1
            state.apply_action(actor, clients[actor].respond(message))
        self.assertGreater(checked, 20)

    def test_native_walk_deals_consistent_worlds(self):
        state = later_round(21)
        clients = {p: ExperiencedAgent3Tuned() for p in PLAYERS}
        agent = ExperiencedAgent4(random.Random(2), search_worlds=0)
        checked = 0
        while not state.done and checked < 6:
            actor = state._actor()
            message = state.message_for(actor)
            if actor == "b" and message["request"]["type"] == "play" and len(message["active_players"]) > 2:
                prepared = prepare(agent, message)
                sampler = BeliefSampler(prepared, "b", message, random.Random(checked), native=self.lib)
                for _ in range(5):
                    world = sampler.sample()
                    dealt = [c for p in sampler.opponents for c in world.hands[p]]
                    self.assertEqual(sorted(dealt, key=repr), sorted(prepared[1], key=repr))
                    for p in sampler.opponents:
                        self.assertEqual(len(world.hands[p]), prepared[2][p])
                        self.assertTrue(set(sampler.pinned[p]) <= set(world.hands[p]))
                self.assertGreater(sampler.steps, 0)
                checked += 1
            state.apply_action(actor, (agent if actor == "b" else clients[actor]).respond(message))
        self.assertGreater(checked, 0)

    def test_invalid_inputs_are_rejected(self):
        import ctypes as ct
        from src.agent_search_c import BeliefObservation
        out = ct.c_double()
        bad = (BeliefObservation * 1)()
        bad[0].kind = 9
        self.assertEqual(self.lib.president_belief_loglik((ct.c_int * 13)(), bad, 1, 0.2, ct.byref(out)), -1)
        self.assertEqual(self.lib.president_belief_loglik((ct.c_int * 13)(), bad, 0, 0.0, ct.byref(out)), -1)
