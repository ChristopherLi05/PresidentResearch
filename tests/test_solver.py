import random
import unittest
from copy import deepcopy
from itertools import combinations, product

from src.president import Card, RANKS, RANK_VALUE, Round
from src.solver import solve_round, _from_round, _moves, _step
from tests.test_president import ready_round


def clone_for_search(round_):
    """Copy engine state without copying historical logs at every branch.

    During play/completion, the only rule dependent on past events is whether
    opening_play has occurred. Keep that marker; inputs and all other mutable
    engine state remain independently copied. Use the real engine for moves.
    """
    opening = [{"type": "opening_play"}] if any(
        e["type"] == "opening_play" for e in round_.events) else []
    return deepcopy(round_, {id(round_.events): opening, id(round_.turn_history): []})


def engine_winner(round_, players=None, cache=None):
    """Independent exhaustive oracle using only real engine transitions."""
    players = tuple(round_._active_in_order()) if players is None else players
    cache = {} if cache is None else cache
    if round_.done:
        return min(players, key=round_.rankings.get)
    key = (tuple(tuple(sorted(round_.hands[p])) for p in players),
           round_.phase, round_.turn, round_.top_rank, round_.top_count,
           round_.top_stack_count, round_.last_player,
           tuple(round_.completion_order), round_._completion_cursor, round_._pending_advance)
    if key in cache:
        return cache[key]
    actor = round_._actor()
    for action in round_.legal_actions(actor):
        child = clone_for_search(round_)
        child.apply_action(actor, action)
        if engine_winner(child, players, cache) == actor:
            cache[key] = actor
            return actor
    cache[key] = next(p for p in players if p != actor)
    return cache[key]


def action_signature(action):
    cards = action.get("cards", [action["card"]] if "card" in action else [])
    return (action["type"], RANK_VALUE[cards[0]["rank"]] if cards else -1, len(cards))


def move_signature(move):
    return move.kind, move.rank, move.count


class SolverTests(unittest.TestCase):
    def test_search_clone_preserves_engine_behavior_and_isolation(self):
        normal = ready_round({"a": [Card("3", "C"), Card("7", "C")],
                              "b": [Card("7", s) for s in "DHS"]})
        opening = deepcopy(normal)
        opening.events = []
        completion = deepcopy(normal)
        completion.apply_action("a", {"type": "play", "cards": [Card("7", "C").json()]})
        for original in (normal, opening, completion):
            before = deepcopy(original.__dict__)
            actor = original._actor()
            cloned = clone_for_search(original)
            self.assertEqual(cloned.legal_actions(actor), original.legal_actions(actor))
            for action in original.legal_actions(actor):
                full = deepcopy(original)
                compact = clone_for_search(original)
                self.assertEqual(full.apply_action(actor, action), compact.apply_action(actor, action))
                self.assertEqual({k: v for k, v in full.__dict__.items() if k not in {"events", "turn_history"}},
                                 {k: v for k, v in compact.__dict__.items() if k not in {"events", "turn_history"}})
                self.assertEqual(full.legal_actions(full._actor()), compact.legal_actions(compact._actor()))
            self.assertEqual(original.__dict__, before)

    def check_position(self, round_, oracle_cache=None):
        before = deepcopy(round_.__dict__)
        result = solve_round(round_)
        self.assertEqual(round_.__dict__, before)
        self.assertIn(result.best_action, round_.legal_actions(result.player))
        oracle_cache = {} if oracle_cache is None else oracle_cache
        expected = engine_winner(round_, cache=oracle_cache)
        self.assertEqual(result.forced_result, "win" if expected == result.player else "loss")
        child = deepcopy(round_)
        child.apply_action(result.player, result.best_action)
        self.assertEqual(engine_winner(child, tuple(round_._active_in_order()), oracle_cache), expected)
        self.assertGreater(result.states_evaluated, 0)
        self.assertGreaterEqual(result.cache_hits, 0)

    def check_tree_transitions(self, round_):
        """Visit every distinct continuation, including strategically bad moves.

        Compare actual engine actions and observable successor states. This
        catches transition bugs even when they do not change a root's winner.
        """
        players = round_._active_in_order()
        seen = set()

        def visit(engine, state):
            if engine.done:
                self.assertIn(state.winner, (0, 1))
                self.assertEqual(players[state.winner], min(players, key=engine.rankings.get))
                return
            self.assertEqual(state.winner, -1)
            self.assertEqual(players[state.actor], engine._actor())
            self.assertEqual(state.hands, tuple(tuple(sum(c.rank == r for c in engine.hands[p])
                                                     for r in RANKS) for p in players))
            self.assertEqual((state.rank, state.count, state.stack),
                             (-1 if engine.top_rank is None else RANK_VALUE[engine.top_rank],
                              engine.top_count, engine.top_stack_count))
            self.assertEqual(bool(state.window), engine.phase == "completion")
            if state.window:
                self.assertEqual(tuple(players[i] for i in state.window),
                                 tuple(engine.completion_order[engine._completion_cursor:]))
            legal = {action_signature(a): a for a in engine.legal_actions(engine._actor())}
            moves = {move_signature(m): m for m in _moves(state)}
            self.assertEqual(set(moves), set(legal))
            # Include engine control fields so a wrongly merged solver state
            # cannot hide a distinct engine continuation.
            key = (state, engine.turn, engine.last_player, engine._pending_advance,
                   tuple(engine.completion_order), engine._completion_cursor)
            if key in seen:
                return
            seen.add(key)
            for signature, action in legal.items():
                child = deepcopy(engine)
                child.apply_action(engine._actor(), action)
                visit(child, _step(state, moves[signature]))

        visit(round_, _from_round(round_, players))
        return len(seen)

    def test_small_positions_against_engine(self):
        rng = random.Random(19)
        deck = [Card(rank, suit) for rank in ("3", "4", "5", "2") for suit in "CDHS"]
        for _ in range(30):
            cards = rng.sample(deck, 6)
            if all(c.rank == "2" for c in cards[:3]) or all(c.rank == "2" for c in cards[3:]):
                continue
            round_ = ready_round({"a": cards[:3], "b": cards[3:]})
            # Check both normal decisions and intermediate completion prompts.
            for _ in range(8):
                if round_.done:
                    break
                self.check_position(round_)
                actor = round_._actor()
                round_.apply_action(actor, rng.choice(round_.legal_actions(actor)))

    def test_inherited_board(self):
        for last in ("c", "d"):
            round_ = ready_round({"a": [Card("3", "C"), Card("4", "C")],
                                  "b": [Card("3", "D"), Card("5", "D")]},
                                 top=("3", 1), last=last)
            self.check_position(round_)
            round_.apply_action("a", {"type": "pass"})
            self.check_position(round_)

    def test_only_bombs_is_a_loss(self):
        round_ = ready_round({"a": [Card("3", "C"), Card("2", "C")],
                              "b": [Card("4", "C")]})
        self.assertEqual(solve_round(round_).forced_result, "loss")

    def test_rejects_non_duel(self):
        round_ = ready_round({p: [Card("3", suit)] for p, suit in zip("abc", "CDH")})
        with self.assertRaises(ValueError):
            solve_round(round_)

    def test_exhaustive_micro_deals_both_leaders(self):
        deck = [Card("3", "C"), Card("3", "D"), Card("4", "C"),
                Card("4", "D"), Card("2", "C")]
        checked = 0
        for allocation in product(range(3), repeat=len(deck)):
            hands = {p: [c for c, owner in zip(deck, allocation) if owner == i]
                     for i, p in enumerate("ab")}
            if any(not any(c.rank != "2" for c in hand) for hand in hands.values()):
                continue
            for actor in "ab":
                with self.subTest(allocation=allocation, actor=actor):
                    round_ = ready_round(hands, turn=actor)
                    self.check_position(round_)
                    self.check_tree_transitions(round_)
                    checked += 1
        self.assertEqual(checked, 300)

    def test_every_rank_and_response_size(self):
        for rank in RANKS[:-1]:
            for count in (1, 2, 3):
                with self.subTest(rank=rank, count=count):
                    round_ = ready_round({"a": [Card(rank, s) for s in "CDH"[:count]],
                                          "b": [Card("2", "S"), Card("A", "S")]},
                                         top=("3", count), last="b")
                    # Board cards are external to these hands; use a distinct
                    # lower rank when the responding rank itself is 3.
                    if rank == "3":
                        round_.top_rank = "4"
                    self.check_position(round_)
                    self.check_tree_transitions(round_)

    def test_bomb_beats_each_group_size(self):
        for count in (1, 2, 3):
            with self.subTest(count=count):
                round_ = ready_round({"a": [Card("3", "C"), Card("2", "C")],
                                      "b": [Card("K", "C")]}, top=("A", count), last="b")
                result = solve_round(round_)
                self.assertEqual(result.forced_result, "win")
                self.assertEqual(result.best_action["type"], "bomb")
                self.check_tree_transitions(round_)

    def test_completion_sizes_and_bomb_penalties(self):
        for stack in (1, 2, 3):
            for bombs in (False, True):
                with self.subTest(stack=stack, bombs=bombs):
                    hand = [Card("7", s) for s in "CDHS"[stack:]]
                    if bombs:
                        hand.append(Card("2", "C"))
                    round_ = ready_round({"a": [Card("8", "C")], "b": hand},
                                         top=("7", 1), last="a")
                    round_.top_stack_count = stack
                    round_._offer_completions("a")
                    round_.apply_action("a", {"type": "decline_completion"})
                    self.check_position(round_)
                    self.check_tree_transitions(round_)
                    result = solve_round(round_)
                    self.assertEqual(result.forced_result, "win")
                    self.assertEqual(result.best_action["type"],
                                     "decline_completion" if bombs else "complete")
                    if bombs:
                        # Completing now loses, but declining preserves a win
                        # by allowing the bomb to be spent first.
                        child = deepcopy(round_)
                        complete = next(a for a in child.legal_actions("b") if a["type"] == "complete")
                        child.apply_action("b", complete)
                        self.assertTrue(child.done)
                        self.assertLess(child.rankings["a"], child.rankings["b"])

    def test_self_completion_and_four_card_leads(self):
        for bomb in (False, True):
            hand = [Card("7", s) for s in "CDHS"] + [Card("8", "C")]
            if bomb:
                hand.append(Card("2", "C"))
            round_ = ready_round({"a": hand, "b": [Card("9", "C"), Card("9", "D")]})
            self.check_position(round_)
            self.check_tree_transitions(round_)
            round_.apply_action("a", {"type": "play", "cards": [Card("7", "C").json()]})
            self.check_position(round_)
            self.assertIn("complete", [a["type"] for a in round_.legal_actions("a")])

    def test_completion_preempts_equal_rank_skip(self):
        round_ = ready_round({"a": [Card("7", "D"), Card("3", "C")],
                              "b": [Card("7", "H"), Card("7", "S")]},
                             top=("7", 1), last="b")
        round_.apply_action("a", {"type": "play", "cards": [Card("7", "D").json()]})
        self.check_position(round_)
        self.check_tree_transitions(round_)
        round_.apply_action("a", {"type": "decline_completion"})
        result = solve_round(round_)
        self.assertEqual(result.best_action["type"], "complete")
        self.assertEqual(result.forced_result, "win")

    def test_equal_rank_single_and_pair_regain_lead(self):
        for count in (1, 2):
            with self.subTest(count=count):
                round_ = ready_round({"a": [Card("7", s) for s in "HS"[:count]] + [Card("3", "C")],
                                      "b": [Card("A", "C")]},
                                     top=("7", count), last="b")
                self.check_position(round_)
                self.check_tree_transitions(round_)
                result = solve_round(round_)
                self.assertEqual(result.forced_result, "win")
                self.assertEqual(action_signature(result.best_action), ("play", RANK_VALUE["7"], count))
                round_.apply_action("a", result.best_action)
                while round_.phase == "completion":
                    round_.apply_action(round_._actor(), {"type": "decline_completion"})
                self.assertEqual(round_._actor(), "a")
                self.assertIsNone(round_.top_rank)

    def test_finishing_play_ends_before_opponent_completion(self):
        round_ = ready_round({"a": [Card("7", "C")],
                              "b": [Card("7", s) for s in "DHS"]})
        self.check_tree_transitions(round_)
        result = solve_round(round_)
        self.assertEqual(result.forced_result, "win")
        round_.apply_action(result.player, result.best_action)
        self.assertTrue(round_.done)
        self.assertLess(round_.rankings["a"], round_.rankings["b"])

    def test_penalized_earlier_player_retains_unplayed_bombs(self):
        round_ = ready_round({"a": [Card("3", "C"), Card("2", "C")],
                              "b": [Card("4", "C")]})
        round_.finish_order = ["c"]
        round_.bomb_finishers = ["d"]
        round_.hands["d"] = [Card("2", "D"), Card("2", "H")]
        self.check_position(round_)
        self.check_tree_transitions(round_)

    def test_two_passes_clear_finished_players_board(self):
        # Regression: stale pass metadata used to recurse forever here.
        round_ = ready_round({"a": [Card("3", "C")], "b": [Card("4", "C")]},
                             top=("A", 1), last="c")
        self.check_position(round_)
        self.check_tree_transitions(round_)
        self.assertEqual(solve_round(round_).best_action, {"type": "pass"})
        for actor in "ab":
            round_.apply_action(actor, {"type": "pass"})
            while round_.phase == "completion":
                self.check_position(round_)
                round_.apply_action(round_._actor(), {"type": "decline_completion"})
        self.assertIsNone(round_.top_rank)
        self.assertEqual(round_._actor(), "a")

    def test_completion_reopens_after_pass(self):
        round_ = ready_round({"a": [Card("7", "D"), Card("7", "H"), Card("7", "S"), Card("3", "C")],
                              "b": [Card("4", "C")]}, top=("7", 1), last="a")
        round_._offer_completions("a")
        for actor in "ab":
            round_.apply_action(actor, {"type": "decline_completion"})
        round_.apply_action("b", {"type": "pass"})
        round_.apply_action("b", {"type": "decline_completion"})
        self.assertIn("complete", [a["type"] for a in round_.legal_actions("a")])
        self.check_position(round_)
        self.check_tree_transitions(round_)

    def test_all_active_seat_pairs_and_inherited_boards(self):
        for pair in combinations("abcd", 2):
            for actor in pair:
                for last in "abcd":
                    if last == actor:
                        continue
                    with self.subTest(pair=pair, actor=actor, last=last):
                        round_ = ready_round({pair[0]: [Card("3", "C"), Card("5", "C")],
                                              pair[1]: [Card("4", "D"), Card("6", "D")]},
                                             turn=actor, top=("7", 1), last=last)
                        self.check_position(round_)
                        self.check_tree_transitions(round_)

    def test_real_round_endgames(self):
        checked = 0
        pairs = set()
        for seed in range(40):
            rng = random.Random(seed)
            round_ = Round(["a", "b", "c", "d"], first_round=True, rng=rng)
            # All checked positions belong to the same two-player endgame.
            # Keep proofs across successive decisions, including declines.
            oracle_cache = {}
            for _ in range(1000):
                if round_.done:
                    break
                if len(round_.active) == 2 and sum(len(round_.hands[p]) for p in round_.active) <= 7:
                    pairs.add(tuple(round_._active_in_order()))
                    self.check_position(round_, oracle_cache)
                    checked += 1
                actor = round_._actor()
                round_.apply_action(actor, rng.choice(round_.legal_actions(actor)))
            self.assertTrue(round_.done, f"seed {seed} failed to terminate")
        self.assertGreater(checked, 100)
        self.assertEqual(len(pairs), 6)

    def test_suit_permutation_and_hand_order_invariance(self):
        round_ = ready_round({"a": [Card("7", "C"), Card("7", "D"), Card("3", "C"), Card("2", "C")],
                              "b": [Card("7", "H"), Card("7", "S"), Card("A", "D")]})
        expected = solve_round(round_)
        changed = deepcopy(round_)
        suits = dict(zip("CDHS", "SHDC"))
        changed.hands = {p: [Card(c.rank, suits[c.suit]) for c in reversed(hand)]
                         for p, hand in changed.hands.items()}
        actual = solve_round(changed)
        self.assertEqual(actual.forced_result, expected.forced_result)
        self.assertEqual(action_signature(actual.best_action), action_signature(expected.best_action))
        self.assertEqual(actual.states_evaluated, expected.states_evaluated)
        self.check_position(changed)

    def test_repeated_calls_and_result_mutation_are_isolated(self):
        round_ = ready_round({"a": [Card("3", "C"), Card("4", "C")], "b": [Card("5", "C")]})
        expected = solve_round(round_)
        altered = solve_round(round_)
        altered.best_action.clear()
        self.assertEqual(solve_round(round_), expected)
        self.check_position(round_)

    def test_rejects_finished_wrong_phase_empty_and_bomb_only_hands(self):
        base = ready_round({"a": [Card("3", "C")], "b": [Card("4", "C")]})
        for phase in ("finished", "draft", "trade_request", "trade_return"):
            with self.subTest(phase=phase):
                round_ = deepcopy(base)
                round_.phase = phase
                with self.assertRaises(ValueError):
                    solve_round(round_)
        for hand in ([], [Card("2", "C")], [Card("2", "C"), Card("2", "D")]):
            with self.subTest(hand=hand):
                round_ = deepcopy(base)
                round_.hands["a"] = hand
                with self.assertRaises(ValueError):
                    solve_round(round_)
