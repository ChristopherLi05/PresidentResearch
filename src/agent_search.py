"""Bounded root Monte Carlo, with public-information heuristic continuations.

No tree, recursive agent calls, or exact solver is used in sampled worlds.
Only complete shared-world batches contribute to the final comparison.
"""

from copy import copy
from math import exp
from time import perf_counter

from src.president import Card, Round, standard_deck


def action_key(action):
    cards = action.get("cards", [])
    if action["type"] == "bomb":
        cards = [action["card"]]
    return action["type"], cards[0]["rank"] if cards else None, len(cards)


def unique_actions(actions):
    return list({action_key(a): a for a in reversed(actions)}.values())[::-1]


def snapshot(round_, player):
    """Avoid copying history; consumers treat this private snapshot as read-only."""
    return {"player": player, "players": round_.players,
            "active_players": round_._active_in_order(), "events": round_.events,
            "hand": [c.json() for c in round_._sorted_hand(player)],
            "top": None if round_.top_rank is None else {
                "rank": round_.top_rank, "count": round_.top_count,
                "stack_count": round_.top_stack_count}}


def prepare(agent, message):
    """Reconstruct public mechanics, rejecting histories we cannot account for."""
    events = message.get("events", [])
    players = list(message.get("players", []))
    active = list(message.get("active_players", []))
    player = message["player"]
    if (len(players) != 4 or len(set(players)) != 4 or player not in active
            or not set(active) <= set(players) or len(active) != len(set(active))
            or not events or events[0].get("type") != "round_started"
            or any(e.get("type") == "twos_announced" for e in events)):
        return None
    own = [Card.from_json(c) for c in message["hand"]]
    played = agent._publicly_played(message)
    known = own + played
    deck = standard_deck()
    if len(set(known)) != len(known) or not set(known) <= set(deck):
        return None
    sizes = agent._opponent_counts(message)
    if sizes.get(player) != len(own) or any(
            sizes.get(p, -1) <= 0 if p in active else sizes.get(p) != 0 for p in players):
        return None
    known_set = set(known)
    unseen = [c for c in deck if c not in known_set]
    if len(unseen) != sum(sizes[p] for p in active if p != player):
        return None
    state = Round(players, first_round=True, deck=deck)
    state.first_round = bool(events[0].get("first_round"))
    state.events = list(events)
    state.hands = {p: own[:] if p == player else [] for p in players}
    state.active = set(active)
    state.finish_order = [e["player"] for e in events
                          if e.get("type") == "player_done" and not e.get("bomb_finish")]
    state.bomb_finishers = [e["player"] for e in events
                            if e.get("type") == "player_done" and e.get("bomb_finish")]
    if set(state.finish_order + state.bomb_finishers) != set(players) - set(active):
        return None
    state.phase = message["request"]["type"]
    state.turn = player
    top = message.get("top")
    if top:
        state.top_rank = top["rank"]
        state.top_count = top["count"]
        state.top_stack_count = top["stack_count"]
        last = next((e for e in reversed(events) if e.get("type") in {"played", "opening_play"}), None)
        if last is None:
            return None
        state.last_player = last["player"]
    if state.phase == "completion":
        source = agent._completion_source(events)
        if source is None or top is None:
            return None
        previous, skip = source
        state.completion_order = ([previous] if previous in active else []) + [
            p for p in state._active_after(previous) if p != previous]
        if player not in state.completion_order:
            return None
        state._completion_cursor = state.completion_order.index(player)
        state._pending_advance = (previous, skip)
    return state, unseen, sizes


def clone(state):
    result = copy(state)
    result.hands = {p: cards[:] for p, cards in state.hands.items()}
    for name in ("events", "finish_order", "bomb_finishers", "completion_order"):
        setattr(result, name, list(getattr(state, name)))
    result.active = set(state.active)
    result.turn_history = []
    return result


def sample_world(prepared, player, rng):
    state, unseen, sizes = prepared
    result = clone(state)
    cards = unseen[:]
    rng.shuffle(cards)
    offset = 0
    for opponent in state.players:
        if opponent != player and opponent in state.active:
            result.hands[opponent] = cards[offset:offset + sizes[opponent]]
            offset += sizes[opponent]
    return result


def heuristic_action(agent, state):
    player = state._actor()
    actions = unique_actions(state.legal_actions(player))
    if len(actions) == 1:
        return actions[0]
    message = snapshot(state, player)
    for action in actions:
        if action["type"] in {"play", "complete"} and len(action["cards"]) == len(message["hand"]):
            return action
    return min(actions, key=agent._play_scorer(message))


def rollout(agent, world, player, action, deadline):
    state = clone(world)
    state.apply_action(player, action)
    normal = int(world.phase == "play")
    while not state.done and player in state.active:
        if perf_counter() >= deadline:
            return None
        if state.phase == "play" and (state._actor() == player or normal >= 16):
            break
        actor = state._actor()
        chosen = heuristic_action(agent, state)
        if perf_counter() >= deadline:
            return None
        normal += int(state.phase == "play")
        state.apply_action(actor, chosen)
    return state


def evaluate(agent, state, player):
    """Expected fraction of other seats beaten, with exact resolved ordering."""
    costs = {}

    def cost(p):
        if p not in costs:
            message = snapshot(state, p)
            counts = agent._counts(message["hand"])
            unseen = agent._unseen(message)
            opponents = agent._opponents(message)
            method = agent._lead_cost if state.top_rank is None and state._actor() == p else agent._hand_cost
            costs[p] = method(counts, unseen, opponents)
        return costs[p]

    def order(p):
        if p in state.finish_order:
            return 0, state.finish_order.index(p)
        if p in state.bomb_finishers:
            return 2, state.bomb_finishers.index(p)
        return 1, 0

    total = 0.0
    for opponent in state.players:
        if opponent == player:
            continue
        if state.rankings:
            total += state.rankings[player] < state.rankings[opponent]
        elif order(player) != (1, 0) or order(opponent) != (1, 0):
            total += order(player) < order(opponent)
        else:
            total += 1.0 / (1.0 + exp(max(-60.0, min(60.0, cost(player) - cost(opponent)))))
    return total / (len(state.players) - 1)


def choose(agent, message, actions):
    from src import agent_search_c
    start = perf_counter()
    deadline = float("inf") if agent.search_time_ms is None else start + agent.search_time_ms / 1000
    stats = agent.last_search = {"worlds": 0, "candidates": 0, "fallback": True,
                                 "reason": "budget", "elapsed_ms": 0.0, "backend": "python",
                                 "attempts": 0, "rejected_worlds": 0, "native_errors": 0}
    try:
        distinct = unique_actions(actions)
        if len(distinct) == 1:
            stats.update(fallback=False, reason="forced")
            return distinct[0]
        for action in distinct:
            if action["type"] in {"play", "complete"} and len(action["cards"]) == len(message["hand"]):
                stats.update(fallback=False, reason="finish")
                return action
        ranked = sorted(distinct, key=agent._play_scorer(message))
        fallback = ranked[0]
        if not agent.search_worlds or perf_counter() >= deadline:
            return fallback
        candidates = [a for i, a in enumerate(ranked)
                      if i < 4 or a["type"] in {"pass", "decline_completion", "bomb"}]
        stats["candidates"] = len(candidates)
        try:
            prepared = prepare(agent, message)
        except (KeyError, IndexError, TypeError, ValueError):
            prepared = None
        if prepared is None:
            stats["reason"] = "unsupported_history"
            return fallback
        native = agent_search_c.backend(agent)
        if native is not None:
            stats["backend"] = "c"
        totals = [0.0] * len(candidates)
        # At most three replacement samples, within the original time budget.
        for _ in range(agent.search_worlds + 3):
            if stats["worlds"] >= agent.search_worlds or perf_counter() >= deadline:
                break
            stats["attempts"] += 1
            world = sample_world(prepared, message["player"], agent.rng)
            # After a play the rules retire every all-2s hand. Such a sampled
            # active opponent contradicts public state and can have no lead.
            if any(not world.hands[p] or all(c.rank == "2" for c in world.hands[p])
                   for p in world.active):
                stats["rejected_worlds"] += 1
                stats["reason"] = "rejected_samples"
                continue
            batch = []
            if native is not None:
                remaining = -1 if agent.search_time_ms is None else max(0, (deadline-perf_counter())*1000)
                try:
                    batch = agent_search_c.batch(native, agent, world, message["player"], candidates, remaining)
                except agent_search_c.NativeSearchRejected:
                    stats["native_errors"] += 1
                    stats["rejected_worlds"] += 1
                    stats["reason"] = "native_rejection"
                    continue
                if batch is None:
                    break
            else:
                for action in candidates:
                    if perf_counter() >= deadline:
                        break
                    state = rollout(agent, world, message["player"], action, deadline)
                    if state is None:
                        break
                    batch.append(evaluate(agent, state, message["player"]))
            if len(batch) != len(candidates) or perf_counter() >= deadline:
                break
            totals = [total + value for total, value in zip(totals, batch)]
            stats["worlds"] += 1
        if not stats["worlds"]:
            return fallback
        stats["fallback"] = False
        stats["reason"] = "search"
        return candidates[max(range(len(candidates)), key=totals.__getitem__)]
    finally:
        stats["elapsed_ms"] = (perf_counter() - start) * 1000
