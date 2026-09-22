"""Measure next-round placement by randomized starting role using BasicClient."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random

from src.client import BasicClient
from src.president import ROLE_NAMES, Round


PLAYERS = ("a", "b", "c", "d")  # Fixed counterclockwise seats.


def role_assignments(games, rng):
    """Randomize roles in blocks of four, balancing each role across seats."""
    for start in range(0, games, 4):
        ranks = [1, 2, 3, 4]
        rng.shuffle(ranks)
        block = [ranks[offset:] + ranks[:offset] for offset in range(4)]
        rng.shuffle(block)
        for assignment in block[:min(4, games - start)]:
            yield dict(zip(PLAYERS, assignment))


def simulate(games=1000, seed=42, max_actions=10000):
    """Run independent later-round trials, including drafting and trading.

    Starting roles are assigned experimentally, rather than earned in a prior
    round. Each trial uses a fresh shuffled deck and four identical agents.
    Results describe this agent policy and the complete bundle of role rules.
    """
    if games <= 0 or max_actions <= 0:
        raise ValueError("games and max_actions must be positive")
    master = random.Random(seed)
    role_rng = random.Random(master.getrandbits(64))
    deck_rng = random.Random(master.getrandbits(64))
    clients = {player: BasicClient() for player in PLAYERS}
    records = []
    for trial, roles in enumerate(role_assignments(games, role_rng), 1):
        deck_seed = deck_rng.getrandbits(64)
        round_ = Round(PLAYERS, first_round=False, roles=roles,
                       rng=random.Random(deck_seed))
        for actions in range(1, max_actions + 1):
            player = round_._actor()
            round_.apply_action(player, clients[player].respond(round_.message_for(player)))
            if round_.done:
                break
        else:
            raise RuntimeError(f"trial {trial} reached the action limit ({max_actions})")
        records.append({"trial": trial, "deck_seed": deck_seed, "starting_roles": roles,
                        "placements": dict(round_.rankings), "actions": actions})
    return {"games": games, "seed": seed, "agent": "BasicClient", "players": list(PLAYERS),
            "design": "Fresh later-round deals; randomized roles balanced across seats in blocks of four.",
            "records": records}


def placement_counts(records):
    """Rows are starting roles; columns are next-round placements (both 1..4)."""
    counts = [[0] * 4 for _ in range(4)]
    for record in records:
        for player, role in record["starting_roles"].items():
            counts[role - 1][record["placements"][player] - 1] += 1
    return counts


def format_report(result):
    counts = placement_counts(result["records"])
    lines = [
        f"{result['games']:,} randomized-role rounds | BasicClient | seed {result['seed']}",
        "Each trial starts with assigned roles, then drafts, trades, and plays a fresh deal.",
        "Rows: starting role. Columns: next-round placement; count (row percentage).",
        "",
        f"{'Starting role':<16} {'1st':>15} {'2nd':>15} {'3rd':>15} {'4th':>15} {'Mean place':>12}",
    ]
    for name, row in zip(ROLE_NAMES, counts):
        total = sum(row)
        cells = [f"{n} ({n / total:.1%})" for n in row]
        mean = sum(place * n for place, n in enumerate(row, 1)) / total
        lines.append(f"{name:<16} " + " ".join(f"{cell:>15}" for cell in cells) + f" {mean:>12.3f}")
    lines += ["", "Lower mean placement is better; equal placement probabilities give a mean of 2.500.",
              "These results measure the combined draft, trade, and starting-turn rules with this agent.",
              "", "Starting-role assignments by seat (President, Vice President, Vice Scum, Scum):"]
    for player in result["players"]:
        assigned = Counter(r["starting_roles"][player] for r in result["records"])
        lines.append(f"  {player}: " + ", ".join(str(assigned[role]) for role in range(1, 5)))
    return "\n".join(lines)


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=positive_int, default=1000, help="number of fresh rounds (default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="reproducible random seed (default: 42)")
    parser.add_argument("--max-actions", type=positive_int, default=10000, help="maximum responses per round")
    parser.add_argument("--output", type=Path, help="optional JSON file with every trial's roles, placements, and deck seed")
    args = parser.parse_args()
    result = simulate(args.games, args.seed, args.max_actions)
    print(format_report(result))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"\nTrial data saved to {args.output}")


if __name__ == "__main__":
    main()
