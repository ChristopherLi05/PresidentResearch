"""Validate a saved President role-simulation result and print its placement table.

By default this checks the JSON's trial structure and recomputes the table from
the stored records. Pass --replay to rerun every recorded deal and verify the
roles, placements, and action count deterministically.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
from typing import Any

from src.client import BasicClient
from src.president import ROLE_NAMES, Round
from src.simulate import PLAYERS, placement_counts


def load_result(path: Path) -> dict[str, Any]:
    """Load and structurally validate a simulation-result JSON file."""
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)

    required = {"games", "seed", "agent", "players", "records"}
    missing = required - set(result)
    if missing:
        raise ValueError(f"missing top-level fields: {', '.join(sorted(missing))}")
    if not isinstance(result["games"], int) or result["games"] <= 0:
        raise ValueError("games must be a positive integer")
    if result["players"] != list(PLAYERS):
        raise ValueError(f"expected players {list(PLAYERS)}, found {result['players']}")
    if len(result["records"]) != result["games"]:
        raise ValueError(f"games says {result['games']}, but file contains {len(result['records'])} records")

    expected_roles = {1, 2, 3, 4}
    for expected_trial, record in enumerate(result["records"], 1):
        if record.get("trial") != expected_trial:
            raise ValueError(f"record {expected_trial}: unexpected trial number {record.get('trial')!r}")
        roles = record.get("starting_roles")
        placements = record.get("placements")
        if not isinstance(roles, dict) or set(roles) != set(PLAYERS) or set(roles.values()) != expected_roles:
            raise ValueError(f"trial {expected_trial}: invalid starting_roles")
        if not isinstance(placements, dict) or set(placements) != set(PLAYERS) or set(placements.values()) != expected_roles:
            raise ValueError(f"trial {expected_trial}: invalid placements")
        if not isinstance(record.get("deck_seed"), int):
            raise ValueError(f"trial {expected_trial}: missing or invalid deck_seed")
        if not isinstance(record.get("actions"), int) or record["actions"] <= 0:
            raise ValueError(f"trial {expected_trial}: missing or invalid actions")
    return result


def verify_role_balance(result: dict[str, Any]) -> None:
    """Confirm each player received each role equally when games is divisible by four."""
    expected = result["games"] // 4
    if result["games"] % 4:
        return
    for player in PLAYERS:
        assigned = Counter(record["starting_roles"][player] for record in result["records"])
        if any(assigned[role] != expected for role in range(1, 5)):
            raise ValueError(f"roles are not balanced for player {player}: {dict(assigned)}")


def replay(result: dict[str, Any]) -> None:
    """Replay every deal and compare its recorded outcome to the engine result."""
    clients = {player: BasicClient() for player in PLAYERS}
    for record in result["records"]:
        round_ = Round(
            PLAYERS,
            first_round=False,
            roles=record["starting_roles"],
            rng=random.Random(record["deck_seed"]),
        )
        for actions in range(1, 10001):
            player = round_._actor()
            if player is None:
                raise RuntimeError(f"trial {record['trial']}: no current actor")
            action = clients[player].respond(round_.message_for(player))
            round_.apply_action(player, action)
            if round_.done:
                break
        else:
            raise RuntimeError(f"trial {record['trial']}: replay reached the action limit")

        if round_.rankings != record["placements"]:
            raise ValueError(
                f"trial {record['trial']}: placement mismatch; "
                f"recorded {record['placements']}, replayed {round_.rankings}"
            )
        if actions != record["actions"]:
            raise ValueError(
                f"trial {record['trial']}: action mismatch; "
                f"recorded {record['actions']}, replayed {actions}"
            )


def print_table(result: dict[str, Any]) -> None:
    counts = placement_counts(result["records"])
    print(f"Verified {result['games']:,} records (seed {result['seed']}).")
    print()
    print(f"{'Starting role':<16} {'1st':>8} {'2nd':>8} {'3rd':>8} {'4th':>8} {'Total':>8}")
    for name, row in zip(ROLE_NAMES, counts):
        print(f"{name:<16}" + "".join(f"{count:>8,}" for count in row) + f"{sum(row):>8,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("results/role_simulation_5000.json"),
        help="simulation JSON to validate (default: results/role_simulation_5000.json)",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="rerun every recorded deal and compare its result; this takes longer",
    )
    args = parser.parse_args()

    result = load_result(args.path)
    verify_role_balance(result)
    if args.replay:
        replay(result)
        print("Deterministic replay passed.")
    print_table(result)


if __name__ == "__main__":
    main()
