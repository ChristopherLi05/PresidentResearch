"""Run a parallel 2v2, single-round comparison of two President agents.

Each worker owns one of the six possible two-seat configurations.  Deals and
starting roles are generated before workers start, so a fixed seed produces
the same trial data as the serial ``arena.py`` runner.

For example:
    python arena_multithread.py src.agent:BasicAgentPlus src.agent:BasicAgent --workers 6
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import random
from typing import Any, Optional

from src.agent import Agent
from src.president import Round
from src.simulate import PLAYERS, play_round, role_assignments
from arena import (
    TEAM_CONFIGURATIONS,
    board_clear_events,
    client_factory,
    format_free_for_all_report,
    format_report,
    positive_int,
    run_free_for_all,
    summarize,
)


def _run_configuration(
    configuration: tuple[str, str],
    evaluated_client: type[Agent],
    opposing_client: type[Agent],
    starting_roles: list[dict[str, int]],
    deal_seeds: list[int],
    max_actions: int,
    progress: Optional[Any],
) -> list[dict[str, Any]]:
    """Run one seating configuration without sharing mutable round state."""
    evaluated_seats = set(configuration)
    opposing_seats = [player for player in PLAYERS if player not in evaluated_seats]
    records = []
    for trial, (roles, deck_seed) in enumerate(zip(starting_roles, deal_seeds), 1):
        clients = {
            player: (evaluated_client() if player in evaluated_seats else opposing_client())
            for player in PLAYERS
        }
        round_ = Round(PLAYERS, first_round=False, roles=roles, rng=random.Random(deck_seed))
        actions = play_round(
            round_, clients, max_actions=max_actions,
            trial_label=f"configuration {''.join(configuration)}, trial {trial}",
        )
        evaluated_placements = [round_.rankings[player] for player in configuration]
        records.append({
            "configuration": list(configuration),
            "trial": trial,
            "deck_seed": deck_seed,
            "starting_roles": roles,
            "placements": dict(round_.rankings),
            "evaluated_placements": evaluated_placements,
            "evaluated_mean_placement": sum(evaluated_placements) / 2,
            "actions": actions,
            "evaluated_actions": sum(
                turn["player"] in evaluated_seats for turn in round_.turn_history
            ),
            "opposing_actions": sum(
                turn["player"] in opposing_seats for turn in round_.turn_history
            ),
            "evaluated_bombs": sum(
                event["type"] == "bombed" and event["player"] in evaluated_seats
                for event in round_.events
            ),
            "opposing_bombs": sum(
                event["type"] == "bombed" and event["player"] in opposing_seats
                for event in round_.events
            ),
            "evaluated_completion_events": sum(
                event["type"] == "completed" and event["player"] in evaluated_seats
                for event in round_.events
            ),
            "opposing_completion_events": sum(
                event["type"] == "completed" and event["player"] in opposing_seats
                for event in round_.events
            ),
            "twos_rule_players": [
                event["player"] for event in round_.events if event["type"] == "twos_announced"
            ],
            "board_clear_events": board_clear_events(round_.events),
        })
        if progress is not None:
            progress.update(1)
    return records


def run_match(
    evaluated_client: type[Agent], opposing_client: type[Agent], *,
    trials_per_configuration: int = 10_000, seed: int = 42,
    max_actions: int = 10_000, workers: Optional[int] = None,
    progress: Optional[Any] = None,
) -> dict[str, Any]:
    """Run the six configurations concurrently with reproducible deal inputs."""
    if trials_per_configuration <= 0 or max_actions <= 0:
        raise ValueError("trials_per_configuration and max_actions must be positive")
    if not (isinstance(evaluated_client, type) and issubclass(evaluated_client, Agent)
            and isinstance(opposing_client, type) and issubclass(opposing_client, Agent)):
        raise ValueError("both clients must be Agent subclasses")
    if workers is not None and workers <= 0:
        raise ValueError("workers must be positive")

    seed_rng = random.Random(seed)
    role_rng = random.Random(seed_rng.getrandbits(64))
    deck_rng = random.Random(seed_rng.getrandbits(64))
    starting_roles = list(role_assignments(trials_per_configuration, role_rng))
    deal_seeds = [deck_rng.getrandbits(64) for _ in range(trials_per_configuration)]
    worker_count = min(workers or (os.cpu_count() or 1), len(TEAM_CONFIGURATIONS))

    records_by_configuration: dict[tuple[str, str], list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="arena") as executor:
        futures = {
            executor.submit(
                _run_configuration, configuration, evaluated_client, opposing_client,
                starting_roles, deal_seeds, max_actions, progress,
            ): configuration
            for configuration in TEAM_CONFIGURATIONS
        }
        for future in as_completed(futures):
            configuration = futures[future]
            records_by_configuration[configuration] = future.result()

    records = [
        record for configuration in TEAM_CONFIGURATIONS
        for record in records_by_configuration[configuration]
    ]
    return {
        "evaluated_client": f"{evaluated_client.__module__}:{evaluated_client.__name__}",
        "opposing_client": f"{opposing_client.__module__}:{opposing_client.__name__}",
        "trials_per_configuration": trials_per_configuration,
        "games": len(records),
        "seed": seed,
        "workers": worker_count,
        "players": list(PLAYERS),
        "design": (
            "Six two-seat 2v2 configurations run concurrently; every configuration uses "
            "the same fixed later-round deal and starting-rank assignments. Clients are "
            "recreated for every single-round trial."
        ),
        "summary": {
            "evaluated": summarize(records, "evaluated"),
            "opposing": summarize(records, "opposing"),
        },
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clients", nargs="+", help="one, two, or four package.module:ClassName agents")
    parser.add_argument("--trials-per-configuration", type=positive_int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-actions", type=positive_int, default=10_000)
    parser.add_argument(
        "--workers", type=positive_int, default=min(os.cpu_count() or 1, len(TEAM_CONFIGURATIONS)),
        help="worker threads, capped at the six seating configurations",
    )
    parser.add_argument("--output", type=Path, help="optional JSON file with trial records")
    args = parser.parse_args()
    if len(args.clients) not in {1, 2, 4}:
        parser.error("provide one agent (all four seats), two agents (2v2), or four agents (1v1v1v1)")
    try:
        from tqdm import tqdm
    except ImportError as error:
        parser.error("tqdm is required for this runner; install it with: pip install -r requirements.txt")
        raise error

    classes = [client_factory(specification) for specification in args.clients]
    games = len(TEAM_CONFIGURATIONS) * args.trials_per_configuration if len(classes) == 2 else args.trials_per_configuration
    with tqdm(total=games, desc="Running parallel arena trials", unit="game") as progress:
        if len(classes) == 2:
            result = run_match(
                classes[0], classes[1], trials_per_configuration=args.trials_per_configuration,
                seed=args.seed, max_actions=args.max_actions, workers=args.workers, progress=progress,
            )
            report = format_report(result)
        else:
            result = run_free_for_all(classes, trials=args.trials_per_configuration, seed=args.seed,
                                      max_actions=args.max_actions, progress=progress)
            report = format_free_for_all_report(result)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"\nTrial data saved to {args.output}")


if __name__ == "__main__":
    main()
