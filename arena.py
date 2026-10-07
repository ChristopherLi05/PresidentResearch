"""Run a balanced 2v2, single-round comparison of two President clients.

For example:
    python test.py src.client:BasicClientPlus src.client:BasicClient
"""

import argparse
from collections import Counter
from importlib import import_module
from itertools import combinations
import json
from math import sqrt
from pathlib import Path
import random
from typing import Any, Callable, Mapping, Optional

from src.agent import Agent
from src.president import ROLE_NAMES, Round
from src.simulate import PLAYERS, play_round, role_assignments


TEAM_CONFIGURATIONS = tuple(combinations(PLAYERS, 2))
Z_95 = 1.96
BOARD_CLEAR_TYPES = ("pass_around", "completion", "2_bomb", "four_of_a_kind_play")
BOARD_CLEAR_LABELS = {
    "pass_around": "pass-around",
    "completion": "completion",
    "2_bomb": "2 bomb",
    "four_of_a_kind_play": "four-of-a-kind",
}


def client_factory(specification: str) -> Callable[[], Agent]:
    """Resolve a client specified as ``package.module:ClassName``."""
    try:
        module_name, class_name = specification.split(":", 1)
        candidate = getattr(import_module(module_name), class_name)
    except (AttributeError, ImportError, ValueError) as error:
        raise ValueError(
            f"client must be package.module:ClassName, not {specification!r}"
        ) from error
    if not isinstance(candidate, type) or not issubclass(candidate, Agent):
        raise ValueError(f"{specification!r} must name a Client subclass")
    return candidate


def confidence_interval(samples: list[float]) -> tuple[float, float]:
    """Return a normal-approximation 95% confidence interval for the mean."""
    mean = sum(samples) / len(samples)
    if len(samples) == 1:
        return mean, mean
    variance = sum((sample - mean) ** 2 for sample in samples) / (len(samples) - 1)
    margin = Z_95 * sqrt(variance / len(samples))
    return mean - margin, mean + margin


def _team_seats(record: Mapping[str, Any], team: str) -> list[str]:
    evaluated_seats = record["configuration"]
    return evaluated_seats if team == "evaluated" else [
        player for player in PLAYERS if player not in evaluated_seats
    ]


def board_clear_events(events: list[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Classify every board clear by the action that caused it."""
    classifications = []
    causes = {
        "passed": "pass_around",
        "completed": "completion",
        "bombed": "2_bomb",
        "played": "four_of_a_kind_play",
        "opening_play": "four_of_a_kind_play",
    }
    gameplay_types = set(causes)
    for index, event in enumerate(events):
        if event["type"] != "board_cleared":
            continue
        cause = next(
            candidate for candidate in reversed(events[:index])
            if candidate["type"] in gameplay_types
        )
        classifications.append({"player": cause["player"], "type": causes[cause["type"]]})
    return classifications


def summarize(records: list[Mapping[str, Any]], team: str) -> dict[str, Any]:
    """Summarize placement and gameplay statistics for one policy's team."""
    games = len(records)
    team_means = [sum(record["placements"][player] for player in _team_seats(record, team)) / 2
                  for record in records]
    counts: Counter[int] = Counter()
    twos_rule_by_initial_rank: Counter[int] = Counter()
    board_clears: Counter[str] = Counter()
    by_initial_rank: dict[int, list[int]] = {rank: [] for rank in range(1, 5)}
    for record in records:
        for player in _team_seats(record, team):
            placement = record["placements"][player]
            counts[placement] += 1
            initial_rank = record["starting_roles"][player]
            if player in record["twos_rule_players"]:
                twos_rule_by_initial_rank[initial_rank] += 1
            by_initial_rank[initial_rank].append(placement)
        team_seats = set(_team_seats(record, team))
        board_clears.update(
            event["type"] for event in record["board_clear_events"]
            if event["player"] in team_seats
        )
    ci_low, ci_high = confidence_interval(team_means)
    placement_by_initial_rank = {}
    for rank, placements in by_initial_rank.items():
        if placements:
            rank_counts = Counter(placements)
            rank_low, rank_high = confidence_interval(placements)
            placement_by_initial_rank[ROLE_NAMES[rank - 1]] = {
                "games": len(placements),
                "mean_placement": sum(placements) / len(placements),
                "mean_placement_95_ci": [rank_low, rank_high],
                "placement_counts": {place: rank_counts[place] for place in range(1, 5)},
            }
    return {
        "games": games,
        "mean_placement": sum(team_means) / games,
        "mean_placement_95_ci": [ci_low, ci_high],
        "placement_counts": {place: counts[place] for place in range(1, 5)},
        "twos_rule_by_initial_position": {
            ROLE_NAMES[rank - 1]: twos_rule_by_initial_rank[rank] for rank in range(1, 5)
        },
        "board_clear_type_percentages": {
            clear_type: 100 * board_clears[clear_type] / sum(board_clears.values())
            if board_clears else 0.0
            for clear_type in BOARD_CLEAR_TYPES
        },
        "mean_completion_events": sum(record[f"{team}_completion_events"] for record in records) / games,
        "mean_actions": sum(record[f"{team}_actions"] for record in records) / games,
        "mean_bombs": sum(record[f"{team}_bombs"] for record in records) / games,
        "placement_by_initial_rank": placement_by_initial_rank,
    }


def run_match(evaluated_client: type[Agent], opposing_client: type[Agent], *,
              trials_per_configuration: int = 10_000, seed: int = 42,
              max_actions: int = 10_000, progress: Optional[Any] = None) -> dict[str, Any]:
    """Run six paired-seat configurations against the same deal seeds.

    One trial is one later-round deal with experimentally assigned starting
    ranks. Both members of a team use the same policy, and all client
    instances are recreated for every trial.
    """
    if trials_per_configuration <= 0 or max_actions <= 0:
        raise ValueError("trials_per_configuration and max_actions must be positive")
    if not (isinstance(evaluated_client, type) and issubclass(evaluated_client, Agent)
            and isinstance(opposing_client, type) and issubclass(opposing_client, Agent)):
        raise ValueError("both clients must be Client subclasses")

    seed_rng = random.Random(seed)
    role_rng = random.Random(seed_rng.getrandbits(64))
    deck_rng = random.Random(seed_rng.getrandbits(64))
    starting_roles = list(role_assignments(trials_per_configuration, role_rng))
    deal_seeds = [deck_rng.getrandbits(64) for _ in range(trials_per_configuration)]
    records = []
    for configuration in TEAM_CONFIGURATIONS:
        evaluated_seats = set(configuration)
        for trial, (roles, deck_seed) in enumerate(zip(starting_roles, deal_seeds), 1):
            clients = {player: (evaluated_client() if player in evaluated_seats else opposing_client())
                       for player in PLAYERS}
            round_ = Round(PLAYERS, first_round=False, roles=roles, rng=random.Random(deck_seed))
            actions = play_round(round_, clients, max_actions=max_actions,
                                 trial_label=f"configuration {''.join(configuration)}, trial {trial}")
            evaluated_placements = [round_.rankings[player] for player in configuration]
            opposing_seats = [player for player in PLAYERS if player not in evaluated_seats]
            records.append({
                "configuration": list(configuration), "trial": trial, "deck_seed": deck_seed,
                "starting_roles": roles,
                "placements": dict(round_.rankings), "evaluated_placements": evaluated_placements,
                "evaluated_mean_placement": sum(evaluated_placements) / 2,
                "actions": actions,
                "evaluated_actions": sum(turn["player"] in evaluated_seats for turn in round_.turn_history),
                "opposing_actions": sum(turn["player"] in opposing_seats for turn in round_.turn_history),
                "evaluated_bombs": sum(event["type"] == "bombed" and event["player"] in evaluated_seats
                                        for event in round_.events),
                "opposing_bombs": sum(event["type"] == "bombed" and event["player"] in opposing_seats
                                        for event in round_.events),
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

    return {
        "evaluated_client": f"{evaluated_client.__module__}:{evaluated_client.__name__}",
        "opposing_client": f"{opposing_client.__module__}:{opposing_client.__name__}",
        "trials_per_configuration": trials_per_configuration, "games": len(records), "seed": seed,
        "players": list(PLAYERS),
        "design": ("Six two-seat 2v2 configurations; every configuration uses the same fixed later-round "
                   "deal and starting-rank assignments. Clients are recreated for every single-round trial."),
        "summary": {
            "evaluated": summarize(records, "evaluated"),
            "opposing": summarize(records, "opposing"),
        },
        "records": records,
    }


def format_report(result: Mapping[str, Any]) -> str:
    lines = [
        f"{result['games']:,} 2v2 later-round trials | {result['evaluated_client']} vs {result['opposing_client']}",
        f"{result['trials_per_configuration']:,} trials per two-seat configuration | seed {result['seed']}", "",
    ]
    for team, label in (("evaluated", result["evaluated_client"]),
                        ("opposing", result["opposing_client"])):
        summary = result["summary"][team]
        low, high = summary["mean_placement_95_ci"]
        counts = summary["placement_counts"]
        lines.extend([
            label,
            f"  Team mean placement: {summary['mean_placement']:.3f} (95% CI {low:.3f} to {high:.3f})",
            "  Player placements: " + ", ".join(f"{place}: {counts[place]:,}" for place in range(1, 5)),
            "  2-rule finishes by initial position: " + ", ".join(
                f"{role}: {count:,}"
                for role, count in summary["twos_rule_by_initial_position"].items()
            ),
            "  Board clears by type: " + ", ".join(
                f"{BOARD_CLEAR_LABELS[clear_type]}: {percentage:.1f}%"
                for clear_type, percentage in summary["board_clear_type_percentages"].items()
            ),
            f"  Mean completion events per game: {summary['mean_completion_events']:.3f}",
            f"  Mean team actions per game: {summary['mean_actions']:.2f}",
            f"  Mean team bombs per game: {summary['mean_bombs']:.3f}",
            "  Final placement by initial rank:",
        ])
        for rank, rank_summary in summary["placement_by_initial_rank"].items():
            rank_low, rank_high = rank_summary["mean_placement_95_ci"]
            lines.append(
                f"    {rank}: mean {rank_summary['mean_placement']:.3f} "
                f"(95% CI {rank_low:.3f} to {rank_high:.3f}; n={rank_summary['games']:,})"
            )
        lines.append("")
    lines.append("Placement is averaged across each policy's two teammates in a game; lower is better.")
    return "\n".join(lines)


def run_free_for_all(client_classes: list[type[Agent]], *, trials: int = 10_000,
                     seed: int = 42, max_actions: int = 10_000,
                     progress: Optional[Any] = None) -> dict[str, Any]:
    """Run either one policy in all seats or one policy in each fixed seat."""
    if len(client_classes) == 1:
        client_classes *= 4
    if len(client_classes) != 4:
        raise ValueError("free-for-all requires one or four Agent classes")
    if trials <= 0 or max_actions <= 0:
        raise ValueError("trials and max_actions must be positive")
    master = random.Random(seed)
    role_rng = random.Random(master.getrandbits(64))
    deck_rng = random.Random(master.getrandbits(64))
    classes = dict(zip(PLAYERS, client_classes))
    records = []
    for trial, roles in enumerate(role_assignments(trials, role_rng), 1):
        round_ = Round(PLAYERS, first_round=False, roles=roles,
                       rng=random.Random(deck_rng.getrandbits(64)))
        play_round(round_, {player: classes[player]() for player in PLAYERS},
                   max_actions=max_actions, trial_label=f"trial {trial}")
        records.append({"trial": trial, "starting_roles": roles,
                        "placements": dict(round_.rankings)})
        if progress is not None:
            progress.update(1)
    summaries = {}
    for player in PLAYERS:
        placements = [record["placements"][player] for record in records]
        low, high = confidence_interval(placements)
        counts = Counter(placements)
        summaries[player] = {"mean_placement": sum(placements) / trials,
                             "mean_placement_95_ci": [low, high],
                             "placement_counts": {place: counts[place] for place in range(1, 5)}}
    return {"mode": "all_same" if len(set(client_classes)) == 1 else "1v1v1v1",
            "trials": trials, "seed": seed,
            "clients": {player: f"{classes[player].__module__}:{classes[player].__name__}"
                        for player in PLAYERS}, "summary": summaries, "records": records}


def format_free_for_all_report(result: Mapping[str, Any]) -> str:
    lines = [f"{result['trials']:,} later-round trials | {result['mode']} | seed {result['seed']}", ""]
    for player in PLAYERS:
        summary = result["summary"][player]
        low, high = summary["mean_placement_95_ci"]
        counts = summary["placement_counts"]
        lines.extend([
            f"{player}: {result['clients'][player]}",
            f"  Mean placement: {summary['mean_placement']:.3f} (95% CI {low:.3f} to {high:.3f})",
            "  Placements: " + ", ".join(f"{place}: {counts[place]:,}" for place in range(1, 5)), "",
        ])
    return "\n".join(lines)


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clients", nargs="+", help="one, two, or four package.module:ClassName agents")
    parser.add_argument("--trials-per-configuration", type=positive_int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-actions", type=positive_int, default=10_000)
    parser.add_argument("--output", type=Path, help="optional JSON file with trial records")
    args = parser.parse_args()
    if len(args.clients) not in {1, 2, 4}:
        parser.error("provide one agent (all four seats), two agents (2v2), or four agents (1v1v1v1)")
    try:
        from tqdm import tqdm
    except ImportError as error:
        parser.error("tqdm is required for this runner; install it with: pip install -r requirements.txt")
        raise error  # Satisfy type checkers; parser.error always exits.
    classes = [client_factory(specification) for specification in args.clients]
    games = len(TEAM_CONFIGURATIONS) * args.trials_per_configuration if len(classes) == 2 else args.trials_per_configuration
    with tqdm(total=games, desc="Running arena trials", unit="game") as progress:
        if len(classes) == 2:
            result = run_match(classes[0], classes[1], trials_per_configuration=args.trials_per_configuration,
                               seed=args.seed, max_actions=args.max_actions, progress=progress)
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
