"""Test post-trade hand-shape differences between starting roles.

Each simulated deal contains one player in each role.  This script replays
only the draft and trade phases, then performs paired, two-sided tests between
roles within each deal.  Pairing controls for the shared deal and trade state.

Run from the repository root::

    python -m src.analyze_hand_significance results/role_simulation_5000.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
from statistics import NormalDist
from typing import Any

from src.analyze_starting_hands import hand_features
from src.agent import BasicAgent
from src.president import ROLE_NAMES, Round
from src.simulate import PLAYERS
from src.verify_simulation import load_result, verify_role_balance


METRICS = ("single", "pair", "triple", "four_kind", "bomb_cards")
ROLE_PAIRS = tuple((left, right) for left in range(1, 5) for right in range(left + 1, 5))


def post_trade_samples(result: dict[str, Any]) -> dict[int, dict[str, list[int]]]:
    """Return one post-trade hand-feature observation per role and deal."""
    samples = {
        role: {metric: [] for metric in METRICS}
        for role in range(1, 5)
    }
    for record in result["records"]:
        clients = {player: BasicAgent() for player in PLAYERS}
        round_ = Round(
            PLAYERS,
            first_round=False,
            roles=record["starting_roles"],
            rng=random.Random(record["deck_seed"]),
        )
        while round_.phase != "play":
            player = round_._actor()
            assert player is not None
            round_.apply_action(player, clients[player].respond(round_.message_for(player)))

        for player in PLAYERS:
            role = record["starting_roles"][player]
            features = hand_features(round_.hands[player])
            for metric in METRICS:
                samples[role][metric].append(features[metric])
    return samples


def paired_test(left: list[int], right: list[int]) -> dict[str, float | int]:
    """Large-sample paired test and confidence interval for a mean difference.

    The metrics are bounded integer counts.  With 5,000 paired deals, the
    normal approximation to the paired-mean sampling distribution is very
    accurate; no independent-observation assumption is made between roles.
    """
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("paired samples must have the same length of at least two")
    differences = [a - b for a, b in zip(left, right)]
    count = len(differences)
    mean = sum(differences) / count
    variance = sum((value - mean) ** 2 for value in differences) / (count - 1)
    standard_error = math.sqrt(variance / count)
    if standard_error == 0:
        statistic = math.inf if mean else 0.0
        p_value = 0.0 if mean else 1.0
        effect = math.inf if mean else 0.0
    else:
        statistic = mean / standard_error
        p_value = math.erfc(abs(statistic) / math.sqrt(2))
        effect = mean / math.sqrt(variance)  # paired standardized mean difference (d_z)
    margin = NormalDist().inv_cdf(0.975) * standard_error
    return {
        "n": count,
        "mean_difference": mean,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
        "z": statistic,
        "p_value": p_value,
        "paired_effect_dz": effect,
    }


def holm_adjust(rows: list[dict[str, Any]]) -> None:
    """Add Holm-adjusted p-values in place for one family of comparisons."""
    ordered = sorted(enumerate(rows), key=lambda item: item[1]["p_value"])
    running = 0.0
    family_size = len(rows)
    for position, (index, row) in enumerate(ordered):
        adjusted = min(1.0, (family_size - position) * row["p_value"])
        running = max(running, adjusted)
        rows[index]["p_holm"] = running


def analyze_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Run all pairwise post-trade role comparisons, corrected per metric."""
    samples = post_trade_samples(result)
    rows: list[dict[str, Any]] = []
    by_metric: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for metric in METRICS:
        for left, right in ROLE_PAIRS:
            row: dict[str, Any] = {
                "metric": metric,
                "role_a": ROLE_NAMES[left - 1],
                "role_b": ROLE_NAMES[right - 1],
                "difference_definition": "role_a minus role_b",
            }
            row.update(paired_test(samples[left][metric], samples[right][metric]))
            rows.append(row)
            by_metric[metric].append(row)
    for metric_rows in by_metric.values():
        holm_adjust(metric_rows)
        for row in metric_rows:
            row["significant_at_0_05"] = row["p_holm"] < 0.05
    return rows


def _format_p(value: float) -> str:
    return "<1e-300" if value == 0 else f"{value:.3g}"


def format_report(rows: list[dict[str, Any]]) -> str:
    """Format pairwise results as a tab-separated report suitable for Excel."""
    lines = [
        "Paired post-trade tests; difference = Role A minus Role B.",
        "Holm correction is applied across the six role-pair comparisons within each metric.",
        "Metric\tRole A\tRole B\tN\tMean difference\t95% CI\td_z\tp (raw)\tp (Holm)\tSignificant (0.05)",
    ]
    for row in rows:
        lines.append(
            f"{row['metric']}\t{row['role_a']}\t{row['role_b']}\t{row['n']}"
            f"\t{row['mean_difference']:.4f}"
            f"\t[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]"
            f"\t{row['paired_effect_dz']:.3f}"
            f"\t{_format_p(row['p_value'])}\t{_format_p(row['p_holm'])}"
            f"\t{'yes' if row['significant_at_0_05'] else 'no'}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", nargs="?", type=Path, default=Path("results/role_simulation_5000.json"),
        help="simulation JSON to analyze (default: results/role_simulation_5000.json)",
    )
    parser.add_argument("--output", type=Path, help="optional JSON path for detailed test results")
    args = parser.parse_args()

    result = load_result(args.path)
    verify_role_balance(result)
    rows = analyze_result(result)
    print(format_report(rows))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"\nTest results saved to {args.output}")


if __name__ == "__main__":
    main()
