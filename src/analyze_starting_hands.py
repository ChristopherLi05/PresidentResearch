"""Profile later-round hands before and after role trades.

The simulation JSON records each deal seed and the assigned roles, but not the
private draft/trade hands.  This module deterministically replays just those
two phases with ``BasicClient`` and reports the hand shapes by starting role.

Run from the repository root::

    python -m src.analyze_starting_hands results/role_simulation_5000.json
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
from typing import Any, Iterable

from src.client import BasicClient
from src.president import ROLE_NAMES, Round
from src.simulate import PLAYERS
from src.verify_simulation import load_result, verify_role_balance


STAGES = ("pre_trade", "post_trade")
GROUP_NAMES = (("single", 1), ("pair", 2), ("triple", 3), ("four_kind", 4))


def hand_features(hand: Iterable[Any]) -> dict[str, int]:
    """Count rank groups and 2s in one 13-card hand.

    ``single`` through ``four_kind`` count distinct ranks with exactly that
    multiplicity.  ``bomb_cards`` counts individual 2s, while ``has_bomb`` is
    one when the hand contains at least one 2.
    """
    counts = Counter(card.rank for card in hand)
    features = {name: sum(count == size for count in counts.values())
                for name, size in GROUP_NAMES}
    features["bomb_cards"] = counts["2"]
    features["has_bomb"] = int(counts["2"] > 0)
    return features


def _empty_totals() -> dict[str, int]:
    return {"hands": 0, **{name: 0 for name, _ in GROUP_NAMES},
            **{f"has_{name}": 0 for name, _ in GROUP_NAMES},
            "bomb_cards": 0, "has_bomb": 0}


def _add_hand(totals: dict[str, int], hand: Iterable[Any]) -> None:
    features = hand_features(hand)
    totals["hands"] += 1
    for name, _ in GROUP_NAMES:
        totals[name] += features[name]
        totals[f"has_{name}"] += int(features[name] > 0)
    totals["bomb_cards"] += features["bomb_cards"]
    totals["has_bomb"] += features["has_bomb"]


def analyze_result(result: dict[str, Any]) -> dict[str, Any]:
    """Replay drafts/trades and return hand-shape summaries by role and stage."""
    totals = {
        stage: {role: _empty_totals() for role in range(1, 5)}
        for stage in STAGES
    }

    for record in result["records"]:
        clients = {player: BasicClient() for player in PLAYERS}
        round_ = Round(
            PLAYERS,
            first_round=False,
            roles=record["starting_roles"],
            rng=random.Random(record["deck_seed"]),
        )

        while round_.phase == "draft":
            player = round_._actor()
            assert player is not None
            round_.apply_action(player, clients[player].respond(round_.message_for(player)))
        if round_.phase != "trade_request":
            raise RuntimeError(f"trial {record['trial']}: draft did not reach trading")
        for player in PLAYERS:
            _add_hand(totals["pre_trade"][record["starting_roles"][player]], round_.hands[player])

        while round_.phase != "play":
            player = round_._actor()
            assert player is not None
            round_.apply_action(player, clients[player].respond(round_.message_for(player)))
        for player in PLAYERS:
            _add_hand(totals["post_trade"][record["starting_roles"][player]], round_.hands[player])

    stages: dict[str, dict[str, dict[str, float | int]]] = {}
    for stage in STAGES:
        stage_summary: dict[str, dict[str, float | int]] = {}
        for role, name in enumerate(ROLE_NAMES, 1):
            bucket = totals[stage][role]
            hands = bucket["hands"]
            if not hands:
                raise ValueError("cannot summarize an empty result")
            summary: dict[str, float | int] = {"hands": hands}
            for group, _ in GROUP_NAMES:
                summary[f"mean_{group}_groups"] = bucket[group] / hands
                summary[f"hands_with_{group}_rate"] = bucket[f"has_{group}"] / hands
            summary["mean_bomb_cards"] = bucket["bomb_cards"] / hands
            summary["hands_with_bomb_rate"] = bucket["has_bomb"] / hands
            stage_summary[name] = summary
        stages[stage] = stage_summary

    return {
        "games": result["games"],
        "agent": result["agent"],
        "stages": stages,
    }


def format_report(analysis: dict[str, Any]) -> str:
    """Format an analysis as a compact, human-readable table."""
    lines = [
        f"{analysis['games']:,} later-round deals | {analysis['agent']}",
        "Group columns are mean rank groups per 13-card hand; hand columns are percentages.",
        "Bomb columns show mean 2s per hand and percentage of hands holding at least one 2.",
    ]
    for stage, title in (("pre_trade", "After draft, before trades"),
                         ("post_trade", "After all trades, before play")):
        lines.extend([
            "",
            title,
            f"{'Starting role':<16} {'S grp':>7} {'P grp':>7} {'T grp':>7} {'4 grp':>7}"
            f" {'S hand':>8} {'P hand':>8} {'T hand':>8} {'4 hand':>8} {'2s/hand':>9} {'has 2':>8}",
        ])
        for role in ROLE_NAMES:
            row = analysis["stages"][stage][role]
            lines.append(
                f"{role:<16}"
                f" {row['mean_single_groups']:>7.3f} {row['mean_pair_groups']:>7.3f}"
                f" {row['mean_triple_groups']:>7.3f} {row['mean_four_kind_groups']:>7.3f}"
                f" {row['hands_with_single_rate']:>7.1%} {row['hands_with_pair_rate']:>7.1%}"
                f" {row['hands_with_triple_rate']:>7.1%} {row['hands_with_four_kind_rate']:>7.1%}"
                f" {row['mean_bomb_cards']:>9.3f} {row['hands_with_bomb_rate']:>7.1%}"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", nargs="?", type=Path, default=Path("results/role_simulation_5000.json"),
        help="simulation JSON to analyze (default: results/role_simulation_5000.json)",
    )
    parser.add_argument("--output", type=Path, help="optional path for the JSON summary")
    args = parser.parse_args()

    result = load_result(args.path)
    verify_role_balance(result)
    analysis = analyze_result(result)
    print(format_report(analysis))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
        print(f"\nSummary saved to {args.output}")


if __name__ == "__main__":
    main()
