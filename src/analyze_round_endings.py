"""Analyze how simulated President rounds and board clears resolve.

The simulation result files retain each deal's seed and starting roles but not
the event log.  This module deterministically replays those deals with the
recorded BasicClient policy, then classifies the resulting events.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping

from src.agent import BasicAgent
from src.president import ROLE_NAMES, Round


FINISH_ORDER = (
    "ordinary_play",
    "completion",
    "played_a_2_bomb",
    "only_2s_penalty",
    "last_remaining_player",
)
CLEAR_ORDER = ("completion", "pass_around", "2_bomb", "four_of_a_kind_play")
TERMINAL_ORDER = ("play", "complete", "bomb")

FINISH_LABELS = {
    "ordinary_play": "Ordinary play emptied hand",
    "completion": "Completion emptied hand",
    "played_a_2_bomb": "2 bomb emptied hand",
    "only_2s_penalty": "Only 2s remained (penalty)",
    "last_remaining_player": "Last active player",
}
CLEAR_LABELS = {
    "completion": "Completion",
    "pass_around": "Pass-around",
    "2_bomb": "2 bomb",
    "four_of_a_kind_play": "Four-of-a-kind play",
}
TERMINAL_LABELS = {
    "play": "Ordinary play",
    "complete": "Completion",
    "bomb": "2 bomb",
}


def _last_gameplay_event(events: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Return the action event responsible for the following board clear."""
    for event in reversed(list(events)):
        if event["type"] in {"passed", "played", "opening_play", "completed", "bombed"}:
            return event
    raise RuntimeError("board clear had no preceding gameplay event")


def analyze(records: Iterable[Mapping[str, Any]], players: Iterable[str]) -> dict[str, Any]:
    """Replay *records* and return counts for round endings and board clears."""
    players = tuple(players)
    terminal_actions: Counter[str] = Counter()
    terminal_twos: Counter[str] = Counter()
    placement_mechanisms: Counter[str] = Counter()
    board_clears: Counter[str] = Counter()
    all_actions: Counter[str] = Counter()
    twos_by_starting_role: Counter[int] = Counter()
    completion_events = 0
    board_round_lengths: list[int] = []
    board_round_decisions: list[int] = []
    terminal_segment_lengths: list[int] = []
    games = 0

    for record in records:
        games += 1
        round_ = Round(
            players,
            first_round=False,
            roles=record["starting_roles"],
            rng=random.Random(record["deck_seed"]),
        )
        clients = {player: BasicAgent() for player in players}
        responses_since_clear = 0
        decisions_since_clear = 0
        while not round_.done:
            player = round_._actor()
            if player is None:
                raise RuntimeError("round requested no player before finishing")
            event_start = len(round_.events)
            was_board_phase = round_.phase in {"play", "completion"}
            # BasicClient needs only the legal actions and public history.
            # Avoid ``message_for`` here: it defensively deep-copies the full
            # event log on every response, which is unnecessary in this
            # internal, read-only replay and makes large analyses very slow.
            message = {
                "player": player,
                "events": round_.events,
                "request": {"type": round_.phase, "legal_actions": round_.legal_actions(player)},
            }
            action = clients[player].respond(message)
            round_.apply_action(player, action)
            new_events = round_.events[event_start:]
            all_actions[action["type"]] += 1
            if was_board_phase:
                # This action began in a board-play phase.  Completion
                # declines are responses, but not a play decision.
                responses_since_clear += 1
                decisions_since_clear += action["type"] != "decline_completion"
            completion_events += sum(event["type"] == "completed" for event in new_events)
            if round_.done:
                terminal_actions[action["type"]] += 1
                terminal_twos[action["type"]] += sum(
                    event["type"] == "twos_announced" for event in new_events
                )
            if any(event["type"] == "board_cleared" for event in new_events):
                board_round_lengths.append(responses_since_clear)
                board_round_decisions.append(decisions_since_clear)
                responses_since_clear = 0
                decisions_since_clear = 0

        # The final segment ends the game without a subsequent board clear.
        terminal_segment_lengths.append(responses_since_clear)

        for index, event in enumerate(round_.events):
            if event["type"] == "twos_announced":
                placement_mechanisms["only_2s_penalty"] += 1
                twos_by_starting_role[record["starting_roles"][event["player"]]] += 1
            elif event["type"] == "player_done":
                if event.get("last_active"):
                    placement_mechanisms["last_remaining_player"] += 1
                elif event.get("bomb_finish"):
                    placement_mechanisms["played_a_2_bomb"] += 1
                else:
                    previous = next(
                        (
                            candidate
                            for candidate in reversed(round_.events[:index])
                            if candidate.get("player") == event["player"]
                            and candidate["type"] in {"played", "opening_play", "completed", "bombed"}
                        ),
                        None,
                    )
                    placement_mechanisms[
                        "completion" if previous and previous["type"] == "completed" else "ordinary_play"
                    ] += 1
            elif event["type"] == "board_cleared":
                previous = _last_gameplay_event(round_.events[:index])
                cause = {
                    "passed": "pass_around",
                    "completed": "completion",
                    "bombed": "2_bomb",
                    "played": "four_of_a_kind_play",
                    "opening_play": "four_of_a_kind_play",
                }[previous["type"]]
                board_clears[cause] += 1

    return {
        "games": games,
        "players_per_game": len(players),
        "terminal_actions": dict(terminal_actions),
        "terminal_actions_with_twos_announced": dict(terminal_twos),
        "placement_mechanisms": dict(placement_mechanisms),
        "board_clears": dict(board_clears),
        "completion_events": completion_events,
        "actions": dict(all_actions),
        "twos_by_starting_role": {
            ROLE_NAMES[role - 1]: twos_by_starting_role[role] for role in range(1, 5)
        },
        "board_rounds": {
            "completed": len(board_round_lengths),
            "mean_responses": sum(board_round_lengths) / len(board_round_lengths),
            "mean_decisions": sum(board_round_decisions) / len(board_round_decisions),
            "minimum_responses": min(board_round_lengths),
            "maximum_responses": max(board_round_lengths),
            "terminal_segments": len(terminal_segment_lengths),
            "mean_terminal_segment_responses": sum(terminal_segment_lengths) / len(terminal_segment_lengths),
        },
    }


def _table(title: str, counts: Mapping[str, int], order: Iterable[str], labels: Mapping[str, str], total: int) -> list[str]:
    lines = [title, f"{'Outcome':<33} {'Count':>8} {'Share':>9}"]
    lines.append(f"{'-' * 33} {'-' * 8} {'-' * 9}")
    for key in order:
        count = counts.get(key, 0)
        lines.append(f"{labels[key]:<33} {count:>8,} {count / total:>8.1%}")
    lines.append(f"{'Total':<33} {total:>8,} {'100.0%':>9}")
    return lines


def format_report(result: Mapping[str, Any]) -> str:
    """Format the analysis as compact, copyable text tables."""
    games = result["games"]
    player_slots = games * result["players_per_game"]
    clear_total = sum(result["board_clears"].values())
    lines = [
        f"{games:,} later-round replays | BasicClient",
        "A pass-around resets the board; it cannot itself finish a round.",
        "A board round starts on a cleared board and ends at the next board clear; draft/trade setup is excluded.",
        "",
    ]
    board_rounds = result["board_rounds"]
    lines.extend([
        "Board-round length",
        f"Completed clear-to-clear rounds: {board_rounds['completed']:,} ({board_rounds['completed'] / games:.2f} per game)",
        f"Mean responses per board round: {board_rounds['mean_responses']:.2f}",
        f"Mean play decisions per board round: {board_rounds['mean_decisions']:.2f}",
        f"Range of responses: {board_rounds['minimum_responses']} to {board_rounds['maximum_responses']}",
        "",
    ])
    lines.extend(_table("Final action that finished the round", result["terminal_actions"], TERMINAL_ORDER, TERMINAL_LABELS, games))
    lines.append("")
    lines.extend(_table("How player placements were assigned", result["placement_mechanisms"], FINISH_ORDER, FINISH_LABELS, player_slots))
    lines.append("")
    twos = result["twos_by_starting_role"]
    lines.extend([
        "Only-2s rule by starting role",
        f"{'Starting role':<33} {'Count':>8} {'Rate':>9}",
        f"{'-' * 33} {'-' * 8} {'-' * 9}",
    ])
    for role in ROLE_NAMES:
        count = twos[role]
        lines.append(f"{role:<33} {count:>8,} {count / games:>8.1%}")
    lines.append(f"{'Total':<33} {sum(twos.values()):>8,} {sum(twos.values()) / player_slots:>8.1%}")
    lines.append("")
    lines.extend(_table("What cleared the board", result["board_clears"], CLEAR_ORDER, CLEAR_LABELS, clear_total))
    lines.extend([
        "",
        f"Completion events: {result['completion_events']:,}",
        "Terminal actions that also announced one or more only-2s penalties: "
        f"{sum(result['terminal_actions_with_twos_announced'].values()):,}",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("results/role_simulation_5000.json"),
        help="simulation JSON produced by src.simulate (default: results/role_simulation_5000.json)",
    )
    parser.add_argument("--output", type=Path, help="optional JSON file for the raw analysis counts")
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    result = analyze(source["records"], source["players"])
    print(format_report(result))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"\nAnalysis saved to {args.output}")


if __name__ == "__main__":
    main()
