"""Reproducible shallow-search latency sample and balanced arena smoke test.

Run: python benchmark_agent_search.py --trials 5
"""
import argparse
from copy import deepcopy
import json
import random
from statistics import mean, median
from time import perf_counter

from arena import run_match, confidence_interval
from src.agent import ExperiencedAgent4, ExperiencedAgent3Tuned
from src.agent_search import heuristic_action
from src.president import Round


class HeuristicAgent4(ExperiencedAgent4):
    def __init__(self):
        super().__init__(search_worlds=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--compare-backends", action="store_true",
                        help="compare Python and C at identical fixed work before normal timing")
    args = parser.parse_args()
    if args.trials < 0:
        parser.error("--trials must be nonnegative (0 skips arena games)")
    policy = HeuristicAgent4()
    samples = {"early": [], "middle": [], "late": []}
    for seed in range(10):
        state = Round(list("abcd"), first_round=True, rng=random.Random(seed))
        while not state.done:
            player = state._actor()
            hand_size = len(state.hands[player])
            phase = "early" if hand_size >= 9 else "middle" if hand_size >= 5 else "late"
            if len(state.active) >= 3 and len(state.legal_actions(player)) > 1 and len(samples[phase]) < 25:
                samples[phase].append(deepcopy(state.message_for(player)))
            state.apply_action(player, heuristic_action(policy, state))
    if args.compare_backends:
        measurements = {"python": [], "c": []}
        decisions = {"python": [], "c": []}
        for index, message in enumerate(m for group in samples.values() for m in group):
            for backend in (("python", "c") if index % 2 else ("c", "python")):
                agent = ExperiencedAgent4(random.Random(42), search_time_ms=None,
                                          search_backend=backend, evaluation_backend=backend)
                decisions[backend].append(agent._choose_play(message, message["request"]["legal_actions"]))
                measurements[backend].append(agent.last_search["elapsed_ms"])
        print(json.dumps({"fixed_work": {backend: {
            "median_ms": median(times), "p95_ms": sorted(times)[int(.95*len(times))],
            "total_ms": sum(times)} for backend, times in measurements.items()},
            "speedup": sum(measurements["python"])/sum(measurements["c"]),
            "decision_mismatches": sum(a != b for a, b in zip(decisions["python"], decisions["c"]))}, indent=2), flush=True)
    latency = {}
    for phase, messages in samples.items():
        records = []
        for message in messages:
            agent = ExperiencedAgent4(random.Random(42))
            agent._choose_play(message, message["request"]["legal_actions"])
            records.append(agent.last_search)
        times = sorted(r["elapsed_ms"] for r in records)
        latency[phase] = {"samples": len(records), "median_ms": median(times),
                          "p95_ms": times[min(len(times) - 1, int(.95 * len(times)))],
                          "mean_completed_worlds": mean(r["worlds"] for r in records),
                          "fallback_fraction": mean(r["fallback"] for r in records)}
    print(json.dumps({"latency": latency}, indent=2), flush=True)
    if not args.trials:
        return
    for opponent in (HeuristicAgent4, ExperiencedAgent3Tuned):
        start = perf_counter()
        result = run_match(ExperiencedAgent4, opponent, trials_per_configuration=args.trials, seed=42)
        differences = []
        for record in result["records"]:
            seats = record["configuration"]
            placements = record["placements"]
            differences.append(mean(placements[p] for p in seats) -
                               mean(placements[p] for p in placements if p not in seats))
        print(json.dumps({"opponent": opponent.__name__, "games": len(differences),
                          "mean_placement_difference": mean(differences),
                          "difference_95_ci": confidence_interval(differences),
                          "seconds": perf_counter() - start}, indent=2), flush=True)


if __name__ == "__main__":
    main()
