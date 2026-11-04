"""Compare native/Python solver latency and the C-engine arena workload.

python benchmark_solver.py --trials 20 --positions 200 --repeats 3
Each arena run has six times --trials games; both backends use identical deals.
"""
import argparse
from copy import deepcopy
import json
import random
from statistics import median
from time import perf_counter
from unittest.mock import patch

from arena_multithread import run_match
from src.agent import ExperiencedAgent3Tuned, BasicAgent
from src.solver import solve_round
from src.solver_c import library


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--positions", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.trials, args.positions, args.repeats) < 1:
        parser.error("counts must be positive")
    if library() is None:
        parser.error("run python -m src.solver_c --build first")
    positions, seen = [], 0
    rng = random.Random(71)

    def collect(state):
        nonlocal seen
        seen += 1
        index = len(positions) if len(positions) < args.positions else rng.randrange(seen)
        if index < args.positions:
            snapshot = deepcopy(state)
            if index == len(positions):
                positions.append(snapshot)
            else:
                positions[index] = snapshot
        return solve_round(state, backend="c")

    with patch("src.agent.solve_round", side_effect=collect):
        reference = run_match(ExperiencedAgent3Tuned, BasicAgent,
                              trials_per_configuration=args.trials, seed=42, workers=1)
    print(json.dumps({"collected_solver_calls": seen, "sampled_positions": len(positions)}), flush=True)
    timings = {"python": [], "c": []}
    nodes, mismatches = [], 0
    for index, position in enumerate(positions):
        results = {}
        for backend in (("python", "c") if index % 2 else ("c", "python")):
            start = perf_counter()
            result = solve_round(position, backend=backend)
            timings[backend].append((perf_counter()-start)*1000)
            results[backend] = result
        mismatches += results["python"] != results["c"]
        nodes.append(results["c"].states_evaluated)
    print(json.dumps({"solver": {backend: {"total_ms": sum(values), "median_ms": median(values),
          "p95_ms": sorted(values)[min(len(values)-1, int(.95*len(values)))]} for backend, values in timings.items()},
          "speedup": sum(timings["python"])/sum(timings["c"]), "result_mismatches": mismatches,
          "largest_search_states": max(nodes)}, indent=2), flush=True)
    whole = {"python": [], "c": []}
    for repeat in range(args.repeats):
        for backend in (("python", "c") if repeat % 2 else ("c", "python")):
            def selected(state):
                return solve_round(state, backend=backend)
            with patch("src.agent.solve_round", new=selected):
                start = perf_counter()
                result = run_match(ExperiencedAgent3Tuned, BasicAgent,
                                   trials_per_configuration=args.trials, seed=42, workers=1)
                elapsed = perf_counter()-start
            if result["records"] != reference["records"]:
                raise AssertionError("backend changed game records")
            whole[backend].append(elapsed)
            print(json.dumps({"repeat": repeat+1, "backend": backend, "seconds": elapsed,
                              "games_per_second": 6*args.trials/elapsed}), flush=True)
    print(json.dumps({"whole_game_median_seconds": {k: median(v) for k, v in whole.items()},
                      "whole_game_speedup": median(whole["python"])/median(whole["c"]),
                      "game_records_identical": True}, indent=2), flush=True)


if __name__ == "__main__":
    main()
