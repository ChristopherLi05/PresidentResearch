"""Tune ExperiencedAgent2/3/4 weights using balanced, paired arena deals.

Run ``python optimize_weights.py --help`` for options.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import json
import math
from pathlib import Path
import random

from arena import run_match
from src.agent import ExperiencedAgent2, ExperiencedAgent3, ExperiencedAgent4
from tqdm import tqdm


@contextmanager
def progress_bar(total, description, enabled=True, unit="round"):
    bar = tqdm(total=total, desc=description, unit=unit, dynamic_ncols=True) if enabled else None
    try:
        yield bar
    finally:
        if bar is not None:
            bar.close()


# Nonnegative search bounds; terminal outcomes and game rules stay fixed.
# Shared hand-policy weights. Agent4 adds its turn-order weights below.
BOUNDS = {
    "CONTROL_RESERVE_WEIGHT": (0.0, 6.0),
    "GROUP_WEIGHT": (0.5, 6.0),
    "CARD_WEIGHT": (0.0, 1.0),
    "SINGLE_WEIGHT": (0.0, 2.0),
    "HAND_CONTROL_WEIGHT": (0.0, 5.0),
    "BOMB_GROUP_PENALTY": (0.0, 8.0),
    "BOMB_COVER_PENALTY": (0.0, 15.0),
    "BOMB_VALUE": (0.0, 3.0),
    "LEAD_CONTROL_WEIGHT": (0.0, 5.0),
    "BOMB_STRANDING_PENALTY": (0.0, 24.0),
    "PASS_COST": (0.0, 2.0),
    "PASS_DANGER_WEIGHT": (0.0, 8.0),
    "CLEAR_COST": (0.0, 2.0),
    "OPPONENT_FINISH_PENALTY": (0.0, 36.0),
    "TRADE_BOMB_COST": (0.0, 5.0),
    "TRADE_RANK_WEIGHT": (0.0, 0.15),
}
AGENT_EXTRA_BOUNDS = {
    ExperiencedAgent4: {
        "NEXT_RESPONSE_PENALTY": (0.0, 6.0),
        "NEXT_FINISH_PENALTY": (0.0, 24.0),
        "NEXT_COMPLETION_PENALTY": (0.0, 8.0),
        "NEXT_BOMB_PENALTY": (0.0, 8.0),
    },
}
AGENTS = {"ExperiencedAgent2": ExperiencedAgent2, "ExperiencedAgent3": ExperiencedAgent3,
          "ExperiencedAgent4": ExperiencedAgent4}


def bounds_for(base):
    """Return exactly the weights implemented by a candidate agent class."""
    return {**BOUNDS, **AGENT_EXTRA_BOUNDS.get(base, {})}


def make_agent(base, weights):
    """Use a new class per candidate so cached costs cannot mix weights."""
    bounds = bounds_for(base)
    if set(weights) != set(bounds):
        raise ValueError("weights must contain exactly the supported weight names")
    for name, value in weights.items():
        low, high = bounds[name]
        if not isinstance(value, (float, int)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} must be finite and within [{low}, {high}]")
    return type("OptimizedAgent", (base,), dict(weights))


def evaluate(base, weights, opponents, trials, seed, max_actions, progress=None):
    candidate = make_agent(base, weights)
    scores = {}
    try:
        for opponent in opponents:
            result = run_match(candidate, opponent, trials_per_configuration=trials,
                               seed=seed, max_actions=max_actions, progress=progress)
            scores[opponent.__name__] = result["summary"]["evaluated"]["mean_placement"]
    finally:
        # Bounded caches otherwise retain instances/classes from old candidates.
        candidate._hand_cost_cached.cache_clear()
        candidate._lead_cost_cached.cache_clear()
        candidate._control_cached.cache_clear()
    return {"mean_placement": sum(scores.values()) / len(scores), "by_opponent": scores}


def propose(best, rng, step, bounds=BOUNDS):
    weights = dict(best)
    # Sparse mutations help identify improvements without changing everything.
    for name in rng.sample(list(bounds), rng.randint(1, 4)):
        low, high = bounds[name]
        weights[name] = min(high, max(low, weights[name] + rng.gauss(0, step * (high - low))))
    return weights


def normalized_weights(weights, bounds=BOUNDS):
    return [(weights[name] - low) / (high - low) for name, (low, high) in bounds.items()]


def decode_weights(vector, bounds=BOUNDS):
    # ES estimates the smoothed objective including this boundary clipping.
    return {name: low + min(1.0, max(0.0, x)) * (high - low)
            for x, (name, (low, high)) in zip(vector, bounds.items())}


def es_update(center, directions, scores, sigma, learning_rate):
    """Ascend negative placement using mirrored Gaussian perturbations."""
    gradient = [sum((minus["mean_placement"] - plus["mean_placement"]) * direction[j]
                    for direction, (plus, minus) in zip(directions, scores))
                / (2 * len(directions) * sigma) for j in range(len(center))]
    return [min(1.0, max(0.0, x + learning_rate * g)) for x, g in zip(center, gradient)]


def evaluate_job(job):
    """Top-level entry point supports Windows process spawning."""
    return evaluate(*job)


def run_es(args, base, opponents, initial, rng, report, measure, bounds, resume=False):
    benchmark_seed = report["train_seed"]
    if resume:
        best = dict(report["best_weights"])
        best_score = dict(report["benchmark"])
        center = normalized_weights(report["center_weights"], bounds)
        first_generation = report["next_generation"]
        used_seeds = {benchmark_seed, report["validation_seed"]} | {
            item["training_seed"] for item in report["history"] if "training_seed" in item
        }
    else:
        best = dict(initial)
        best_score = measure(best, args.benchmark_trials, benchmark_seed)
        center = normalized_weights(initial, bounds)
        first_generation = 1
        used_seeds = {benchmark_seed, report["validation_seed"]}
        report.update(method="es", pairs=args.pairs, sigma=args.sigma, learning_rate=args.learning_rate,
                      workers=args.workers, benchmark_seed=benchmark_seed,
                      benchmark_trials_per_configuration=args.benchmark_trials,
                      best_weights=best, benchmark=best_score, center_weights=initial,
                      next_generation=first_generation, rng_state=rng.getstate())
        report["history"].append({"iteration": 0, "weights": best, "benchmark": best_score})
        save_report(args.output, report)
    pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        final_generation = first_generation + args.iterations
        for generation in range(first_generation, final_generation):
            seed = rng.getrandbits(64)
            while seed in used_seeds:
                seed = rng.getrandbits(64)
            used_seeds.add(seed)
            directions = [[rng.gauss(0, 1) for _ in center] for _ in range(args.pairs)]
            candidates = [decode_weights([x + sign * args.sigma * e for x, e in zip(center, direction)], bounds)
                          for direction in directions for sign in (1, -1)]
            jobs = [(base, weights, opponents, args.trials, seed, args.max_actions) for weights in candidates]
            print(f"Generation {generation}/{final_generation - 1}: evaluating {2 * args.pairs} perturbations...", flush=True)
            with progress_bar(len(jobs), "Perturbations", not args.no_progress, "candidate") as bar:
                scores = [None] * len(jobs)
                if pool:
                    futures = {pool.submit(evaluate_job, job): i for i, job in enumerate(jobs)}
                    for future in as_completed(futures):
                        # Completion order must not scramble mirrored pairs.
                        scores[futures[future]] = future.result()
                        if bar is not None:
                            bar.update(1)
                else:
                    for i, job in enumerate(jobs):
                        scores[i] = evaluate_job(job)
                        if bar is not None:
                            bar.update(1)
            paired_scores = list(zip(scores[::2], scores[1::2]))
            center = es_update(center, directions, paired_scores, args.sigma, args.learning_rate)
            weights = decode_weights(center, bounds)
            score = measure(weights, args.benchmark_trials, benchmark_seed)
            if score["mean_placement"] < best_score["mean_placement"]:
                best, best_score = weights, score
            report["history"].append({"iteration": generation, "training_seed": seed,
                                      "directions": directions, "perturbation_scores": scores,
                                      "weights": weights, "benchmark": score})
            report.update(best_weights=best, benchmark=best_score, center_weights=weights,
                          next_generation=generation + 1, rng_state=rng.getstate())
            save_report(args.output, report)
            print(f"Benchmark {score['mean_placement']:.4f}; best {best_score['mean_placement']:.4f}", flush=True)
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
    return best


def save_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def export_agent(path, base, weights):
    lines = ["# Generated by optimize_weights.py; see the adjacent JSON report.",
             f"from src.agent import {base.__name__}", "", "",
             f"class OptimizedAgent({base.__name__}):"]
    lines.extend(f"    {name} = {value!r}" for name, value in weights.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def positive_float(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def tuples(value):
    """Restore random.Random's tuple-based state after JSON decoding."""
    if isinstance(value, list):
        return tuple(tuples(item) for item in value)
    return value


def load_resume(path, parser):
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        configuration = report["configuration"]
        if report["method"] != "es" or "rng_state" not in report or "next_generation" not in report:
            raise ValueError("checkpoint is not resumable ES output")
        return report, configuration
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"invalid --resume checkpoint: {error}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("es", "hill"), default="es")
    parser.add_argument("--no-progress", action="store_true", help="disable tqdm progress bars")
    parser.add_argument("--agent", choices=AGENTS, default="ExperiencedAgent3")
    parser.add_argument("--opponents", choices=AGENTS, nargs="+", default=["ExperiencedAgent2"])
    parser.add_argument("--iterations", type=positive_int, default=30,
                        help="ES generations to run (additional when resuming), or hill-climb candidates")
    parser.add_argument("--pairs", type=positive_int, default=16, help="mirrored perturbation pairs per ES generation")
    parser.add_argument("--sigma", type=positive_float, default=0.1, help="ES noise in normalized weight space")
    parser.add_argument("--learning-rate", type=positive_float, default=0.05)
    parser.add_argument("--workers", type=positive_int, default=1, help="ES evaluation processes")
    parser.add_argument("--benchmark-trials", type=positive_int, default=40, help="fixed checkpoint-selection deals per seat pair")
    parser.add_argument("--trials", type=positive_int, default=20, help="training deals per each of six seat pairs")
    parser.add_argument("--validation-trials", type=positive_int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-actions", type=positive_int, default=10000)
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument("--initial", type=Path, help="start a new search from best_weights in a previous JSON report")
    checkpoint.add_argument("--resume", type=Path, help="continue a resumable ES checkpoint")
    parser.add_argument("--output", type=Path, default=Path("results/optimized_agent.json"),
                        help="JSON checkpoint; also writes a sibling .py agent")
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != ".json":
        parser.error("--output must end in .json")
    resumed_report = None
    if args.resume:
        resumed_report, configuration = load_resume(args.resume, parser)
        for name, value in configuration.items():
            setattr(args, name, value)
    base = AGENTS[args.agent]
    opponents = [AGENTS[name] for name in dict.fromkeys(args.opponents)]
    bounds = bounds_for(base)
    defaults = {name: getattr(base, name) for name in bounds}
    initial = defaults.copy()
    if args.initial:
        try:
            initial = json.loads(args.initial.read_text(encoding="utf-8"))["best_weights"]
            make_agent(base, initial)
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(f"invalid --initial: {error}")
    if resumed_report:
        rng = random.Random()
        rng.setstate(tuples(resumed_report["rng_state"]))
        report = resumed_report
        report.update(status="searching", workers=args.workers)
        train_seed, validation_seed = report["train_seed"], report["validation_seed"]
    else:
        rng = random.Random(args.seed)
        train_seed = rng.getrandbits(64)
        validation_seed = rng.getrandbits(64)
        while validation_seed == train_seed:
            validation_seed = rng.getrandbits(64)
        configuration = {"agent": args.agent, "opponents": args.opponents, "pairs": args.pairs,
                         "sigma": args.sigma, "learning_rate": args.learning_rate,
                         "benchmark_trials": args.benchmark_trials, "trials": args.trials,
                         "validation_trials": args.validation_trials, "max_actions": args.max_actions,
                         "seed": args.seed, "method": args.method}
        report = {"method": args.method, "agent": args.agent, "opponents": [op.__name__ for op in opponents],
                  "seed": args.seed, "train_seed": train_seed, "validation_seed": validation_seed,
                  "trials_per_configuration": args.trials, "validation_trials_per_configuration": args.validation_trials,
                  "max_actions": args.max_actions, "bounds": bounds, "configuration": configuration,
                  "history": [], "status": "searching"}

    def measure(weights, trials, seed):
        description = "Validation" if seed == validation_seed else "Benchmark" if args.method == "es" else "Evaluation"
        with progress_bar(6 * trials * len(opponents), description, not args.no_progress) as bar:
            return evaluate(base, weights, opponents, trials, seed, args.max_actions, progress=bar)

    print(f"Each training evaluation runs {6 * args.trials * len(opponents)} rounds; lower placement is better.", flush=True)
    print("Resuming checkpoint..." if resumed_report else "Evaluating initial weights...", flush=True)
    if args.method == "es":
        best = run_es(args, base, opponents, initial, rng, report, measure, bounds, bool(resumed_report))
    else:
        best = run_hill(args, initial, rng, report, measure, train_seed, bounds)

    report["status"] = "validating"
    save_report(args.output, report)
    print("Validating defaults and selected weights on separate deals...", flush=True)
    baseline = measure(defaults, args.validation_trials, validation_seed)
    selected = measure(best, args.validation_trials, validation_seed)
    delta = selected["mean_placement"] - baseline["mean_placement"]
    report.update(status="complete", validation={"defaults": baseline, "selected": selected,
                                                "placement_delta": delta})
    save_report(args.output, report)
    export_agent(args.output.with_suffix(".py"), base, best)
    print(f"Validation: defaults {baseline['mean_placement']:.4f}, selected {selected['mean_placement']:.4f}, delta {delta:+.4f}")
    if delta >= 0:
        print("Selected weights did not improve validation placement; defaults remain unchanged.")
    print(f"Saved {args.output} and {args.output.with_suffix('.py')}")
    return 0


def run_hill(args, initial, rng, report, measure, train_seed, bounds):
    best = dict(initial)
    best_score = measure(best, args.trials, train_seed)
    report["history"].append({"iteration": 0, "weights": dict(best), **best_score})
    report.update(best_weights=best, training=best_score)
    save_report(args.output, report)
    for iteration in range(1, args.iterations + 1):
        # Anneal perturbation size from 20% toward 3% of each allowed range.
        step = 0.20 * (0.03 / 0.20) ** ((iteration - 1) / max(1, args.iterations - 1))
        candidate = propose(best, rng, step, bounds)
        score = measure(candidate, args.trials, train_seed)
        improved = score["mean_placement"] < best_score["mean_placement"]
        if improved:
            best, best_score = candidate, score
        report["history"].append({"iteration": iteration, "weights": candidate, **score})
        report.update(best_weights=best, training=best_score)
        save_report(args.output, report)
        print(f"{iteration}/{args.iterations}: {score['mean_placement']:.4f}; best {best_score['mean_placement']:.4f}"
              + (" improved" if improved else ""), flush=True)

    return best


if __name__ == "__main__":
    raise SystemExit(main())
