"""Compare configurations of ExperiencedAgent4's search against one baseline.

Every variant plays the same fixed deals (same seed) in the six 2v2 seating
configurations of ``arena_multithread``.  Lowercase variant keys are
``ExperiencedAgent4`` constructor keywords, UPPERCASE keys override class-level
weights on a generated subclass, and ``danger=0`` zeroes the four NEXT_*_PENALTY
weights.  Search diagnostics are collected inside the worker processes.

    python tune_search.py --variant base:search_worlds=8 \\
        --variant deep:search_worlds=32,search_depth=none,danger=0 \\
        --baseline src.agent:ExperiencedAgent3Tuned --trials-per-configuration 400 --workers 12
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Optional

from arena import TEAM_CONFIGURATIONS, client_factory, positive_int, summarize
from arena_multithread import trial_record
from src.president_c import Round
from src.simulate import PLAYERS, play_round, role_assignments

DANGER_WEIGHTS = ("NEXT_RESPONSE_PENALTY", "NEXT_FINISH_PENALTY",
                  "NEXT_COMPLETION_PENALTY", "NEXT_BOMB_PENALTY")
CHUNK = 5  # trials per worker task


def parse_value(text: str) -> Any:
    lowered = text.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text.strip()


def parse_variant(spec: str) -> dict[str, Any]:
    """Parse ``NAME:key=value,...`` into ``{name, kwargs, weights}``."""
    name, _, body = spec.partition(":")
    if not name or not name.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"variant needs a NAME before ':', got {spec!r}")
    kwargs: dict[str, Any] = {}
    weights: dict[str, Any] = {}
    for item in filter(None, (part.strip() for part in body.split(","))):
        key, equals, text = item.partition("=")
        if not equals or not key:
            raise ValueError(f"variant {name!r}: expected key=value, got {item!r}")
        value = parse_value(text)
        if key == "danger":
            weights.update({weight: value for weight in DANGER_WEIGHTS})
        elif key.isupper():
            weights[key] = value
        else:
            kwargs[key] = value
    kwargs.setdefault("search_time_ms", None)
    return {"name": name, "kwargs": kwargs, "weights": weights}


def build_agent_class(variant: dict[str, Any]) -> type:
    """Return an ExperiencedAgent4 subclass with weights and kwargs baked in.

    Weights are plain class attributes, so the native backend still accepts it.
    Each choose call is recorded in ``cls.decisions``.
    """
    from src.agent import ExperiencedAgent4
    for weight in variant["weights"]:
        if not hasattr(ExperiencedAgent4, weight):
            raise ValueError(f"variant {variant['name']!r}: unknown weight {weight}")
    defaults = dict(variant["kwargs"])

    class Variant(ExperiencedAgent4):
        decisions: list = []

        def __init__(self, rng=None, **kwargs):
            super().__init__(rng, **{**defaults, **kwargs})

        def _choose_play(self, message, actions):
            action = super()._choose_play(message, actions)
            type(self).decisions.append(decision_record(self, message, actions, action))
            return action

    for weight, value in variant["weights"].items():
        setattr(Variant, weight, value)
    Variant.__name__ = Variant.__qualname__ = f"ExperiencedAgent4[{variant['name']}]"
    return Variant


def decision_record(agent, message, actions, action) -> list:
    """[reason, elapsed_ms, worlds, overrode, backend, searched] for one decision."""
    stats = agent.last_search
    reason = stats.get("reason", "none")
    searched = stats.get("candidates", 0) > 0
    overrode = stats.get("overrode")
    if overrode is None and searched:
        from src.agent_search import unique_actions
        first = min(unique_actions(actions), key=agent._play_scorer(message))
        overrode = action != first
    return [reason, stats.get("elapsed_ms", 0.0), stats.get("worlds", 0),
            bool(overrode), stats.get("backend", "python"), searched]


def check_variant(variant: dict[str, Any]) -> str:
    """Instantiate once; raise a clear error on bad arguments. Returns backend eligibility."""
    from src import agent_search_c
    cls = build_agent_class(variant)
    try:
        agent = cls()
    except (TypeError, ValueError) as error:
        raise ValueError(f"variant {variant['name']!r}: constructor rejected {variant['kwargs']}: {error}") from error
    return "native-eligible" if agent_search_c.supported(agent) else "python-only"


_classes: dict[str, type] = {}


def _play_chunk(task) -> tuple[int, int, list[dict[str, Any]]]:
    index, variant, baseline_spec, configuration, first, deals = task
    key = json.dumps(variant, sort_keys=True)
    if key not in _classes:
        _classes[key] = build_agent_class(variant)
    evaluated, baseline = _classes[key], client_factory(baseline_spec)
    seats = set(configuration)
    records = []
    for trial, (roles, deck_seed) in enumerate(deals, first):
        evaluated.decisions = []
        clients = {p: (evaluated() if p in seats else baseline()) for p in PLAYERS}
        round_ = Round(PLAYERS, first_round=False, roles=roles, rng=random.Random(deck_seed))
        try:
            actions = play_round(round_, clients, max_actions=10_000,
                                 trial_label=f"{variant['name']} {''.join(configuration)} trial {trial}")
            record = trial_record(configuration, trial, deck_seed, roles, round_, actions)
        finally:
            round_.close()
        record["search_decisions"] = evaluated.decisions
        records.append(record)
    return index, first, records


def make_deals(trials: int, seed: int):
    seed_rng = random.Random(seed)
    role_rng = random.Random(seed_rng.getrandbits(64))
    deck_rng = random.Random(seed_rng.getrandbits(64))
    roles = list(role_assignments(trials, role_rng))
    return list(zip(roles, [deck_rng.getrandbits(64) for _ in range(trials)]))


def search_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [d for record in records for d in record.get("search_decisions", [])]
    searched = [d for d in rows if d[5]]
    elapsed = sorted(d[1] for d in searched)
    p95 = elapsed[min(len(elapsed) - 1, math.ceil(0.95 * len(elapsed)) - 1)] if elapsed else 0.0
    n = len(searched)
    return {
        "choose_calls": len(rows),
        "search_decisions": n,
        "override_rate": sum(d[3] for d in searched) / n if n else 0.0,
        "mean_worlds": sum(d[2] for d in searched) / n if n else 0.0,
        "mean_elapsed_ms": sum(elapsed) / n if n else 0.0,
        "p95_elapsed_ms": p95,
        "reasons": dict(Counter(d[0] for d in rows)),
        "backends": dict(Counter(d[4] for d in searched)),
    }


def run_variant(variant, baseline_spec, deals, pool, progress=None) -> dict[str, Any]:
    tasks = [(c, variant, baseline_spec, configuration, first + 1, deals[first:first + CHUNK])
             for c, configuration in enumerate(TEAM_CONFIGURATIONS)
             for first in range(0, len(deals), CHUNK)]
    chunks = []
    for index, first, records in pool.map(_play_chunk, tasks):
        chunks.append(((index, first), records))
        if progress is not None:
            progress.update(len(records))
    records = [r for _, recs in sorted(chunks, key=lambda item: item[0]) for r in recs]
    summary = summarize(records, "evaluated")
    return {
        "name": variant["name"], "spec": variant,
        "summary": {k: summary[k] for k in ("games", "mean_placement", "mean_placement_95_ci", "placement_counts")},
        "search": search_diagnostics(records),
        "records": records,
    }


def format_table(results: list[dict[str, Any]]) -> str:
    header = (f"{'variant':<16}{'games':>6}{'mean':>8}{'95% CI':>18}{'1st/2nd/3rd/4th':>22}"
              f"{'decis':>8}{'ovr%':>6}{'worlds':>7}{'ms':>7}{'p95ms':>7}  backend")
    lines = [header]
    for r in sorted(results, key=lambda r: r["summary"]["mean_placement"]):
        s, d = r["summary"], r["search"]
        lo, hi = s["mean_placement_95_ci"]
        counts = "/".join(str(s["placement_counts"][p]) for p in range(1, 5))
        lines.append(
            f"{r['name']:<16}{s['games']:>6}{s['mean_placement']:>8.4f}{f'[{lo:.3f},{hi:.3f}]':>18}"
            f"{counts:>22}{d['search_decisions']:>8}{100 * d['override_rate']:>6.1f}"
            f"{d['mean_worlds']:>7.1f}{d['mean_elapsed_ms']:>7.2f}{d['p95_elapsed_ms']:>7.2f}  "
            f"{','.join(f'{k}:{v}' for k, v in d['backends'].items()) or '-'}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", action="append", required=True, metavar="NAME:key=value,...")
    parser.add_argument("--baseline", default="src.agent:ExperiencedAgent3Tuned")
    parser.add_argument("--trials-per-configuration", type=positive_int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=positive_int, default=12, help="total worker processes")
    parser.add_argument("--output", type=Path, help="JSON path (default results/search_tuning/<timestamp>.json)")
    args = parser.parse_args(argv)
    try:
        variants = [parse_variant(spec) for spec in args.variant]
        if len({v["name"] for v in variants}) != len(variants):
            raise ValueError("variant names must be unique")
        client_factory(args.baseline)
        plans = {v["name"]: check_variant(v) for v in variants}
    except ValueError as error:
        parser.error(str(error))
    try:
        from tqdm import tqdm
    except ImportError:
        parser.error("tqdm is required; use .venv/Scripts/python.exe")

    deals = make_deals(args.trials_per_configuration, args.seed)
    games = len(TEAM_CONFIGURATIONS) * len(deals)
    results, started = [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for variant in variants:
            begin = time.perf_counter()
            with tqdm(total=games, desc=variant["name"], unit="game") as progress:
                result = run_variant(variant, args.baseline, deals, pool, progress)
            result["planned_backend"] = plans[variant["name"]]
            result["wall_seconds"] = time.perf_counter() - begin
            results.append(result)
    print(format_table(results))
    for r in results:
        used = r["search"]["backends"]
        if r["planned_backend"] == "native-eligible" and set(used) == {"python"}:
            print(f"WARNING {r['name']}: eligible for native but searched on python (DLL missing or stale? "
                  "run python -m src.agent_search_c --build)")
        print(f"{r['name']}: {r['planned_backend']}, {r['wall_seconds']:.1f}s, reasons {r['search']['reasons']}")
    output = args.output or Path("results/search_tuning") / f"{datetime.now():%Y%m%d_%H%M%S}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline": args.baseline, "seed": args.seed, "workers": args.workers,
        "trials_per_configuration": args.trials_per_configuration,
        "variant_specs": args.variant, "wall_seconds": time.perf_counter() - started,
        "results": results,
    }
    output.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()
