"""Unattended overnight run: confirm search variants, then compare against every agent.

    .venv/Scripts/python.exe overnight_search.py
    .venv/Scripts/python.exe overnight_search.py --smoke   # 1-trial dry run of both steps

Step 1 runs tune_search.py on a fresh seed against ExperiencedAgent3Tuned.
Step 2 runs compare_agent.py for search_variants:ExperiencedAgent4Full64M2
against every built-in agent. Step 2 runs even if step 1 fails. Everything
is logged under results/overnight/<timestamp>/.
"""

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
FULL = "search_depth=none,search_through_own_turns=true,danger=0"
VARIANTS = [
    "control:search_worlds=0,danger=0",
    "shipped:search_worlds=8",
    f"full64m2:search_worlds=64,search_margin=2,{FULL}",
    f"full64m2c8:search_worlds=64,search_margin=2,search_candidates=8,{FULL}",
    f"full128m1:search_worlds=128,search_margin=1,{FULL}",
]
COMPARE_AGENT = "search_variants:ExperiencedAgent4Full64M2"


def check_native():
    """Fail fast if the native search library is missing or stale."""
    from src.agent import ExperiencedAgent4
    from src import agent_search_c
    agent_search_c.backend(ExperiencedAgent4(search_backend="c"))


def run(name, command, folder):
    log = folder / f"{name}.log"
    print(f"[{datetime.now():%H:%M:%S}] {name}: {' '.join(command[1:])}\n    log: {log}", flush=True)
    start = time.perf_counter()
    with log.open("w", encoding="utf-8") as handle:
        code = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT).returncode
    print(f"[{datetime.now():%H:%M:%S}] {name}: exit {code} after {(time.perf_counter()-start)/60:.1f} min",
          flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="1 trial per configuration, to check the pipeline")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tune-trials", type=int, default=1000)
    parser.add_argument("--compare-trials", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=12, help="tune_search worker processes")
    parser.add_argument("--parallel-matchups", type=int, default=6, help="compare_agent concurrent opponents")
    args = parser.parse_args()
    tune_trials, compare_trials = (1, 1) if args.smoke else (args.tune_trials, args.compare_trials)

    try:
        check_native()
    except RuntimeError as error:
        sys.exit(f"Native search unavailable: {error}\nRun: .venv/Scripts/python.exe -m src.agent_search_c --build")

    folder = ROOT / "results" / "overnight" / f"{datetime.now():%Y%m%d_%H%M%S}{'_smoke' if args.smoke else ''}"
    folder.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    tune = [python, "tune_search.py", "--seed", str(args.seed), "--trials-per-configuration", str(tune_trials),
            "--workers", str(args.workers), "--output", str(folder / "tune.json")]
    for variant in VARIANTS:
        tune += ["--variant", variant]
    codes = {"tune": run("tune", tune, folder)}
    codes["compare"] = run("compare", [python, "compare_agent.py", COMPARE_AGENT,
                                       "--trials-per-configuration", str(compare_trials),
                                       "--parallel-matchups", str(args.parallel_matchups)], folder)
    print(f"Done: {codes}. Results in {folder} (comparison JSONs in results/comparisons/)", flush=True)
    sys.exit(0 if not any(codes.values()) else 1)


if __name__ == "__main__":
    main()
