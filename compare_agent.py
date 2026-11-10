"""Compare an agent against every other built-in policy in concurrent processes."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import re
import subprocess
import sys
from threading import Event, Lock

OPPONENTS = (
    "RandomAgent", "BasicAgent", "BasicAgentPlus", "ExperiencedAgent",
    "ExperiencedAgent2", "ExperiencedAgent3", "ExperiencedAgent3Tuned",
    "ExperiencedAgent4", "SwitchPAgent", "HumanRulesAgent3",
)
ROOT = Path(__file__).resolve().parent
STOPPING = Event()
ACTIVE_PROCESSES: set[subprocess.Popen] = set()
PROCESS_LOCK = Lock()


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer") from None
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def stop_active_processes() -> None:
    """Terminate every live matchup process after an interrupt."""
    STOPPING.set()
    with PROCESS_LOCK:
        processes = tuple(ACTIVE_PROCESSES)
    for process in processes:
        try:
            if process.poll() is None:
                process.terminate()
        except OSError:
            # It may have ended between poll() and terminate().
            pass


def run_opponent(
    agent: str, opponent: str, trials: int, seed: int, workers: int, progress,
) -> tuple[int, str]:
    if STOPPING.is_set():
        return 130, ""
    filename = f"{agent.replace(':', '_')}_vs_{opponent.replace(':', '_')}_seed{seed}.json"
    command = [
        sys.executable, str(ROOT / "arena_multithread.py"), agent, opponent,
        "--trials-per-configuration", str(trials), "--seed", str(seed),
        "--workers", str(workers), "--output", str(ROOT / "results/comparisons" / filename),
        "--progress-stream",
    ]
    # A separate process group prevents Ctrl+C from racing the supervisor's
    # coordinated shutdown. Only the parent may write to the terminal.
    popen_options = {"cwd": ROOT, "stdout": subprocess.PIPE,
                     "stderr": subprocess.STDOUT, "text": True,
                     "errors": "replace", "bufsize": 1}
    if os.name == "nt":
        popen_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_options["start_new_session"] = True
    process = subprocess.Popen(command, **popen_options)
    with PROCESS_LOCK:
        ACTIVE_PROCESSES.add(process)
        stopping = STOPPING.is_set()
    try:
        if stopping:
            process.terminate()
        progress.reset()
        progress.set_postfix_str("running")
        output = []
        for line in process.stdout:
            match = re.fullmatch(r"@@ARENA_PROGRESS (\d+)\n?", line)
            if match:
                count = int(match.group(1))
                progress.update(max(0, count - progress.n))
            else:
                output.append(line)
        code = process.wait()
        progress.set_postfix_str("done" if code == 0 else "failed")
        return code, "".join(output)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
        process.stdout.close()
        with PROCESS_LOCK:
            ACTIVE_PROCESSES.discard(process)


def main(argv=None) -> int:
    STOPPING.clear()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("agent", help="class name or package.module:ClassName")
    parser.add_argument("--trials-per-configuration", type=positive_int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=positive_int, default=1,
                        help="threads per matchup (default: 1)")
    parser.add_argument("--parallel-matchups", type=positive_int, default=2,
                        help="concurrent opponent processes (default: 2)")
    args = parser.parse_args(argv)
    trials, seed = args.trials_per_configuration, args.seed
    workers, parallel = args.workers, args.parallel_matchups
    try:
        agent = args.agent
        if ":" not in agent:
            agent = "src.agent:" + agent
        if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", agent):
            raise ValueError("Agent must be a class name or package.module:ClassName")
        from arena import client_factory
        client_factory(agent)
    except (ValueError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    opponents = [f"src.agent:{name}" for name in OPPONENTS
                 if f"src.agent:{name}" != agent]
    print(f"Comparing {agent}; concurrent matchups: {min(parallel, len(opponents))}",
          flush=True)
    failed = False
    from tqdm import tqdm
    # Allocate rows together and share tqdm's in-process rendering lock.
    # Disable cursor movement when output is redirected to a log or IDE pipe.
    bars = [tqdm(total=6 * trials, desc=opponent.rsplit(":", 1)[-1],
                 position=position, unit="game", dynamic_ncols=True, leave=True,
                 disable=not sys.stderr.isatty(), postfix="queued")
            for position, opponent in enumerate(opponents)]
    reports = {}
    interrupted = False
    executor = ThreadPoolExecutor(max_workers=min(parallel, len(opponents)))
    jobs = {}
    try:
        jobs = {
            executor.submit(run_opponent, agent, opponent, trials, seed, workers, bar): opponent
            for opponent, bar in zip(opponents, bars)
        }
        for job in as_completed(jobs):
            opponent = jobs[job]
            try:
                code, output = job.result()
                reports[opponent] = (code, output)
                if code:
                    failed = True
            except Exception as error:
                reports[opponent] = (None, f"Could not complete matchup: {error}\n")
                failed = True
    except KeyboardInterrupt:
        stop_active_processes()
        for job in jobs:
            job.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        interrupted = True
    else:
        executor.shutdown(wait=True)
    finally:
        for bar in reversed(bars):
            bar.close()
        if sys.stderr.isatty():
            print("\n" * len(bars), end="", file=sys.stderr)
    if interrupted:
        print("Interrupted: stopped all active matchup processes.", file=sys.stderr)
        return 130
    # Reports cannot shift active progress rows: print only after all bars close.
    print("\nALL MATCHUPS FINISHED — COMPARISON RESULTS")
    print(f"Evaluated agent: {agent}")
    print(f"Seed: {seed} | Trials per seating: {trials:,} | Games per matchup: {6 * trials:,}")
    print("Lower mean placement is better.")
    for index, opponent in enumerate(opponents, 1):
        code, report = reports[opponent]
        status = "SUCCESS" if code == 0 else (
            "FAILED (launch/runtime error)" if code is None else f"FAILED (exit {code})"
        )
        print("\n" + "=" * 80)
        print(f"COMPARISON {index}/{len(opponents)} | {status}")
        print(f"Evaluated: {agent}")
        print(f"Opponent:  {opponent}")
        print("-" * 80)
        print(report.rstrip() or "(No report produced.)")
        print("=" * 80)
    if failed:
        print("\nOne or more comparisons failed; see the labeled sections above.")
        return 1
    print("All comparisons finished. JSON reports are in results/comparisons/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
