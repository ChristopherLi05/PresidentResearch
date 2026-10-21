import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import compare_agent


class CompareAgentTests(unittest.TestCase):
    def test_child_progress_is_consumed_without_leaking_terminal_output(self):
        class Bar:
            n = 0

            def reset(self):
                self.n = 0

            def set_postfix_str(self, value):
                self.status = value

            def update(self, amount):
                self.n += amount

        with tempfile.TemporaryDirectory(prefix="compare-progress-test-") as directory:
            root = Path(directory)
            (root / "arena_multithread.py").write_text(
                "print('@@ARENA_PROGRESS 1', flush=True)\n"
                "print('@@ARENA_PROGRESS 6', flush=True)\n"
                "print('Final report')\n", encoding="utf-8"
            )
            bar = Bar()
            with patch("compare_agent.ROOT", root):
                code, output = compare_agent.run_opponent(
                    "src.agent:BasicAgent", "src.agent:RandomAgent", 1, 42, 1, bar
                )
            self.assertEqual(code, 0)
            self.assertEqual(bar.n, 6)
            self.assertEqual(bar.status, "done")
            self.assertEqual(output, "Final report\n")

    def tearDown(self):
        compare_agent.STOPPING.clear()
        with compare_agent.PROCESS_LOCK:
            compare_agent.ACTIVE_PROCESSES.clear()

    def test_interrupt_terminates_every_active_process(self):
        class Process:
            def __init__(self):
                self.terminated = False

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

        first, second = Process(), Process()
        with compare_agent.PROCESS_LOCK:
            compare_agent.ACTIVE_PROCESSES.update((first, second))
        compare_agent.stop_active_processes()
        self.assertTrue(compare_agent.STOPPING.is_set())
        self.assertTrue(first.terminated)
        self.assertTrue(second.terminated)
    def test_bad_limits_fail_before_launch(self):
        for limit in ("0", "-1", "abc", "1.5"):
            with self.subTest(limit=limit), patch(
                "compare_agent.run_opponent"
            ) as run, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    compare_agent.main(["BasicAgent", "--parallel-matchups", limit])
                self.assertEqual(error.exception.code, 2)
                run.assert_not_called()

    def test_launch_failure_is_reported_and_other_jobs_finish(self):
        def run(agent, opponent, *args):
            if opponent.endswith(":RandomAgent"):
                raise OSError("simulated process launch failure")
            return 0, ""

        errors = io.StringIO()
        with patch("compare_agent.run_opponent", side_effect=run) as worker, (
            contextlib.redirect_stdout(errors)
        ), contextlib.redirect_stderr(errors):
            self.assertEqual(compare_agent.main(["BasicAgent"]), 1)
            self.assertEqual(worker.call_count, 6)
        self.assertIn("simulated process launch failure", errors.getvalue())

    def test_abrupt_child_exit_needs_no_completion_file(self):
        with tempfile.TemporaryDirectory(prefix="compare-agent-test-") as directory:
            root = Path(directory)
            (root / "arena_multithread.py").write_text(
                "import os\nos._exit(7)\n", encoding="utf-8"
            )
            errors = io.StringIO()
            with patch("compare_agent.ROOT", root), (
                contextlib.redirect_stdout(errors)
            ), contextlib.redirect_stderr(errors):
                self.assertEqual(compare_agent.main(["BasicAgent"]), 1)
            self.assertEqual(errors.getvalue().count("(exit 7)"), 6)
            self.assertEqual(list(root.glob("*.done")), [])

    def test_reports_wait_for_all_jobs_and_use_opponent_order(self):
        output = io.StringIO()
        completed = []

        def run(agent, opponent, *args):
            self.assertNotIn("COMPARISON RESULTS", output.getvalue())
            self.assertNotIn("REPORT FOR", output.getvalue())
            completed.append(opponent)
            return 0, f"REPORT FOR {opponent}\n"

        with patch("compare_agent.run_opponent", side_effect=run), (
            contextlib.redirect_stdout(output)
        ), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(compare_agent.main(["BasicAgent"]), 0)
        self.assertEqual(len(completed), 6)
        text = output.getvalue()
        opponents = [f"src.agent:{name}" for name in compare_agent.OPPONENTS
                     if name != "BasicAgent"]
        positions = [text.index(f"Opponent:  {opponent}\n") for opponent in opponents]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(text.count("Evaluated: src.agent:BasicAgent"), 6)
        self.assertEqual(text.count("| SUCCESS"), 6)


if __name__ == "__main__":
    unittest.main()
