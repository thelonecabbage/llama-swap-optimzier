import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import run_benchmark_tests as runner


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run_benchmark_tests.py"


class RunBenchmarkTestsCliTests(unittest.TestCase):
    def test_parser_uses_environment_paths_as_defaults(self):
        parser = runner.make_parser(
            {
                "BENCHMARK_SCRIPT": "/configured/benchmark.py",
                "BENCHMARK_WRAPPER_LOG_DIR": "/configured/logs",
            }
        )

        args = parser.parse_args(["plan.conf"])

        self.assertEqual(args.benchmark, "/configured/benchmark.py")
        self.assertEqual(args.log_dir, "/configured/logs")

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.temp_dir.name)
        self.test_plan = self.work_dir / "tests.conf"
        self.log_dir = self.work_dir / "logs"
        self.invocations = self.work_dir / "invocations.jsonl"
        self.fake_benchmark = self.work_dir / "fake_benchmark.py"
        self.fake_benchmark.write_text(
            "#!" + sys.executable + "\n"
            "import json, os, sys\n"
            "with open(os.environ['INVOCATIONS'], 'a', encoding='utf-8') as stream:\n"
            "    stream.write(json.dumps(dict(os.environ)) + '\\n')\n"
            "raise SystemExit(int(os.environ.get('EXIT_CODE', '0')))\n",
            encoding="utf-8",
        )
        self.fake_benchmark.chmod(0o755)

    def tearDown(self):
        self.temp_dir.cleanup()

    def run_cli(self, *args, extra_env=None):
        env = os.environ.copy()
        env["INVOCATIONS"] = str(self.invocations)
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            [sys.executable, str(SCRIPT), *map(str, args)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def read_invocations(self):
        return [json.loads(line) for line in self.invocations.read_text().splitlines()]

    def test_dry_run_redacts_secrets_and_keeps_values_literal(self):
        self.test_plan.write_text(
            "[test literal]\n"
            "API_KEY=never-print-this\n"
            "VALUE=$HOME;$(touch SHOULD_NOT_EXIST)\n",
            encoding="utf-8",
        )

        result = self.run_cli(
            "--dry-run",
            "--benchmark",
            self.fake_benchmark,
            "--log-dir",
            self.log_dir,
            self.test_plan,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("never-print-this", result.stdout)
        self.assertIn("API_KEY=<redacted>", result.stdout)
        self.assertIn("$HOME;$(touch SHOULD_NOT_EXIST)", result.stdout)
        self.assertFalse((self.work_dir / "SHOULD_NOT_EXIST").exists())
        self.assertFalse(self.invocations.exists())

    def test_test_name_becomes_default_config_id_and_values_are_passed_literally(self):
        self.test_plan.write_text(
            "[test literal-name]\n"
            "VALUE=$HOME;$(touch SHOULD_NOT_EXIST)\n",
            encoding="utf-8",
        )

        result = self.run_cli(
            "--benchmark",
            self.fake_benchmark,
            "--log-dir",
            self.log_dir,
            self.test_plan,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        invocation = self.read_invocations()[0]
        self.assertEqual(invocation["CONFIG_ID"], "literal-name")
        self.assertEqual(invocation["VALUE"], "$HOME;$(touch SHOULD_NOT_EXIST)")
        self.assertFalse((self.work_dir / "SHOULD_NOT_EXIST").exists())

    def test_python_benchmark_runs_without_executable_permission(self):
        self.fake_benchmark.chmod(0o600)
        self.test_plan.write_text("[test plain-python]\n", encoding="utf-8")

        result = self.run_cli(
            "--benchmark",
            self.fake_benchmark,
            "--log-dir",
            self.log_dir,
            self.test_plan,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.read_invocations()), 1)

    def test_malformed_assignment_outside_test_block_is_rejected(self):
        self.test_plan.write_text("VALUE=orphan\n", encoding="utf-8")

        result = self.run_cli("--dry-run", self.test_plan)

        self.assertEqual(result.returncode, 2)
        self.assertIn("assignment outside", result.stderr)
        self.assertFalse(self.invocations.exists())

    def test_failure_stops_by_default_but_continue_mode_runs_later_tests(self):
        self.test_plan.write_text(
            "[test first]\nEXIT_CODE=7\n"
            "[test second]\nEXIT_CODE=0\n",
            encoding="utf-8",
        )
        base_args = ("--benchmark", self.fake_benchmark, "--log-dir", self.log_dir)

        stopped = self.run_cli(*base_args, self.test_plan)
        self.assertEqual(stopped.returncode, 1)
        self.assertEqual(len(self.read_invocations()), 1)

        self.invocations.unlink()
        continued = self.run_cli("--continue-on-error", *base_args, self.test_plan)
        self.assertEqual(continued.returncode, 1)
        self.assertEqual(len(self.read_invocations()), 2)


if __name__ == "__main__":
    unittest.main()