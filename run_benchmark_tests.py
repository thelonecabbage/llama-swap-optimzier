#!/usr/bin/env python3
"""Run benchmark suites described by a literal KEY=value test plan."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from config_env import EnvConfigError, load_environment


ASSIGNMENT_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
TEST_HEADER_RE = re.compile(r"^\[test[\t ]+(.+)\]$")
SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


class TestPlanError(ValueError):
    pass


def parse_test_plan(path: Path) -> list[tuple[str, list[tuple[str, str]]]]:
    tests: list[tuple[str, list[tuple[str, str]]]] = []
    current_name: str | None = None
    current_env: list[tuple[str, str]] = []

    def finish_test() -> None:
        nonlocal current_name, current_env
        if current_name is not None:
            tests.append((current_name, current_env))
        current_name = None
        current_env = []

    try:
        with path.open("r", encoding="utf-8") as test_file:
            for line_number, raw_line in enumerate(test_file, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                header = TEST_HEADER_RE.fullmatch(line)
                if header:
                    finish_test()
                    current_name = header.group(1).strip()
                    if not current_name:
                        raise TestPlanError(f"empty test name at {path}:{line_number}")
                    continue

                if current_name is None:
                    raise TestPlanError(
                        f"assignment outside a [test NAME] block at {path}:{line_number}"
                    )

                assignment = ASSIGNMENT_RE.fullmatch(line)
                if assignment is None:
                    raise TestPlanError(
                        f"invalid line at {path}:{line_number}: {raw_line.rstrip()}\n"
                        "Expected KEY=value or [test NAME]"
                    )

                key, value = assignment.groups()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                current_env.append((key, value))
    except OSError as error:
        raise TestPlanError(f"cannot read test file {path}: {error}") from error

    finish_test()
    if not tests:
        raise TestPlanError(f"no [test NAME] blocks found in {path}")
    return tests


def is_secret(key: str) -> bool:
    return any(marker in key for marker in SECRET_MARKERS)


def display_assignment(key: str, value: str) -> str:
    return f"{key}=<redacted>" if is_secret(key) else f"{key}={value}"


def safe_test_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def write_line(stream, text: str) -> None:
    stream.write(text + "\n")
    stream.flush()


def run_and_tee(command: list[str], env: dict[str, str], log_file: Path) -> int:
    with log_file.open("a", encoding="utf-8") as output:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            output.write(line)
            output.flush()
        return process.wait()


def make_parser(environment: dict[str, str] | None = None) -> argparse.ArgumentParser:
    environment = environment or {}
    parser = argparse.ArgumentParser(
        description="Run benchmark_all_models.py for each test-plan block."
    )
    parser.add_argument(
        "--benchmark",
        default=environment.get("BENCHMARK_SCRIPT", "./benchmark_all_models.py"),
        help="Path to benchmark_all_models.py",
    )
    parser.add_argument(
        "--log-dir",
        default=environment.get("BENCHMARK_WRAPPER_LOG_DIR", "./benchmark-wrapper-logs"),
        help="Wrapper log directory",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Run later tests even if one test fails",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the runs without executing them",
    )
    parser.add_argument("test_file", nargs="?", help="Test-plan file")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        environment = load_environment()
    except EnvConfigError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    parser = make_parser(environment)
    args = parser.parse_args(argv)

    if not args.test_file:
        parser.error("TEST_FILE is required")

    benchmark = Path(args.benchmark)
    test_path = Path(args.test_file)
    if not test_path.is_file() or not os.access(test_path, os.R_OK):
        print(f"ERROR: cannot read test file: {test_path}", file=sys.stderr)
        return 2
    if not args.dry_run and not benchmark.is_file():
        print(f"ERROR: benchmark script does not exist: {benchmark}", file=sys.stderr)
        return 2

    try:
        tests = parse_test_plan(test_path)
    except TestPlanError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    log_dir = Path(args.log_dir)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(f"ERROR: cannot create log directory {log_dir}: {error}", file=sys.stderr)
        return 2

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_log = log_dir / f"run-{timestamp}.summary.log"
    with summary_log.open("w", encoding="utf-8") as summary:
        write_line(summary, f"Test plan:  {test_path}")
        write_line(summary, f"Benchmark:  {benchmark}")
        write_line(summary, f"Tests:      {len(tests)}")

    print(f"Test plan:  {test_path}")
    print(f"Benchmark:  {benchmark}")
    print(f"Tests:      {len(tests)}\n")

    passed = 0
    failed = 0
    skipped = 0

    for index, (name, assignments) in enumerate(tests, start=1):
        safe_name = safe_test_name(name)
        test_log = log_dir / f"{timestamp}-{safe_name}.log"
        env_args = list(assignments)
        if not any(key == "CONFIG_ID" for key, _ in env_args):
            env_args.append(("CONFIG_ID", name))

        print("================================================================")
        print(f"[{index}/{len(tests)}] {name}")
        print("================================================================")
        print("Environment:", end="")
        for key, value in env_args:
            print(f" {shlex.quote(display_assignment(key, value))}", end="")
        print()

        if args.dry_run:
            safe_args = [display_assignment(key, value) for key, value in env_args]
            rendered_env = " ".join(shlex.quote(value) for value in safe_args)
            print(
                f"Would run: env {rendered_env} {shlex.quote(sys.executable)} "
                f"{shlex.quote(str(benchmark))}\n"
            )
            skipped += 1
            continue

        started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        with test_log.open("w", encoding="utf-8") as output:
            write_line(output, f"# test: {name}")
            write_line(output, f"# started: {started_at}")
            write_line(output, f"# benchmark: {benchmark}")
            safe_env = " ".join(
                shlex.quote(display_assignment(key, value)) for key, value in env_args
            )
            write_line(output, f"# env: {safe_env}\n")

        child_env = environment.copy()
        child_env.update(env_args)
        start_time = time.time()
        result = run_and_tee([sys.executable, str(benchmark)], child_env, test_log)
        elapsed = int(time.time() - start_time)

        status = "PASS" if result == 0 else f"FAIL({result})"
        if result == 0:
            passed += 1
        else:
            failed += 1

        with summary_log.open("a", encoding="utf-8") as summary:
            write_line(summary, f"{name}\t{status}\t{elapsed}s\t{test_log}")
        print(f"{name}\t{status}\t{elapsed}s\t{test_log}\n")

        if result != 0 and not args.continue_on_error:
            message = "Stopping after failure. Use --continue-on-error to run the remaining tests."
            print(message)
            with summary_log.open("a", encoding="utf-8") as summary:
                write_line(summary, message)
            break

    print("\n================================================================")
    print("Wrapper summary")
    print("================================================================")
    print(f"Passed:  {passed}")
    print(f"Failed:  {failed}")
    if args.dry_run:
        print(f"Dry-run: {skipped}")
    print(f"Summary: {summary_log}")
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())