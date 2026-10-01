#!/usr/bin/env python3
"""Run repeatable llama-swap benchmark suites in SWAP or DIRECT mode."""

import argparse
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

from config_env import EnvConfigError, load_environment


HERE = Path(__file__).resolve().parent
BENCH = HERE / "benchmark_llama_swap.py"
CASES = HERE / "benchmark_cases.json"
RESULTS = HERE / "benchmark_results.jsonl"
SUMMARY = HERE / "benchmark_summary.csv"
LOG_DIR = HERE / "benchmark-logs"

SHORT_CASES = ("quick_chat", "it_troubleshooting", "coding_bug", "rag_grounding")
LONG_CASES = ("long_context_8k", "long_context_24k")
MULTIMODAL_CASES = (
    "synthetic_image_grid_vqa",
    "synthetic_image_sequence_tracking",
    "synthetic_audio_tone_count",
    "real_image_vqa",
    "real_document_qa",
    "real_audio_transcribe",
)
VL_CASES = SHORT_CASES + MULTIMODAL_CASES


class BenchmarkConfigError(ValueError):
    """Raised when a sweep setting cannot be applied safely."""


class SweepError(RuntimeError):
    def __init__(self, message: str, exit_code: int = 1):
        super().__init__(message)
        self.exit_code = exit_code


@dataclass(frozen=True)
class Suite:
    model: str
    runs: int
    name: str
    cases: tuple[str, ...]


def env_value(env: Mapping[str, str], name: str, default: str) -> str:
    return env.get(name) or default


def is_true(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on", "enable", "enabled"}


def is_false(value: str) -> bool:
    return value.lower() in {"0", "false", "no", "off", "disable", "disabled"}


def validate_bool(name: str, value: str) -> None:
    if not is_true(value) and not is_false(value):
        raise BenchmarkConfigError(
            f"{name} must be one of: 1/0, true/false, yes/no, on/off; got '{value}'"
        )


def configured_models(env: Mapping[str, str], name: str) -> tuple[str, ...]:
    raw = env.get(name, "")
    if not raw:
        return ()
    return tuple(part for part in re.split(r"[\s,]+", raw.strip()) if part)


def selected_model_names(env: Mapping[str, str]) -> tuple[str, ...]:
    explicit = configured_models(env, "MODELS")
    if explicit:
        return explicit

    explicit_model = configured_models(env, "MODEL")
    if explicit_model:
        return explicit_model

    models = configured_models(env, "CORE_MODELS") + configured_models(env, "VL_MODELS")
    if models:
        return models

    return configured_models(env, "BENCHMARK_MODELS")


def model_defaults(env: Mapping[str, str]) -> dict[str, dict[str, str]]:
    raw = env.get("MODEL_DEFAULTS_JSON", "")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise BenchmarkConfigError(f"MODEL_DEFAULTS_JSON is not valid JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise BenchmarkConfigError("MODEL_DEFAULTS_JSON must be a JSON object")
    defaults: dict[str, dict[str, str]] = {}
    for model, settings in parsed.items():
        if not isinstance(model, str) or not isinstance(settings, dict):
            raise BenchmarkConfigError("MODEL_DEFAULTS_JSON must map model IDs to setting objects")
        defaults[model] = {str(name): str(value) for name, value in settings.items()}
    return defaults


def resolve_model_defaults(model: str, env: Mapping[str, str]) -> dict[str, str]:
    defaults = {
        "CTX_SIZE": "32768",
        "PARALLEL": "1",
        "THREADS": "6",
        "THREADS_BATCH": "6",
        "GPU_LAYERS": "0",
        "BATCH_SIZE": "256",
        "UBATCH_SIZE": "64",
        "CACHE_TYPE_K": "",
        "CACHE_TYPE_V": "",
        "FLASH_ATTN": "",
        "REASONING": "",
        "REASONING_EFFORT": "",
        "FORCE_CPU": "0",
        "JINJA": "1",
        "MODEL_PATH": "",
        "MMPROJ": "",
    }
    configured_defaults = model_defaults(env).get(model)
    if configured_defaults is None and not env.get("MODEL_PATH"):
        raise BenchmarkConfigError(
            f"Unknown MODEL '{model}'. Set MODEL_PATH explicitly for custom models."
        )
    if configured_defaults:
        defaults.update(configured_defaults)
    for name in tuple(defaults):
        if name in env and env[name]:
            defaults[name] = env[name]
    if not defaults["MODEL_PATH"]:
        raise BenchmarkConfigError(f"No model path resolved for {model}")
    return defaults


def server_supports(server_help: str, flag: str) -> bool:
    # Match an option token, not a prefix such as --lazy inside --lazy-load.
    return re.search(rf"(?<!\S){re.escape(flag)}(?=\s|=|$)", server_help) is not None


def first_supported_flag(server_help: str, *flags: str) -> str | None:
    return next((flag for flag in flags if server_supports(server_help, flag)), None)


def build_direct_server_args(
    model: str,
    env: Mapping[str, str],
    server_help: str,
    port: str = "18080",
) -> list[str]:
    defaults = resolve_model_defaults(model, env)
    args = [
        "--model",
        defaults["MODEL_PATH"],
        "--host",
        env_value(env, "DIRECT_HOST", "0.0.0.0"),
        "--port",
        port,
        "--ctx-size",
        defaults["CTX_SIZE"],
        "--parallel",
        defaults["PARALLEL"],
        "--threads",
        defaults["THREADS"],
        "--threads-batch",
        defaults["THREADS_BATCH"],
        "--n-gpu-layers",
        defaults["GPU_LAYERS"],
        "--batch-size",
        defaults["BATCH_SIZE"],
        "--ubatch-size",
        defaults["UBATCH_SIZE"],
    ]

    if server_supports(server_help, "--alias"):
        args.extend(("--alias", model))
    if is_true(defaults["JINJA"]):
        if not server_supports(server_help, "--jinja"):
            raise BenchmarkConfigError("JINJA was requested, but installed llama-server does not advertise --jinja")
        args.append("--jinja")

    def add_value(env_name: str, flag: str, value: str) -> None:
        if value:
            if not server_supports(server_help, flag):
                raise BenchmarkConfigError(
                    f"{env_name} was requested, but installed llama-server does not advertise {flag}"
                )
            args.extend((flag, value))

    def add_optional_alias(env_name: str, value: str, *flags: str) -> None:
        if not value:
            return
        selected = first_supported_flag(server_help, *flags)
        if selected is None:
            raise BenchmarkConfigError(f"{env_name} was requested, but no supported flag was found: {', '.join(flags)}")
        args.extend((selected, value))

    add_value("MMPROJ", "--mmproj", defaults["MMPROJ"])
    add_value("DEVICE", "--device", env.get("DEVICE", ""))

    explicit_device_use = any(
        env.get(name) and is_true(env[name]) for name in ("KV_OFFLOAD", "OP_OFFLOAD")
    )
    if defaults["GPU_LAYERS"] == "0" and not env.get("DEVICE") and not explicit_device_use:
        if server_supports(server_help, "--device"):
            args.extend(("--device", "none"))
        if not env.get("OP_OFFLOAD") and server_supports(server_help, "--no-op-offload"):
            args.append("--no-op-offload")
    elif (
        is_true(defaults["FORCE_CPU"])
        and "GPU_LAYERS" not in env
        and not env.get("DEVICE")
        and not explicit_device_use
    ):
        if server_supports(server_help, "--device"):
            args.extend(("--device", "none"))
        if not env.get("OP_OFFLOAD") and server_supports(server_help, "--no-op-offload"):
            args.append("--no-op-offload")

    if env.get("CPU_MOE"):
        validate_bool("CPU_MOE", env["CPU_MOE"])
        if is_true(env["CPU_MOE"]):
            if not server_supports(server_help, "--cpu-moe"):
                raise BenchmarkConfigError("CPU_MOE was requested, but installed llama-server does not advertise --cpu-moe")
            args.append("--cpu-moe")
    add_optional_alias("CPU_MOE_LAYERS", env.get("CPU_MOE_LAYERS", ""), "--n-cpu-moe", "--cpu-moe-layers")
    add_optional_alias("CPU_FFN", env.get("CPU_FFN", ""), "--n-cpu-ffn", "--cpu-ffn")

    add_value("CACHE_TYPE_K", "--cache-type-k", defaults["CACHE_TYPE_K"])
    add_value("CACHE_TYPE_V", "--cache-type-v", defaults["CACHE_TYPE_V"])
    for name, flag in (("KV_OFFLOAD", "--no-kv-offload"), ("OP_OFFLOAD", "--no-op-offload"), ("MMAP", "--no-mmap")):
        if env.get(name):
            validate_bool(name, env[name])
            if is_false(env[name]):
                if not server_supports(server_help, flag):
                    raise BenchmarkConfigError(f"{name} was requested, but installed llama-server does not advertise {flag}")
                args.append(flag)
    if env.get("MLOCK"):
        validate_bool("MLOCK", env["MLOCK"])
        if is_true(env["MLOCK"]):
            if not server_supports(server_help, "--mlock"):
                raise BenchmarkConfigError("MLOCK was requested, but installed llama-server does not advertise --mlock")
            args.append("--mlock")

    add_value("FLASH_ATTN", "--flash-attn", defaults["FLASH_ATTN"])
    add_optional_alias("LAZY_MODE", env.get("LAZY_MODE", ""), "--lazy", "--lazy-load", "--load-lazy")

    if env.get("REPACK"):
        validate_bool("REPACK", env["REPACK"])
        repack_flag = "--repack" if is_true(env["REPACK"]) else "--no-repack"
        if not server_supports(server_help, repack_flag):
            raise BenchmarkConfigError(f"REPACK was requested, but installed llama-server does not advertise {repack_flag}")
        args.append(repack_flag)

    add_value("FIT", "--fit", env.get("FIT", ""))
    add_optional_alias("FIT_TARGET", env.get("FIT_TARGET", ""), "--fit-target", "--fit-target-mib")

    spec_type = env.get("SPEC_TYPE", "")
    spec_draft_n_max = env.get("SPEC_DRAFT_N_MAX", "")
    if env.get("MTP_N"):
        mtp_n = env["MTP_N"]
        if not mtp_n.isascii() or not mtp_n.isdigit():
            raise BenchmarkConfigError(f"MTP_N must be a non-negative integer; got '{mtp_n}'")
        if int(mtp_n) > 0:
            if spec_type and spec_type != "draft-mtp":
                raise BenchmarkConfigError(f"MTP_N>0 conflicts with SPEC_TYPE={spec_type}")
            spec_type = "draft-mtp"
            spec_draft_n_max = spec_draft_n_max or mtp_n
    add_value("SPEC_TYPE", "--spec-type", spec_type)
    add_value("SPEC_DRAFT_N_MAX", "--spec-draft-n-max", spec_draft_n_max)
    add_value("SPEC_DRAFT_N_MIN", "--spec-draft-n-min", env.get("SPEC_DRAFT_N_MIN", ""))
    add_optional_alias("DRAFT_MODEL", env.get("DRAFT_MODEL", ""), "--spec-draft-model", "--model-draft", "--draft-model")

    for name, flag, value in (
        ("REASONING", "--reasoning", defaults["REASONING"]),
        ("REASONING_EFFORT", "--reasoning-effort", defaults["REASONING_EFFORT"]),
        ("REASONING_BUDGET", "--reasoning-budget", env.get("REASONING_BUDGET", "")),
        ("CACHE_RAM", "--cache-ram", env.get("CACHE_RAM", "")),
        ("CPU_STRICT", "--cpu-strict", env.get("CPU_STRICT", "")),
        ("CPU_STRICT_BATCH", "--cpu-strict-batch", env.get("CPU_STRICT_BATCH", "")),
        ("POLL", "--poll", env.get("POLL", "")),
        ("POLL_BATCH", "--poll-batch", env.get("POLL_BATCH", "")),
    ):
        add_value(name, flag, value)

    if env.get("DIRECT_EXTRA_ARGS"):
        args.extend(env["DIRECT_EXTRA_ARGS"].split())
    return args


def plan_suites(env: Mapping[str, str]) -> list[Suite]:
    short_runs = int(env_value(env, "SHORT_RUNS", "3"))
    long_runs = int(env_value(env, "LONG_RUNS", "2"))
    selected_model = env.get("MODEL", "")
    custom_cases_value = env.get("BENCH_CASES", "")
    core_models = configured_models(env, "CORE_MODELS")
    vl_models = configured_models(env, "VL_MODELS")
    unified_models = configured_models(env, "MODELS")

    if unified_models:
        if custom_cases_value:
            cases = tuple(custom_cases_value.replace(",", " ").split())
            return [Suite(model, short_runs, "custom", cases) for model in unified_models]
        suites = [
            suite
            for model in unified_models
            for suite in (
                Suite(model, short_runs, "short-core", SHORT_CASES),
                Suite(model, long_runs, "long-context", LONG_CASES),
            )
        ]
        return suites

    if custom_cases_value:
        cases = tuple(custom_cases_value.replace(",", " ").split())
        models = (selected_model,) if selected_model else core_models + vl_models
        if not models:
            raise BenchmarkConfigError("Set MODEL, MODELS, CORE_MODELS, VL_MODELS, or BENCHMARK_MODELS")
        return [Suite(model, short_runs, "custom", cases) for model in models]

    if selected_model:
        if selected_model in vl_models:
            return [Suite(selected_model, short_runs, "vl-text-subset", VL_CASES)]
        return [
            Suite(selected_model, short_runs, "short-core", SHORT_CASES),
            Suite(selected_model, long_runs, "long-context", LONG_CASES),
        ]

    suites = [
        suite
        for model in core_models
        for suite in (
            Suite(model, short_runs, "short-core", SHORT_CASES),
            Suite(model, long_runs, "long-context", LONG_CASES),
        )
    ]
    suites.extend(Suite(model, short_runs, "vl-text-subset", VL_CASES) for model in vl_models)
    if not suites:
        raise BenchmarkConfigError("Set MODEL, MODELS, CORE_MODELS, VL_MODELS, or BENCHMARK_MODELS")
    return suites


def build_benchmark_command(
    suite: Suite,
    *,
    benchmark: Path,
    cases_file: Path,
    results: Path,
    summary: Path,
    base_url: str,
    config_id: str,
    warmups: int,
    timeout: int,
    started_at: str,
    direct: bool,
    direct_args: str = "",
) -> list[str]:
    note = f"Automated sweep {suite.name}; started {started_at}"
    if direct:
        note += f"; direct={direct_args}"
    command = [
        sys.executable,
        str(benchmark),
        "--base-url",
        base_url,
        "--cases-file",
        str(cases_file),
        "--results",
        str(results),
        "--summary",
        str(summary),
        "--model",
        suite.model,
        "--config-id",
        config_id,
        "--runs",
        str(suite.runs),
        "--warmups",
        str(warmups),
        "--timeout",
        str(timeout),
        "--notes",
        note,
    ]
    if direct:
        command.extend(("--skip-model-check", "--api-key", ""))
    for case in suite.cases:
        command.extend(("--case", case))
    return command


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class RunLog:
    def __init__(self, path: Path):
        self.path = path

    def write(self, message: str = "") -> None:
        print(message, flush=True)
        with self.path.open("a", encoding="utf-8") as log_file:
            log_file.write(message + "\n")


class DirectServer:
    def __init__(self, env: Mapping[str, str], log: RunLog, server_log: Path):
        self.env = env
        self.log = log
        self.server_log = server_log
        self.container = env.get("DIRECT_CONTAINER", "")
        if not self.container:
            raise SweepError("DIRECT_MODE requires DIRECT_CONTAINER")
        self.port = env_value(env, "DIRECT_PORT", "18080")
        self.pidfile = f"/tmp/benchmark-llama-server-{self.port}.pid"
        self.process: subprocess.Popen | None = None
        self.started = False
        self.base_url = ""
        self.server_args: list[str] = []
        self.resolved_args = ""

    def command(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        result = subprocess.run(
            ["docker", *args], capture_output=True, text=True, check=False
        )
        if check and result.returncode:
            detail = result.stderr.strip() or result.stdout.strip()
            raise SweepError(detail or f"docker {' '.join(args)} failed")
        return result

    def load_capabilities(self) -> str:
        if shutil.which("docker") is None:
            raise SweepError("Required command not found: docker")
        inspected = self.command("inspect", self.container, check=False)
        if inspected.returncode:
            raise SweepError(f"DIRECT_CONTAINER '{self.container}' is not running/available")

        restart = env_value(self.env, "DIRECT_RESTART_CONTAINER", "1")
        validate_bool("DIRECT_RESTART_CONTAINER", restart)
        if is_true(restart):
            self.log.write(f"Restarting {self.container} to clear resident model state before DIRECT test...")
            self.command("restart", self.container)

        version = self.command("exec", self.container, "/app/llama-server", "--version", check=False)
        help_result = self.command("exec", self.container, "/app/llama-server", "--help", check=False)
        if help_result.returncode:
            raise SweepError(f"Could not run /app/llama-server --help in {self.container}")
        self.log.write("llama-server version:")
        for line in version.stdout.splitlines()[:5]:
            if line:
                self.log.write(f"  {line}")
        return help_result.stdout + help_result.stderr

    def container_ip(self) -> str:
        result = self.command("inspect", self.container)
        try:
            data = json.loads(result.stdout)[0]
            networks = data.get("NetworkSettings", {}).get("Networks", {}) or {}
            for config in networks.values():
                address = (config or {}).get("IPAddress")
                if address:
                    return str(address)
        except (IndexError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise SweepError(f"Could not parse Docker network information: {error}") from error
        raise SweepError(f"Could not determine an address for {self.container}")

    def check_container_file(self, path: str, label: str) -> None:
        result = self.command("exec", self.container, "test", "-f", path, check=False)
        if result.returncode:
            raise SweepError(f"{label} does not exist inside {self.container}: {path}")

    def start(self, model: str) -> bool:
        server_help = self.load_capabilities()
        try:
            self.server_args = build_direct_server_args(model, self.env, server_help, self.port)
        except BenchmarkConfigError as error:
            raise SweepError(str(error)) from error

        self.check_container_file(self.server_args[1], "Model file")
        if "--mmproj" in self.server_args:
            self.check_container_file(self.server_args[self.server_args.index("--mmproj") + 1], "MMPROJ")

        self.base_url = f"http://{self.container_ip()}:{self.port}"
        self.resolved_args = shlex.join(["/app/llama-server", *self.server_args])

        stale_cleanup = (
            'pidfile="$1"; '
            'if [ -f "$pidfile" ]; then '
            'pid="$(cat "$pidfile" 2>/dev/null || true)"; '
            '[ -z "$pid" ] || kill -TERM "$pid" 2>/dev/null || true; '
            'rm -f "$pidfile"; fi'
        )
        self.command("exec", self.container, "sh", "-c", stale_cleanup, "sh", self.pidfile, check=False)

        self.log.write("DIRECT server command:")
        self.log.write(f"  {self.resolved_args}")
        self.log.write(f"DIRECT server log: {self.server_log}")
        self.log.write(f"DIRECT endpoint:   {self.base_url}")
        self.server_log.parent.mkdir(parents=True, exist_ok=True)
        with self.server_log.open("w", encoding="utf-8") as server_output:
            self.process = subprocess.Popen(
                [
                    "docker",
                    "exec",
                    self.container,
                    "sh",
                    "-c",
                    'pidfile="$1"; shift; echo $$ > "$pidfile"; exec "$@"',
                    "sh",
                    self.pidfile,
                    "/app/llama-server",
                    *self.server_args,
                ],
                stdout=server_output,
                stderr=subprocess.STDOUT,
            )
        self.started = True

        start_timeout = int(env_value(self.env, "DIRECT_START_TIMEOUT", "300"))
        deadline = time.monotonic() + start_timeout
        while time.monotonic() < deadline:
            try:
                request = urllib.request.Request(f"{self.base_url}/v1/models")
                with urllib.request.urlopen(request, timeout=2):
                    self.log.write("DIRECT llama-server is ready.")
                    return True
            except (urllib.error.URLError, TimeoutError, OSError):
                pass
            if self.process.poll() is not None:
                self.log.write("DIRECT llama-server exited before becoming ready.")
                self.log_server_tail()
                return False
            time.sleep(1)

        self.log.write(f"DIRECT llama-server did not become ready within {start_timeout}s.")
        self.log_server_tail()
        return False

    def log_server_tail(self) -> None:
        try:
            lines = self.server_log.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
        except OSError:
            lines = []
        self.log.write("Last 80 server-log lines:")
        for line in lines:
            self.log.write(line)

    def cleanup(self) -> None:
        if not self.started:
            return
        if self.pidfile:
            cleanup_command = (
                'pidfile="$1"; '
                'if [ -f "$pidfile" ]; then '
                'pid="$(cat "$pidfile" 2>/dev/null || true)"; '
                'if [ -n "$pid" ]; then '
                'kill -TERM "$pid" 2>/dev/null || true; i=0; '
                'while [ "$i" -lt 20 ] && kill -0 "$pid" 2>/dev/null; do '
                'sleep 0.25; i=$((i+1)); done; '
                'kill -KILL "$pid" 2>/dev/null || true; fi; '
                'rm -f "$pidfile"; fi'
            )
            self.command("exec", self.container, "sh", "-c", cleanup_command, "sh", self.pidfile, check=False)
        if self.process is not None:
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        self.started = False


def discover_models(base_url: str, api_key: str) -> list[str]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(f"{base_url.rstrip('/')}/v1/models", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
        raise SweepError(f"Could not read {base_url.rstrip('/')}/v1/models: {error}") from error
    models = []
    for item in payload.get("data", []):
        if isinstance(item, dict) and item.get("id"):
            models.append(str(item["id"]))
    return models


def is_local_base_url(base_url: str) -> bool:
    try:
        hostname = urllib.parse.urlparse(base_url).hostname or ""
    except ValueError:
        return False
    return hostname in {"localhost", "127.0.0.1", "0.0.0.0", "::1"} or hostname.startswith("localhost.")


def start_local_mock_server(env: Mapping[str, str], log: RunLog) -> tuple[str, set[str], subprocess.Popen]:
    raw_base = env.get("LLAMA_SWAP_BASE_URL", "http://localhost:8080").strip()
    if not raw_base:
        raw_base = "http://localhost:8080"
    if not is_local_base_url(raw_base):
        raise SweepError(f"Local mock server requires a loopback base URL, got '{raw_base}'")

    parsed_url = urllib.parse.urlparse(raw_base)
    host = parsed_url.hostname or "127.0.0.1"
    port = parsed_url.port or 8080
    models = tuple(selected_model_names(env)) or ("mock-model",)
    server_path = HERE / "mock_llama_swap_server.py"
    if not server_path.is_file():
        raise SweepError(f"Missing mock server helper: {server_path}")

    log.write(f"SWAP endpoint is unavailable at {raw_base}; starting local mock llama-swap server...")
    command = [sys.executable, str(server_path), "--host", host, "--port", str(port)]
    for model in models:
        command.extend(["--model", model])
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        text=True,
    )

    deadline = time.monotonic() + 10
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            available = set(discover_models(raw_base, env.get("LLAMA_SWAP_API_KEY", "")))
            return raw_base, available, process
        except SweepError as error:
            last_error = error
            time.sleep(0.2)
    if process.poll() is not None:
        raise SweepError(f"Mock llama-swap server exited before becoming ready: {process.returncode}")
    raise SweepError(f"Mock llama-swap server did not become ready at {raw_base}: {last_error}")


def stream_process(command: Sequence[str], env: Mapping[str, str], log: RunLog) -> int:
    process = subprocess.Popen(
        list(command),
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        with log.path.open("a", encoding="utf-8") as output:
            output.write(line)
    return process.wait()


def run_sweep(env: Mapping[str, str]) -> int:
    config_id = env_value(env, "CONFIG_ID", "baseline")
    short_runs = int(env_value(env, "SHORT_RUNS", "3"))
    long_runs = int(env_value(env, "LONG_RUNS", "2"))
    warmups = int(env_value(env, "WARMUPS", "1"))
    timeout = int(env_value(env, "TIMEOUT", "1800"))
    direct = is_true(env_value(env, "DIRECT_MODE", "0"))
    started_at = now_iso()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = LOG_DIR / f"benchmark-{config_id}-{timestamp}.log"
    server_log = LOG_DIR / f"direct-server-{config_id}-{timestamp}.log"
    log = RunLog(log_path)

    if shutil.which("python3") is None:
        raise SweepError("Required command not found: python3")
    if not BENCH.is_file():
        raise SweepError(f"Missing {BENCH}")
    if not CASES.is_file():
        raise SweepError(f"Missing {CASES}")

    if not direct and not env.get("LLAMA_SWAP_API_KEY"):
        log.write("WARNING: LLAMA_SWAP_API_KEY is not set.")
        log.write("If llama-swap authentication is enabled, discovery and benchmarks will fail.")

    log.write("============================================================")
    log.write("llama-swap benchmark sweep")
    log.write(f"Started:       {started_at}")
    log.write(f"Mode:          {'DIRECT' if direct else 'SWAP'}")
    log.write(f"Config ID:     {config_id}")
    log.write(f"Short runs:    {short_runs}")
    log.write(f"Long runs:     {long_runs}")
    log.write(f"Warmups:       {warmups}")
    log.write(f"Timeout:       {timeout}s")
    log.write(f"Results:       {RESULTS}")
    log.write(f"Summary:       {SUMMARY}")
    log.write(f"Log:           {log_path}")
    log.write("============================================================")

    base_url = env.get("LLAMA_SWAP_BASE_URL", "")
    direct_server: DirectServer | None = None
    local_mock_server: subprocess.Popen | None = None
    completed: list[str] = []
    skipped: list[str] = []
    failures: list[str] = []

    try:
        if direct:
            model = env.get("MODEL", "")
            if not model:
                raise SweepError("DIRECT_MODE requires MODEL=<model-id>")
            direct_server = DirectServer(env, log, server_log)
            if not direct_server.start(model):
                failures.append(f"{model}:direct-server-start")
                log.write(f"FAIL  {model}: direct server did not start")
                return 2
            available_models = {model}
            base_url = direct_server.base_url
        else:
            if not base_url:
                raise SweepError("SWAP mode requires LLAMA_SWAP_BASE_URL")
            try:
                available_models = set(discover_models(base_url, env.get("LLAMA_SWAP_API_KEY", "")))
            except SweepError:
                if is_local_base_url(base_url):
                    base_url, available_models, local_mock_server = start_local_mock_server(env, log)
                else:
                    raise
            log.write("")
            log.write("Models advertised by llama-swap:")
            for model in sorted(available_models):
                log.write(f"  - {model}")
            log.write("")

        for suite in plan_suites(env):
            label = f"{suite.model} ({suite.name})"
            if suite.model not in available_models:
                log.write(f"SKIP  {label}: model is not available in this benchmark mode")
                skipped.append(f"{suite.model}:{suite.name}")
                continue

            direct_args = direct_server.resolved_args if direct_server else ""
            command = build_benchmark_command(
                suite,
                benchmark=BENCH,
                cases_file=CASES,
                results=RESULTS,
                summary=SUMMARY,
                base_url=base_url,
                config_id=config_id,
                warmups=warmups,
                timeout=timeout,
                started_at=started_at,
                direct=direct,
                direct_args=direct_args,
            )
            log.write("------------------------------------------------------------")
            log.write(f"RUN   {suite.model}")
            log.write(f"Suite: {suite.name}")
            log.write(f"Cases: {' '.join(suite.cases)}")
            log.write(f"Runs:  {suite.runs} measured + {warmups} warmup(s)")
            log.write("------------------------------------------------------------")

            result = stream_process(command, env, log)
            if result == 0:
                completed.append(f"{suite.model}:{suite.name}")
                log.write(f"PASS  {label}")
            else:
                failures.append(f"{suite.model}:{suite.name}:rc={result}")
                log.write(f"FAIL  {label}, exit code {result}")
            log.write("")
    finally:
        if direct_server is not None:
            direct_server.cleanup()
        if local_mock_server is not None:
            local_mock_server.terminate()
            try:
                local_mock_server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                local_mock_server.kill()
                local_mock_server.wait(timeout=5)

    log.write("============================================================")
    log.write(f"Sweep complete: {now_iso()}")
    log.write("")
    if completed:
        log.write("Completed suites:")
        for item in completed:
            log.write(f"  + {item}")
        log.write("")
    if skipped:
        log.write("Skipped suites:")
        for item in skipped:
            log.write(f"  - {item}")
        log.write("")
    if failures:
        log.write("Failed suites:")
        for item in failures:
            log.write(f"  ! {item}")
        log.write("")
    log.write(f"Raw results: {RESULTS}")
    log.write(f"Summary:     {SUMMARY}")
    log.write(f"Run log:     {log_path}")
    if direct:
        log.write(f"Server log:  {server_log}")
    if SUMMARY.is_file():
        log.write("")
        log.write("Current summary:")
        try:
            for line in SUMMARY.read_text(encoding="utf-8").splitlines():
                log.write(line)
        except OSError:
            pass
    return 2 if failures else 0


def make_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Benchmark configured llama-swap models or a direct llama-server process."
    )


def main() -> int:
    make_parser().parse_args()
    try:
        env = load_environment()
    except EnvConfigError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    old_term_handler = signal.getsignal(signal.SIGTERM)

    def stop_on_term(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop_on_term)
    try:
        return run_sweep(env)
    except BenchmarkConfigError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    except SweepError as error:
        log_dir = LOG_DIR
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            latest = max(log_dir.glob("benchmark-*.log"), key=lambda item: item.stat().st_mtime)
            RunLog(latest).write(f"ERROR: {error}")
        except (OSError, ValueError):
            print(f"ERROR: {error}", file=sys.stderr)
        return error.exit_code
    except KeyboardInterrupt:
        return 130
    finally:
        signal.signal(signal.SIGTERM, old_term_handler)


if __name__ == "__main__":
    raise SystemExit(main())