#!/usr/bin/env python3
"""
Repeatable benchmark harness for a llama-swap / llama.cpp OpenAI-compatible endpoint.

Goals:
- Keep benchmark prompts stable.
- Measure cold-ish first request separately from warmed runs.
- Measure time to first token (TTFT) with streaming.
- Capture prompt/completion token counts when the server returns usage.
- Calculate wall-clock decode throughput.
- Capture llama.cpp timing fields when present.
- Append every run to JSONL so future comparisons are auditable.
- Generate a compact CSV summary using medians.

No third-party Python packages are required.

Examples:
  python3 benchmark_llama_swap.py \
      --model text-model \
      --runs 3

  python3 benchmark_llama_swap.py \
      --model text-model \
      --model code-model \
      --case quick_chat \
      --case it_troubleshooting \
      --runs 3

  python3 benchmark_llama_swap.py \
      --model text-model \
      --case long_context_8k \
      --case long_context_24k \
      --runs 2
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from config_env import EnvConfigError, load_environment


DEFAULT_CASES = Path(__file__).with_name("benchmark_cases.json")
DEFAULT_RESULTS = Path(__file__).with_name("benchmark_results.jsonl")
DEFAULT_SUMMARY = Path(__file__).with_name("benchmark_summary.csv")


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def read_first_matching(path: str, prefix: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith(prefix):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def host_metadata() -> Dict[str, Any]:
    mem_total_kib = None
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_total_kib = int(line.split()[1])
                    break
    except OSError:
        pass

    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cpu_model": read_first_matching("/proc/cpuinfo", "model name"),
        "mem_total_gib": round(mem_total_kib / 1024 / 1024, 2) if mem_total_kib else None,
    }


def auth_headers(api_key: Optional[str]) -> Dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def http_json(
    url: str,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
    timeout: int = 600,
) -> Tuple[int, Dict[str, Any]]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers=auth_headers(api_key),
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {"raw": raw}
        return e.code, body


def list_models(base_url: str, api_key: Optional[str]) -> List[str]:
    code, body = http_json(base_url.rstrip("/") + "/v1/models", api_key=api_key)
    if code != 200:
        raise RuntimeError(f"/v1/models returned HTTP {code}: {json.dumps(body)[:1000]}")
    out = []
    for item in body.get("data", []):
        if isinstance(item, dict) and item.get("id"):
            out.append(str(item["id"]))
    return out


def load_cases(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("benchmark_cases.json must contain a JSON array")
    return data


def make_needle_context(approx_tokens: int, needle: str, seed: int = 17) -> str:
    # Deliberately deterministic. Character count is only an approximation;
    # authoritative prompt-token count comes from server usage when available.
    target_chars = approx_tokens * 4
    lines = []
    i = 0
    inserted = False
    while sum(map(len, lines)) < target_chars:
        i += 1
        value = (i * 7919 + seed * 104729) % 1000003
        if not inserted and sum(map(len, lines)) >= target_chars // 2:
            lines.append(
                f"RECORD {i:05d}: project=ORCHID status=verified access_code={needle} "
                f"checksum={value:06d}. This is the only authoritative access code.\n"
            )
            inserted = True
        else:
            lines.append(
                f"RECORD {i:05d}: project=archive-{i % 97:02d} status=normal "
                f"checksum={value:06d} note=synthetic benchmark filler.\n"
            )
    if not inserted:
        lines.append(f"AUTHORITATIVE access_code={needle}\n")
    return "".join(lines)


def materialize_case(case: Dict[str, Any]) -> Dict[str, Any]:
    case = json.loads(json.dumps(case))  # deep copy without external deps
    gen = case.get("generator")
    if gen:
        gtype = gen.get("type")
        if gtype != "needle_haystack":
            raise ValueError(f"Unsupported generator type: {gtype}")
        needle = str(gen["needle"])
        approx_tokens = int(gen["approx_tokens"])
        context = make_needle_context(approx_tokens, needle, int(gen.get("seed", 17)))
        question = gen.get(
            "question",
            "Using only the records above, return the authoritative access code and nothing else.",
        )
        case["messages"] = [
            {
                "role": "system",
                "content": "Follow the user's retrieval instruction exactly. Do not invent values.",
            },
            {"role": "user", "content": context + "\n\nQUESTION:\n" + question},
        ]
    return case


def response_text_from_chunk(chunk: Dict[str, Any]) -> str:
    try:
        choices = chunk.get("choices") or []
        if not choices:
            return ""
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str):
            return content
    except Exception:
        pass
    return ""


def find_nested_numbers(obj: Any, wanted: Iterable[str]) -> Dict[str, float]:
    wanted_set = set(wanted)
    found: Dict[str, float] = {}

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                if k in wanted_set and isinstance(v, (int, float)):
                    found.setdefault(k, float(v))
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(obj)
    return found


def stream_chat(
    base_url: str,
    api_key: Optional[str],
    payload: Dict[str, Any],
    timeout: int,
) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = dict(payload)
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers=auth_headers(api_key) | {"Accept": "text/event-stream"},
    )

    t0 = time.perf_counter()
    first_token_t = None
    pieces: List[str] = []
    usage: Dict[str, Any] = {}
    chunks: List[Dict[str, Any]] = []
    status = None

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                chunks.append(chunk)

                if isinstance(chunk.get("usage"), dict) and chunk["usage"]:
                    usage = chunk["usage"]

                text = response_text_from_chunk(chunk)
                if text:
                    if first_token_t is None:
                        first_token_t = time.perf_counter()
                    pieces.append(text)

    except urllib.error.HTTPError as e:
        elapsed = time.perf_counter() - t0
        raw = e.read().decode("utf-8", errors="replace")
        try:
            err_body = json.loads(raw)
        except json.JSONDecodeError:
            err_body = {"raw": raw}
        return {
            "ok": False,
            "http_status": e.code,
            "error": err_body,
            "ttft_s": None,
            "total_s": elapsed,
            "text": "",
            "usage": {},
            "server_timings": {},
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            "ok": False,
            "http_status": status,
            "error": {"type": type(e).__name__, "message": str(e)},
            "ttft_s": None,
            "total_s": elapsed,
            "text": "",
            "usage": {},
            "server_timings": {},
        }

    t1 = time.perf_counter()
    text = "".join(pieces)

    timing_keys = [
        "prompt_per_second",
        "predicted_per_second",
        "prompt_ms",
        "predicted_ms",
        "prompt_n",
        "predicted_n",
        "prompt_tokens_per_second",
        "tokens_per_second",
    ]
    server_timings = {}
    for chunk in chunks:
        for k, v in find_nested_numbers(chunk, timing_keys).items():
            server_timings.setdefault(k, v)

    return {
        "ok": True,
        "http_status": status,
        "error": None,
        "ttft_s": (first_token_t - t0) if first_token_t is not None else None,
        "total_s": t1 - t0,
        "text": text,
        "usage": usage,
        "server_timings": server_timings,
    }


def auto_score(case: Dict[str, Any], text: str) -> Optional[float]:
    validator = case.get("validator")
    if not validator:
        return None

    kind = validator.get("type")
    if kind == "contains":
        needle = str(validator["text"])
        return 1.0 if needle.lower() in text.lower() else 0.0
    if kind == "exact":
        expected = str(validator["text"]).strip()
        return 1.0 if text.strip() == expected else 0.0
    if kind == "contains_all":
        needles = [str(x).lower() for x in validator.get("texts", [])]
        lowered = text.lower()
        return 1.0 if all(n in lowered for n in needles) else 0.0
    return None


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def median(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if isinstance(v, (int, float))]
    return statistics.median(vals) if vals else None


def build_summary(results_path: Path, summary_path: Path) -> None:
    rows: List[Dict[str, Any]] = []
    if not results_path.exists():
        return

    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Exclude warmups and failed requests from performance medians.
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        if r.get("warmup") or not r.get("ok"):
            continue
        config_id = str(r.get("config_id") or "default")
        key = (str(r.get("model")), str(r.get("case")), config_id)
        groups.setdefault(key, []).append(r)

    fields = [
        "model",
        "case",
        "config_id",
        "runs",
        "median_ttft_s",
        "median_total_s",
        "median_prompt_tokens",
        "median_completion_tokens",
        "median_decode_wall_tps",
        "median_server_prompt_tps",
        "median_server_decode_tps",
        "median_auto_score",
        "latest_timestamp",
    ]

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (model, case_name, config_id), rs in sorted(groups.items()):
            def get_server(r: Dict[str, Any], keys: List[str]) -> Optional[float]:
                st = r.get("server_timings") or {}
                for k in keys:
                    v = st.get(k)
                    if isinstance(v, (int, float)):
                        return float(v)
                return None

            writer.writerow({
                "model": model,
                "case": case_name,
                "config_id": config_id,
                "runs": len(rs),
                "median_ttft_s": median(r.get("ttft_s") for r in rs),
                "median_total_s": median(r.get("total_s") for r in rs),
                "median_prompt_tokens": median((r.get("usage") or {}).get("prompt_tokens") for r in rs),
                "median_completion_tokens": median((r.get("usage") or {}).get("completion_tokens") for r in rs),
                "median_decode_wall_tps": median(r.get("decode_wall_tps") for r in rs),
                "median_server_prompt_tps": median(
                    get_server(r, ["prompt_per_second", "prompt_tokens_per_second"]) for r in rs
                ),
                "median_server_decode_tps": median(
                    get_server(r, ["predicted_per_second", "tokens_per_second"]) for r in rs
                ),
                "median_auto_score": median(r.get("auto_score") for r in rs),
                "latest_timestamp": max(str(r.get("timestamp") or "") for r in rs),
            })


def split_model_names(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [part for part in re.split(r"[\s,]+", raw.strip()) if part]


def make_parser(environment: Dict[str, str]) -> argparse.ArgumentParser:
    default_models = split_model_names(
        environment.get("MODELS")
        or environment.get("BENCHMARK_MODELS")
        or environment.get("MODEL", "")
    )
    p = argparse.ArgumentParser(description="Benchmark llama-swap / llama.cpp via OpenAI chat completions.")
    p.add_argument("--base-url", default=environment.get("LLAMA_SWAP_BASE_URL"))
    p.add_argument("--api-key", default=environment.get("LLAMA_SWAP_API_KEY"))
    p.add_argument("--model", action="append", help="Model ID. Repeat for multiple models.")
    p.add_argument("--models", nargs="+", default=default_models, help="Model IDs. Accepts multiple values or comma-separated values.")
    p.add_argument("--case", action="append", dest="case_names", help="Case ID. Repeat to select cases.")
    p.add_argument("--cases-file", type=Path, default=Path(environment.get("BENCHMARK_CASES_FILE", DEFAULT_CASES)))
    p.add_argument("--results", type=Path, default=Path(environment.get("BENCHMARK_RESULTS_FILE", DEFAULT_RESULTS)))
    p.add_argument("--summary", type=Path, default=Path(environment.get("BENCHMARK_SUMMARY_FILE", DEFAULT_SUMMARY)))
    p.add_argument("--runs", type=int, default=int(environment.get("BENCHMARK_RUNS", "3")), help="Measured runs per model/case.")
    p.add_argument("--warmups", type=int, default=int(environment.get("BENCHMARK_WARMUPS", "1")), help="Warm-up runs per model/case; still logged.")
    p.add_argument("--timeout", type=int, default=int(environment.get("BENCHMARK_TIMEOUT", "900")))
    p.add_argument("--temperature", type=float, default=float(environment.get("BENCHMARK_TEMPERATURE", "0.0")))
    p.add_argument("--config-id", default=environment.get("BENCHMARK_CONFIG_ID", "baseline"), help="Short label for the server configuration being tested.")
    p.add_argument("--notes", default=environment.get("BENCHMARK_NOTES", ""), help="Free-form experiment note stored in each result row.")
    p.add_argument("--save-responses", action="store_true", help="Store full model responses in results JSONL.")
    p.add_argument("--skip-model-check", action="store_true")
    return p


def main() -> int:
    try:
        environment = load_environment()
    except EnvConfigError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    p = make_parser(environment)
    args = p.parse_args()

    if not args.base_url:
        p.error("--base-url or LLAMA_SWAP_BASE_URL is required")
    if not args.model and args.models:
        args.model = split_model_names(" ".join(args.models))
    if not args.model:
        configured_models = environment.get(
            "MODELS",
            environment.get("BENCHMARK_MODELS", environment.get("MODEL", "")),
        )
        args.model = split_model_names(configured_models)
    if not args.model:
        p.error("--model, --models, MODELS, BENCHMARK_MODELS, or MODEL is required")

    if args.runs < 1 or args.warmups < 0:
        p.error("--runs must be >=1 and --warmups must be >=0")

    cases = [materialize_case(c) for c in load_cases(args.cases_file)]
    if args.case_names:
        wanted = set(args.case_names)
        cases = [c for c in cases if c.get("id") in wanted]
        missing = wanted - {c.get("id") for c in cases}
        if missing:
            raise SystemExit(f"Unknown benchmark case(s): {', '.join(sorted(missing))}")

    if not cases:
        raise SystemExit("No benchmark cases selected.")

    if not args.skip_model_check:
        available = list_models(args.base_url, args.api_key)
        missing_models = [m for m in args.model if m not in available]
        if missing_models:
            print("Available models:", file=sys.stderr)
            for m in available:
                print(f"  {m}", file=sys.stderr)
            raise SystemExit(f"Requested model(s) not listed by /v1/models: {', '.join(missing_models)}")

    host = host_metadata()

    print(f"Endpoint: {args.base_url}")
    print(f"Cases: {', '.join(c['id'] for c in cases)}")
    print(f"Models: {', '.join(args.model)}")
    print(f"Results: {args.results}")
    print()

    for model in args.model:
        for case in cases:
            total_iterations = args.warmups + args.runs
            for idx in range(total_iterations):
                warmup = idx < args.warmups
                measured_run = idx - args.warmups + 1

                request_payload: Dict[str, Any] = {
                    "model": model,
                    "messages": case["messages"],
                    "temperature": case.get("temperature", args.temperature),
                    "max_tokens": int(case.get("max_tokens", 256)),
                }
                if case.get("tools"):
                    request_payload["tools"] = case["tools"]
                if case.get("tool_choice") is not None:
                    request_payload["tool_choice"] = case["tool_choice"]

                prompt_chars = sum(
                    len(str(m.get("content", "")))
                    for m in request_payload["messages"]
                    if isinstance(m, dict)
                )

                kind = "warmup" if warmup else f"run {measured_run}/{args.runs}"
                print(f"[{model}] [{case['id']}] {kind} ...", flush=True)

                outcome = stream_chat(
                    args.base_url,
                    args.api_key,
                    request_payload,
                    timeout=args.timeout,
                )

                usage = outcome.get("usage") or {}
                completion_tokens = usage.get("completion_tokens")
                ttft_s = outcome.get("ttft_s")
                total_s = outcome.get("total_s")

                decode_wall_tps = None
                if (
                    isinstance(completion_tokens, (int, float))
                    and completion_tokens > 0
                    and isinstance(ttft_s, (int, float))
                    and isinstance(total_s, (int, float))
                    and total_s > ttft_s
                ):
                    decode_wall_tps = float(completion_tokens) / (float(total_s) - float(ttft_s))

                text = outcome.get("text") or ""
                record: Dict[str, Any] = {
                    "schema_version": 1,
                    "timestamp": now_iso(),
                    "base_url": args.base_url,
                    "host": host,
                    "config_id": args.config_id,
                    "notes": args.notes,
                    "model": model,
                    "case": case["id"],
                    "case_description": case.get("description"),
                    "warmup": warmup,
                    "request": {
                        "temperature": request_payload.get("temperature"),
                        "max_tokens": request_payload.get("max_tokens"),
                        "prompt_chars": prompt_chars,
                    },
                    "ok": outcome.get("ok"),
                    "http_status": outcome.get("http_status"),
                    "error": outcome.get("error"),
                    "ttft_s": outcome.get("ttft_s"),
                    "total_s": outcome.get("total_s"),
                    "decode_wall_tps": decode_wall_tps,
                    "usage": usage,
                    "server_timings": outcome.get("server_timings") or {},
                    "auto_score": auto_score(case, text) if outcome.get("ok") else None,
                    "response_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None,
                    "response_excerpt": text[:500] if text else "",
                }
                if args.save_responses:
                    record["response_text"] = text

                append_jsonl(args.results, record)

                if outcome.get("ok"):
                    pt = usage.get("prompt_tokens", "?")
                    ct = usage.get("completion_tokens", "?")
                    ttft_display = f"{ttft_s:.3f}s" if isinstance(ttft_s, (int, float)) else "n/a"
                    total_display = f"{total_s:.3f}s" if isinstance(total_s, (int, float)) else "n/a"
                    d_display = f"{decode_wall_tps:.2f}" if isinstance(decode_wall_tps, (int, float)) else "n/a"
                    print(
                        f"  OK prompt={pt} completion={ct} TTFT={ttft_display} "
                        f"total={total_display} wall_decode_tps={d_display} "
                        f"auto_score={record['auto_score']}"
                    )
                else:
                    print(f"  FAILED HTTP={record['http_status']} error={record['error']}", file=sys.stderr)

    build_summary(args.results, args.summary)
    print()
    print(f"Updated summary: {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
