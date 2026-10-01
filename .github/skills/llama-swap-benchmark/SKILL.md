---
name: llama-swap-benchmark
description: >-
  Use when working in the llama-swap-optimizer repository: running or modifying
  benchmark_all_models.py, benchmark_llama_swap.py, run_benchmark_tests.py, or
  their .env/.defaults/.env.example configuration. Triggers: "run a benchmark",
  "benchmark llama-swap", "sweep models", "DIRECT mode benchmark", "SWAP mode
  benchmark", "dry-run the benchmark", "add a model to the sweep", "edit
  .defaults", "edit .env.example", "benchmark test plan", "swap-smoke",
  "optimize llama-swap models", "tune inference settings".
---

# llama-swap Benchmark Tooling

Benchmark harness for a llama-swap / llama.cpp OpenAI-compatible endpoint. Three
scripts, dependency-free (standard library only), Python 3.10+.

## File map

| File | Purpose |
|------|---------|
| `benchmark_llama_swap.py` | Runs chat-completion benchmark cases against one or more models at a given endpoint. Lowest-level script. |
| `benchmark_all_models.py` | Orchestrates suites across models in SWAP mode (existing llama-swap endpoint) or DIRECT mode (starts llama-server in a Docker container itself). |
| `run_benchmark_tests.py` | Runs `benchmark_all_models.py` once per `[test NAME]` block in a literal KEY=value test-plan file (e.g. `examples/swap-smoke.conf`). |
| `benchmark_cases.json` | Stable prompt/workload definitions shared by all scripts. Each case declares `task` and `modalities`; most are text-only, a handful exercise image/audio input. |
| `synthetic_media.py` | Dependency-free, deterministic PNG/WAV generation for synthetic multimodal cases and capability probes. No third-party image/audio libraries. |
| `model_capabilities.py` | Resolves which input modalities (text/image/audio) a model accepts: declared config first, then a local cache, then a live content-agnostic probe. |
| `fetch_assets.py` | Manual, one-time script that downloads a few pinned, checksum-verified, CC-BY-NC-SA-4.0 real sample assets (photo, scanned document, speech clip) into gitignored `assets/`, for the `real_*` benchmark cases. Never run automatically. |
| `config_env.py` | Loads `.defaults` → `.env` → process environment, in that precedence order. |
| `.defaults` | Checked into git. Generic tuning defaults shared by every environment (run counts, timeouts, llama-server flags like CTX_SIZE/THREADS/BATCH_SIZE). |
| `.env` | Gitignored. Environment-specific config only: endpoint URL, API key, model selection/paths, DIRECT container name/host/port, optional `MODEL_CAPABILITIES_JSON`/asset path overrides. |
| `.env.example` | Template for `.env`; copy with `cp .env.example .env`. |
| `assets/` | Gitignored. Populated by `fetch_assets.py`; holds the real image/audio files used by `real_*` cases. |
| `ROADMAP.md` | Backlog, including a list of HuggingFace pipeline-tag families that are architecture-limited and permanently out of scope for this OpenAI-compatible-endpoint harness. |

## Configuration precedence

```
.defaults  <  .env  <  process environment variables  <  CLI flags
```

Never add environment-specific values (real endpoints, container names, model
file paths) to `.defaults` — those belong in `.env`. Never add generic tunables
(run counts, timeouts, batch sizes) to `.env` unless overriding the shared
default for this machine.

## Model selection

`MODELS` is the canonical selection variable (comma-separated model IDs).
Legacy aliases `MODEL`, `CORE_MODELS`, `VL_MODELS`, `BENCHMARK_MODELS` still
work as fallbacks for backward compatibility — prefer `MODELS` in new config.

`MODEL_DEFAULTS_JSON` is a JSON map of model ID → launch settings
(`MODEL_PATH`, `GPU_LAYERS`, `MMPROJ`, etc.), not a selection list.

## Modality/capability detection

`benchmark_llama_swap.py` skips a case against a model (writing a JSONL skip
record, not a failure) unless the model's detected capabilities cover the
case's `modalities`. Detection order: `MODEL_CAPABILITIES_JSON` (explicit,
per-model) → `MMPROJ`/`MODALITIES` inference from `MODEL_DEFAULTS_JSON` →
`model_capabilities_cache.json` (gitignored) → a live probe against the real
endpoint using a tiny synthetic image/audio payload from `synthetic_media.py`,
judged only by HTTP status, never response content. Pass
`--skip-capability-check` to force every case to run regardless; pass
`--refresh-capabilities` to ignore the cache and re-probe.

## Optimization context

Before tuning, load the machine-local optimization profile, using
[this template](./assets/optimization-profile.example.md) if one does not
exist. Keep the filled profile outside version control (for example under
`~/.config/llama-swap-optimizer/`); store only paths or identifiers, never API
keys. The profile is guidance for choosing and judging experiments, not a
configuration file consumed by the benchmark scripts. Do not place host-specific
facts in this skill or shared `.defaults`. Check the profile against the live
machine and active config at the start of each campaign; record unknowns rather
than filling them with guesses.

Resolve these questions before an unattended campaign:

- What is being optimized, in priority order (quality, TTFT, decode latency,
  throughput, context, memory), and what are the hard quality, resource, and
  runtime limits? If priorities or limits are missing, ask before choosing a
  winner or scheduling a large batch.
- Which active `llama-swap.yaml`, llama-swap/llama.cpp versions, hardware,
  quantized model files, and model capabilities are actually deployed? Verify
  the config path and model IDs rather than assuming `/v1/models` describes
  hardware, quantization, context size, or launch settings.
- Which prompt/output lengths, concurrency, model-switch frequency, and
  capabilities reflect real use? Map these to existing benchmark cases; add
  representative cases before optimizing for an uncovered workload.
- What is the baseline config, permissible parameter range, restart/unload
  policy, time budget, and report/archive destination? Do not restart a live
  service without authorization.

Reject a candidate that breaches a hard limit or degrades correctness, even
if it is faster. Compare eligible candidates on the prioritized metrics with
repeat runs against the same baseline, recording warm versus cold behavior,
failure rate, memory peaks, and run conditions. Treat marginal improvements as
inconclusive until repeated; `BENCHMARKING.md` provides initial thresholds,
not universal acceptance criteria. The existing exact/contains validators are
narrow checks, so review representative outputs for real task quality before
recommending a YAML change. Mark unmeasured metrics and uncertain conclusions
explicitly.

## Parameter selection

Before designing a tuning batch, follow the
[optimization search guide](./references/optimization-search.md). It adapts the
user's llama-swap optimization guidance to this repository's Python harness:
choose parameter families from the observed bottleneck, model architecture,
hardware, and workload; verify each variable is actually consumed and its
flag supported by the installed llama-server. Do not repeat an equivalent
trustworthy baseline. Prefer a broad, interpretable unattended batch of
single-variable probes, safe boundary tests, and a few motivated interactions
over a full Cartesian product. Only refine values after examining results.

DIRECT mode varies launch settings through `benchmark_all_models.py` but may
restart a Docker container. SWAP mode measures the running llama-swap config;
putting `GPU_LAYERS` or similar flags in a SWAP test block does not change the
server. Never assume the attached guide's `.sh` wrappers or example variable
names exist here. Do not change production YAML for exploratory sweeps without
explicit authorization and a rollback path.

## Live benchmark cycle

Use this cycle when the user requests an actual benchmark campaign, not when
editing the tools or previewing a plan. A live run requires an explicit request
to contact the server. Do not silently switch to DIRECT mode.

1. **Connect.** Resolve `LLAMA_SWAP_BASE_URL` and optional
  `LLAMA_SWAP_API_KEY` from the existing local configuration. Check that the
  endpoint responds before planning a live run; do not print the key or a
  credential-bearing URL. Record the start time, current config hash, and
  existing output file sizes/row counts so this campaign can be separated
  from previous results. Verify the optimization profile against the running
  deployment. See `BENCHMARKING.md` for environment checks.
2. **Discover.** Fetch `/v1/models` with the configured authentication and
  parse the JSON `data[].id` values. An unreachable endpoint or empty model
  list is a blocker, not a reason to guess model names.
3. **Verify selection.** Compare advertised IDs with `MODELS`, the requested
  model set, and model-specific capabilities (text, vision, context). State
  exactly which IDs and workloads will be tested and which will be excluded.
  Ask the user to resolve ambiguous selection or operational constraints
  *before* starting; if the request already defines them, proceed without an
  extra confirmation loop.
4. **Schedule experiments.** Build a test plan with descriptive, distinct
  `CONFIG_ID` values and one primary variable per comparison. Prefer a broad
  set of plausible model/case/configuration combinations over a tiny pilot:
  some failures are useful evidence. Use the parameter selection guide to
  prioritize the batch and include a reference baseline only when needed.
  Cover the profile's representative prompt
  sizes, output lengths, and concurrency where the harness supports them;
  document unsupported workload dimensions instead of claiming coverage.
  Respect stated resource/time limits and avoid incompatible workloads.
  Preview with `--dry-run`, then run the plan
  with `--continue-on-error` so a failed block does not cancel later ones.
  The wrapper executes blocks sequentially; do not claim they run in
  parallel. Once launched, let the campaign run without interactive input.
  Use completion notifications or sparse progress checks roughly every
  30-60 minutes, adjusted to model runtime; avoid repeated status polling or
  reading full logs into chat while work is in progress. Do not start a live
  batch if no unattended execution mechanism is available.
5. **Analyze after completion.** Inspect the new rows in
  `benchmark_results.jsonl`, the corresponding `benchmark_summary.csv`,
  `benchmark-logs/`, and `benchmark-wrapper-logs/`. Count successes,
  skips, failures, timeouts, and missing results; compare quality and latency
  against the baseline, including cold/warm behavior and variance. Diagnose
  failures from relevant logs before selecting the next batch. Use targeted summaries
  or appropriately sized analysis models where available, passing only
  relevant, credential-scrubbed excerpts rather than entire logs. The CSV is
  generated from the shared results file, so isolate this campaign by row,
  timestamp, and `config_id` before drawing conclusions.
6. **Iterate if needed.** If evidence leaves a material question open, return
  to selection and schedule targeted follow-up experiments around measured
  boundaries, close candidates, or unresolved interactions. Record why each
  new batch is needed; otherwise stop testing.
7. **Report.** Write a dated report covering the endpoint/configuration
  (without secrets), tested models and workloads, methods, failed and skipped
  runs, measurements, limitations, the profile's objective and hard limits,
  baseline comparisons, and actionable recommendations. Include
  concrete proposed changes to `llama-swap.yaml` with the affected model
  entries and reasoning. Only edit the real YAML after locating it and when
  the user has authorized configuration changes; do not guess its schema or
  replace unrelated settings. Clearly distinguish measured improvements from
  untested recommendations.
8. **Archive.** After all processes finish and the report is written, gather
  campaign-specific logs, plan, report, and results into a dated folder such
  as `benchmark-history/YYYY-MM-DD-HHMMSS/`. Do not move a shared results or
  summary file containing older campaigns: preserve it in place and archive
  a campaign-only extract or a copy, with the original row boundaries noted.
  Move only logs known to belong to this campaign, leaving previous logs and
  any files still in use untouched. Confirm the archive is readable and
  report its path.

## Safety rules (see AGENTS.md)

- Do not run live benchmarks against a real server unless the task explicitly
  calls for it — they send real requests and may start/stop a server.
- DIRECT mode restarts a Docker container by default
  (`DIRECT_RESTART_CONTAINER=1`) and requires Docker access. Do not enable it
  for a routine check.
- Preserve existing `benchmark_results.jsonl` / `benchmark_summary.csv` files;
  never truncate or overwrite them outside the script's own append logic.
- Never put API keys or other secrets anywhere but the gitignored `.env`; keep
  them out of logs, test-plan files, and commit history.
- Prefer `run_benchmark_tests.py --dry-run <plan>` to preview a run without
  contacting any server.
- `fetch_assets.py` is the only script that contacts a third-party host
  (huggingface.co); run it only when explicitly asked, never as part of a
  routine benchmark or test cycle.

## Common commands

```bash
# Unit tests (standard library unittest, no pytest)
python3 -m unittest discover -v

# CLI sanity checks
python3 benchmark_llama_swap.py --help
python3 benchmark_all_models.py --help
python3 run_benchmark_tests.py --help

# Safe, non-live preview of the checked-in smoke test plan
python3 run_benchmark_tests.py --dry-run examples/swap-smoke.conf

# Live SWAP-mode smoke run (only if explicitly requested)
python3 run_benchmark_tests.py examples/swap-smoke.conf

# List/fetch the real assets used by real_image_vqa/real_document_qa/real_audio_transcribe
# (one-time, explicit, contacts huggingface.co — never run as part of a routine check)
python3 fetch_assets.py --list
python3 fetch_assets.py
```

## Test-plan file format (`examples/*.conf`)

```ini
[test swap-smoke]
CONFIG_ID=swap-smoke
DIRECT_MODE=0
BENCH_CASES=quick_chat
SHORT_RUNS=1
LONG_RUNS=1
WARMUPS=0
TIMEOUT=300
```

Each `[test NAME]` block becomes one invocation of `benchmark_all_models.py`
with those KEY=value pairs as its environment. Keys containing `KEY`, `TOKEN`,
`SECRET`, or `PASSWORD` are redacted in `--dry-run`/log output.

## Working on this codebase

- Follow existing style: stdlib only, no third-party dependencies.
- This repo practices TDD — add/adjust a test in `tests/` first, confirm it
  fails, then implement, then rerun `python3 -m unittest discover -v`.
- After any change to `benchmark_llama_swap.py`, `benchmark_all_models.py`, or
  `run_benchmark_tests.py`, re-run the three `--help` invocations above plus
  the dry-run smoke check before claiming the change works.
- Test files mirror the source modules: `tests/test_benchmark_llama_swap.py`,
  `tests/test_benchmark_all_models.py`, `tests/test_benchmark_cases.py`,
  `tests/test_config_env.py`, `tests/test_run_benchmark_tests.py`,
  `tests/test_synthetic_media.py`, `tests/test_model_capabilities.py`,
  `tests/test_fetch_assets.py`. Network-touching code (`fetch_assets.py`,
  live probes in `model_capabilities.py`) is tested with mocked
  `urllib.request.urlopen`/injected callables, never real network calls.
- New benchmark case generator types go in `materialize_case()` in
  `benchmark_llama_swap.py`; new validator kinds go in `auto_score()`. Add a
  `task`/`modalities` pair to any new `benchmark_cases.json` entry.
