# llama-swap benchmark harness

## Files

- `benchmark_llama_swap.py` — dependency-free OpenAI-compatible benchmark runner.
- `benchmark_all_models.py` — model-suite orchestrator for SWAP and DIRECT modes.
- `run_benchmark_tests.py` — test-plan runner with dry-run support.
- `benchmark_cases.json` — stable benchmark workload definitions.
- `benchmark_results.jsonl` — append-only raw benchmark log.
- `benchmark_summary.csv` — generated median summary.
- `.defaults` — checked-in generic tuning defaults, shared by every environment.
- `.env` — local, ignored environment config: endpoints, credentials, model selection/paths.
- `BENCHMARKING.md` — operating procedure.

## Workspace Quick Start

Use Linux with Python 3.10 or newer. The runners and tests use only the Python
standard library, so no package installation or virtual environment is required.
Open this repository folder in VS Code, run **Python: Select Interpreter**, and
choose Python 3.10 or newer. Install the recommended Python extensions when
prompted. From **Tasks: Run Task**, choose `Tests: unit`, `Benchmark: show sweep
help`, or `Benchmark: dry-run SWAP smoke` for a non-live preview.

From the repository root, verify the checkout and preview the sample run:

```bash
cp .env.example .env
# Edit .env for the local endpoint, credentials, models, and paths.
# Generic tuning defaults (run counts, timeouts, llama-server flags) live in
# the checked-in .defaults file; override them in .env only when needed.
python3 -m unittest discover -v
python3 run_benchmark_tests.py --dry-run examples/swap-smoke.conf
```

The dry run prints the planned command without contacting llama-swap. It writes
only a wrapper summary under `benchmark-wrapper-logs/`.

To run the one-case SWAP smoke benchmark, first set `LLAMA_SWAP_BASE_URL`,
`MODEL`, and any required `LLAMA_SWAP_API_KEY` in the ignored `.env` file:

```bash
python3 run_benchmark_tests.py examples/swap-smoke.conf
```

The smoke plan requests one run of `quick_chat` for the configured Qwen model.
The model must already be advertised by the endpoint. Results are appended to
`benchmark_results.jsonl`; summaries and logs are written to the ignored local
output paths.

DIRECT mode requires Docker access and may restart the selected container by
default. Do not enable it for an initial workspace check; review the DIRECT
settings and their effects before running that mode.

## Authentication

If llama-swap requires an API key, place it only in `.env`:

```dotenv
LLAMA_SWAP_API_KEY=replace-with-your-key
```

The script sends it as:

```text
Authorization: Bearer YOUR_KEY
```

Do not put the API key into benchmark files or commit it to Git.

## First baseline

```bash
python3 benchmark_llama_swap.py \
  --model text-model \
  --config-id baseline \
  --runs 3
```

For a slower model, start with shorter tests:

```bash
python3 benchmark_llama_swap.py \
  --model text-model \
  --case quick_chat \
  --case it_troubleshooting \
  --case coding_bug \
  --config-id baseline \
  --runs 3
```

Then test long context separately:

```bash
python3 benchmark_llama_swap.py \
  --model text-model \
  --case long_context_8k \
  --case long_context_24k \
  --config-id baseline \
  --runs 2
```

## Comparing a tuning change

1. Run a baseline.
2. Back up `llama_swap.yaml`.
3. Change one important variable.
4. Restart llama-swap.
5. Run the same benchmark cases with a new `--config-id`.

Example:

```bash
python3 benchmark_llama_swap.py \
  --model text-model \
  --case quick_chat \
  --case long_context_8k \
  --config-id kv-q4 \
  --notes 'Changed K/V cache from q8_0 to q4_0' \
  --runs 3
```

Compare `benchmark_summary.csv`.

## What the metrics mean

- `TTFT`: request start until the first streamed text arrives. This includes any model-switch/load latency, queueing, prompt ingestion, and server-side preparation.
- `total_s`: complete wall-clock response time.
- `decode_wall_tps`: completion tokens divided by wall time after the first token. It is useful for comparisons but is not identical to llama.cpp's internal decode TPS.
- `server_prompt_tps`: populated only if llama.cpp exposes compatible internal timing fields in the streamed response.
- `server_decode_tps`: populated only if the server exposes compatible internal timing fields.
- `auto_score`: simple task-specific validator. It is not a general intelligence score.

The first request for a model may include model loading. Warm-up rows are therefore stored but excluded from summary medians.

## Results are append-only

Do not edit old benchmark rows to make later results look cleaner. If a run was invalid, add a note in the next experiment or archive the entire result file and start a clearly named new series.

A benchmark is useful only if the server configuration used for it can be identified.
