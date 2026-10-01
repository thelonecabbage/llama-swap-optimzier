# Agent Instructions

## Project

This repository contains benchmark tooling for llama-swap:
- `benchmark_llama_swap.py` runs benchmark cases against a server.
- `benchmark_all_models.py` orchestrates model benchmark suites.
- `run_benchmark_tests.py` runs suites from test configuration files.
- `synthetic_media.py` generates deterministic synthetic images/audio for
  multimodal cases (no third-party image/audio libraries).
- `model_capabilities.py` resolves which input modalities (text/image/audio)
  a model accepts, via declared config, cache, or live probing.
- `fetch_assets.py` is a manual, one-time script that downloads a few
  pinned, checksum-verified, licensed real sample assets (image/audio) into
  the gitignored `assets/` directory for the real-asset benchmark cases.

Keep changes focused and follow the existing Python style. Use Python 3.10 or
newer; the scripts and tests use only the standard library (including
`fetch_assets.py`, which uses only `urllib`/`hashlib`).

## Validation

Run the unit tests with:

```sh
python3 -m unittest discover -v
```

Check each command-line interface with:

```sh
python3 benchmark_llama_swap.py --help
python3 benchmark_all_models.py --help
python3 run_benchmark_tests.py --help
```

Use `run_benchmark_tests.py --dry-run` when checking a test configuration
without executing benchmarks. The checked-in `examples/swap-smoke.conf` is the
safe preview plan for a one-case SWAP run.

## Benchmark Safety

Benchmark runs may send requests to a configured server, start or stop a server, and write result and log files. Avoid running live benchmarks unless the task calls for them. Preserve existing result files, and do not expose API keys or other secrets in logs or changes.

`fetch_assets.py` is the only script that reaches out to a third-party host
(Hugging Face) and only when run explicitly by name — never invoke it as part
of a routine benchmark run. Benchmark cases needing those assets skip
gracefully with a clear message when `assets/` is absent.