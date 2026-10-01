# Agent Instructions

## Project

This repository contains benchmark tooling for llama-swap:
- `benchmark_llama_swap.py` runs benchmark cases against a server.
- `benchmark_all_models.py` orchestrates model benchmark suites.
- `run_benchmark_tests.py` runs suites from test configuration files.

Keep changes focused and follow the existing Python style. Use Python 3.10 or
newer; the scripts and tests use only the standard library.

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