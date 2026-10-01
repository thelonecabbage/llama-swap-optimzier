# Roadmap

Planned and candidate improvements for the benchmark harness. This is a
backlog, not a commitment — items move up when a real benchmarking need
justifies the work. Keep entries specific enough to act on; vague entries
("make it better") are not useful here.

## Measurement gaps

- **Concurrency workload.** `benchmark_all_models.py` runs blocks
  sequentially; there is no way to measure throughput, tail latency, or
  fairness under simultaneous requests. Add a concurrent-request case type
  and report per-request and aggregate latency.
- **Longer generation cases.** Current cases cap at ~192 output tokens, too
  short to reliably rank decode acceleration (threads, speculative decoding,
  MTP). Add one or more long-output cases (coding, prose, reasoning) alongside
  the existing short ones.
- **True cold-start measurement.** Warm-up rows are logged but are not
  guaranteed to reflect a true cold load, since llama-swap TTL can keep a
  model resident. Add an explicit unload/restart step (where the installed
  llama-swap version supports it) and a dedicated cold-load case.
- **Automated memory/telemetry capture.** Peak memory is currently recorded
  manually by watching `free -h` / `ps` during a run. Add an optional sampling
  thread that records RSS/VRAM over the run and attaches it to the result row.
- **Repeated-prefix / prompt cache reuse.** No case currently measures
  follow-up TTFT after a shared prefix, so KV/prompt-cache reuse savings are
  unmeasured. Add a repeated-prefix case pair (cold request, then a follow-up
  with a shared prefix).
- **Speculative decoding / draft acceptance.** No case records draft
  acceptance rate or speculative-decoding overhead. Add instrumentation when
  the server exposes these fields, and a case that compares a draft model or
  ngram speculation against baseline.
- **Vision and embedding workloads.** `benchmark_cases.json` is text-chat
  only; there is no coverage for multimodal or embedding-specific
  correctness/latency. Add separate case sets rather than inferring coverage
  from chat TPS.

## Tooling and reporting

- **Structured comparison report.** Today, comparing `CONFIG_ID` runs means
  manually reading `benchmark_summary.csv` and `benchmark_results.jsonl`. Add
  a small script/report that diffs two config IDs and prints percentage
  deltas for TTFT, total time, and TPS, flagging changes within noise.
- **Config hash capture.** `BENCHMARKING.md` recommends manually hashing
  `llama_swap.yaml` into `--notes`. Consider having the wrapper capture and
  store this automatically when the file is reachable.
- **DIRECT-mode rollback guarantees.** DIRECT mode can restart the selected
  container but does not promise recovery in all failure modes. Document (or
  implement) a safer rollback path before DIRECT mode sees heavier use.

## Out of scope for now

- A GUI or web dashboard — CSV/JSONL plus ad hoc scripts are sufficient at
  current usage levels.
- Multi-host/distributed benchmarking — this harness targets a single
  configured endpoint at a time.
