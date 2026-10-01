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
- **Vision and embedding workloads.** `benchmark_cases.json` now includes
  synthetic and real-asset vision/audio cases (VQA, document QA, image
  sequences, tone counting, transcription) plus a capability-detection layer
  that skips cases a model can't accept. Embedding-specific correctness/
  latency (`/v1/embeddings`) is still uncovered — add a dedicated embedding
  case set (retrieval accuracy on a small fixed corpus, embedding latency vs.
  input length) rather than inferring it from chat TPS.

## Uncovered task types (architecture-limited)

llama-swap/llama.cpp exposes an OpenAI-compatible chat/completions and
embeddings endpoint. The following HuggingFace pipeline-tag families cannot
be exercised through that interface at all, regardless of case design, so
they stay out of scope unless the harness grows support for a different
serving backend (e.g. diffusers, a TTS server, a tabular/RL runtime):

- **Image/video/3D generation.** text-to-image, image-to-image,
  image-to-video, text-to-video, text-to-3d, image-to-3d, unconditional
  image generation.
- **Dense vision perception.** object detection, zero-shot object detection,
  image segmentation, depth estimation, keypoint detection, mask generation.
- **Audio generation.** text-to-speech, text-to-audio, voice activity
  detection — llama.cpp can consume audio input but has no audio output path.
- **Structured/tabular data.** tabular classification, tabular regression,
  time-series forecasting.
- **Reinforcement learning.** RL and robotics pipeline tags assume a
  simulator/environment loop with no chat-completions equivalent.
- **Graph machine learning.** graph classification/regression and related
  tags operate on graph-structured inputs, not token sequences.
- **Any-to-any / multimodal generation.** tags that produce new
  image/audio/video outputs from mixed inputs (as opposed to describing or
  answering questions about them, which the existing vision/audio cases
  already cover).



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
