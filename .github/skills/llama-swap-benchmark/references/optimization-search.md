# Model optimization search guide

Use this guide with the optimization profile in the parent skill. It adapts the
provided llama-swap optimization skill to the actual Python harness; its sample
values are hypotheses, not universally good settings.

## Establish what can be measured

1. Inspect the active `llama-swap.yaml`, model paths, quantizations, backend,
   hardware, previous results, and the installed `/app/llama-server --help` and
   `--version` when DIRECT mode is authorized. Check that each proposed
   environment variable is read by `benchmark_all_models.py`, maps to a
   supported flag, and changes the **resolved** server settings in the logs.
   An accepted flag does not prove the backend used it. Avoid silent auto-fit
   changes: compare actual placement/context; test `FIT=off` only if supported.
2. Reuse an equivalent, reproducible baseline rather than repeating an
   exhaustive sweep. Otherwise measure one with a recorded model file, build,
   backend, config hash, context, placement, threads, batch, KV, flash attention,
   and reasoning policy. Keep prompts, generation settings, host load, and
   warm/cold state comparable. Distinct `CONFIG_ID` values identify each run.
3. Baseline each relevant case before choosing knobs. This harness measures
   TTFT, total time, prompt/decode rates when available, token counts, and
   narrow auto-scores. Review JSONL failures, variance, and representative
   responses; a CSV median excludes failed and warmup runs. Memory, load time,
   draft acceptance, and true cold-start behavior need separate measurement
   unless the actual artifacts contain them. Never invent missing evidence.

## Choose parameters by bottleneck and workload

| Observed constraint | First parameter families | Appropriate tests and limits |
| --- | --- | --- |
| Poor TTFT or prompt ingestion | `GPU_LAYERS`, `BATCH_SIZE`, `UBATCH_SIZE`, `THREADS_BATCH`, `FLASH_ATTN`, KV placement | `rag_grounding`, `long_context_8k`, `long_context_24k`; verify context fits and record memory. |
| Slow generation | Placement, `THREADS`, supported `SPEC_TYPE`/`MTP_N` or draft model | Add long-output cases: current cases generate at most 192 tokens, too short to rank decode acceleration reliably. Record acceptance if available. |
| Cannot fit desired context | `CTX_SIZE`, `CACHE_TYPE_K/V`, `KV_OFFLOAD`, placement, optional `FIT` | Run 8K/24K retrieval at the intended context, check quality and memory, then refine the fit boundary. Lower-precision KV is not a free quality improvement. |
| Slow loading or model switching | Supported `MMAP`, `MLOCK`, `LAZY_MODE`, llama-swap residency policy | Add controlled switch/cold-start runs and load/page-fault measurements; warmups alone do not prove cold loads. |
| Many simultaneous requests | `PARALLEL`, placement, context/KV and cache settings | Add a concurrency workload measuring throughput, tail latency, fairness, and memory. The current wrapper is sequential. |
| Repeated large prefixes | Supported `CACHE_RAM` and server cache/slot controls | Add repeated-prefix requests and measure follow-up TTFT; a single `rag_grounding` request cannot show reuse. |
| Quality/latency tradeoff for thinking | `REASONING`, `REASONING_EFFORT`, `REASONING_BUDGET` | Change only in a separate policy sweep; measure task success and generated tokens, not just raw TPS. |

Do not claim vision, embeddings, long generation, speculative acceptance, or
concurrency are covered by the checked-in `benchmark_cases.json`. Add suitable
tests or state that these dimensions remain unmeasured.

## Architecture-dependent ordering

- **Dense:** coarse CPU/partial/full `GPU_LAYERS` and device placement first;
  then KV/memory, batch/threads, flash attention, and supported speculation.
  `CPU_FFN` is an optional later partial-CPU placement probe, not a default.
- **MoE:** check `CPU_MOE`, `CPU_MOE_LAYERS` and GPU-layer interactions before
  dense-style micro-tuning. Record actual expert placement, page faults, and
  useful throughput; total parameters do not equal active parameters per token.
- **Vision:** preserve `MMPROJ` and verify multimodal workloads and projector
  placement before judging changes. Text-only cases cannot establish vision
  quality. **Embedding:** use embedding-specific tests, not chat TPS.
- **Shared-memory APU:** test partial offload, `KV_OFFLOAD`, `OP_OFFLOAD`, and
  memory headroom explicitly. More GPU layers may be slower or exceed the
  Vulkan heap even when host RAM remains available.
- **Large context:** test KV precision and flash attention only when supported;
  scale context to real usage rather than selecting the advertised maximum.
  RoPE/YaRN/SWA controls are model-specific quality experiments, not generic
  throughput toggles.

Then consider decode acceleration: native MTP, a draft model, or ngram
speculation only if the model and installed server support the method. Start
with off versus a few supported draft depths; compare long outputs across
coding, prose, and reasoning. Record draft acceptance, overhead, and memory if
available. Keep reasoning policy fixed during hardware comparisons; test effort
or token budget separately and then a small MTP-by-reasoning interaction matrix.
Late-stage refinements include repacking, CPU affinity/polling, lazy loading,
tensor overrides, and cache behavior, each only when relevant and measurable.

## Design a broad but interpretable batch

- Include a reference to the existing baseline or rerun it when environment
  drift makes comparison invalid. Do not duplicate a valid exhaustive baseline.
- Sweep high-impact placement and memory settings coarsely (for example GPU
  layers 0/4/8/12/16; threads 4/6/8; batch 128/256/512; ubatch 32/64/128)
  only within the supported, safe ranges for this model and host. These are
  examples, never a universal grid. Vary one primary setting per probe.
- Include expected-safe failure boundaries when cleanup is reliable; e.g. if
  16 GPU layers fit and 20 fail, a follow-up can try 17/18/19. Do not turn a
  likely host-crashing or data-losing test into an unattended experiment.
- Add a few justified interactions such as best GPU range + CPU-MoE, GPU + KV
  precision/offload, KV savings + needed context, or best speculation + chosen
  reasoning policy. Do not run the full Cartesian product by default.
- Choose repetitions to match noise (e.g. `SHORT_RUNS=3`, `LONG_RUNS=2`,
  `WARMUPS=1` initially). Repeat close finalists more often, use fewer runs
  for slow boundary probes, and retain the same workload for A/B comparison.
- Estimate total duration and disk/memory limits, preview the plan using
  `run_benchmark_tests.py --dry-run`, and use `--continue-on-error` for safe
  independent tests. The wrapper runs blocks **sequentially**. In DIRECT mode
  verify Docker access, container restart policy, cleanup, and rollback before
  launching; it does not promise llama-swap restoration in all failure modes.

`MODELS` in `.env` takes precedence over `MODEL` for suite selection. Set an
explicit `MODELS` in each plan block when narrowing scope; for DIRECT mode also
set `MODEL` to the same single model ID because DIRECT server startup reads
`MODEL`. Resolve its model path via `MODEL_DEFAULTS_JSON` or `MODEL_PATH`.
A plan block changes launch flags only in DIRECT mode. In SWAP mode the running YAML determines these settings;
do not label env-only changes as experiments if they do not reach the server.
Keep `CONFIG_ID` descriptive and never place secrets in a test plan.

## Interpret, iterate, and hand off

Use the CSV for a quick overview, JSONL for per-run comparisons, and only
relevant logs for failure diagnosis (unsupported flags, OOM, Vulkan allocation,
timeout, HTTP errors, CPU fallback). Compare baseline-relative prompt TPS,
decode TPS, TTFT, end-to-end time, correctness, stability, and measured memory.
Report both absolute and percentage deltas; differences within variance are
inconclusive. A faster run that violates a quality or resource floor is not a
winner. Consider Pareto-efficient options for interactive, long-context, and
deep-reasoning roles instead of forcing one universal configuration.

If evidence is insufficient, propose a specific next batch around an observed
boundary, close candidates, missing workload, or interaction; stop testing
decisive losers. If evidence supports a recommendation, provide a minimal YAML
diff or complete model segment preserving model ID, path, projector, aliases,
metadata, and unrelated settings. State the tradeoffs and the smallest final
SWAP-mode validation run. If a user must run the batch elsewhere, request the
CSV and JSONL afterward, with relevant logs for failures; do not ask them to
manually transcribe measurements already in the artifacts.