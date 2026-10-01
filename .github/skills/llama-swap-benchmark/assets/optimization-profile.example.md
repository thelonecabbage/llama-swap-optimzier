# Llama-swap optimization profile (template)

Copy this to a private location outside version control, such as
`~/.config/llama-swap-optimizer/optimization-profile.md`. This is planning
context, not a file the benchmark scripts read. Record `unknown` for facts not
yet verified. Do not include API keys, tokens, or credential-bearing URLs;
those belong in the gitignored `.env`.

## Objective and limits

- Use case and model roles (chat, coding, RAG, vision): unknown
- Priority order (quality, TTFT, decode speed, throughput, context, memory): unknown
- Hard correctness/quality floor and how it is evaluated: unknown
- Maximum acceptable TTFT / latency and minimum throughput: unknown
- RAM/VRAM ceiling and minimum usable context: unknown
- Campaign time budget, maintenance window, and acceptable failure rate: unknown
- Rule for accepting changes (repeat count, meaningful improvement threshold): unknown

## Deployment

- Active `llama-swap.yaml` path and config hash: unknown
- llama-swap version, llama.cpp/llama-server version, backend: unknown
- CPU, RAM, GPU(s), VRAM, drivers and relevant acceleration: unknown
- Model IDs, model file paths, quantizations, sizes, vision projectors, context limits: unknown
- Current per-model launch flags, TTL/unload policy, and server concurrency: unknown
- Endpoint identity (no credentials) and whether it is shared with live users: unknown
- Permitted settings to vary and safe ranges per model: unknown
- Permission to restart/unload services; rollback procedure: unknown

## Representative workload

- Model-to-task mapping and important real-world prompts: unknown
- Prompt lengths and output lengths (typical, high percentile, maximum): unknown
- Concurrent request levels and traffic pattern: unknown
- Frequency of model switches and cold vs warm performance importance: unknown
- Required capabilities (vision, tools, structured output, long context): unknown
- Benchmark case IDs matching these workloads; missing cases to add: unknown

## Baseline and outputs

- Baseline config ID, measurement date, results/log paths: unknown
- Baseline quality, error rate, TTFT, latency, throughput, peak memory: unknown
- Required comparison conditions (same model, prompts, host load, versions): unknown
- Dated report destination and dated archive root: unknown
- YAML change policy (proposals only, or permission to edit active config): unknown

Review this profile against the active server and configuration before each
campaign. Document measured facts separately from assumptions, and update it
when models, hardware, or objectives change.