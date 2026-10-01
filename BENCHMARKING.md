# Benchmarking procedure

## 1. Capture the active configuration

Before each experiment:

```bash
date
sha256sum "$LLAMA_SWAP_CONFIG_FILE"
docker ps --filter "name=$DIRECT_CONTAINER"
docker inspect "$DIRECT_CONTAINER" --format '{{.Config.Image}}'
free -h
```

If practical, save the YAML hash in the `--notes` field.

## 2. Confirm the model starts correctly

```bash
curl -s \
  -H "Authorization: Bearer $LLAMA_SWAP_API_KEY" \
  "${LLAMA_SWAP_BASE_URL%/}/v1/models" | python3 -m json.tool
```

A model returning HTTP 500 is not ready for performance benchmarking.

## 3. Run baseline cases

Start with:

- `quick_chat`
- `it_troubleshooting`
- `coding_bug`
- `rag_grounding`

Then add the long-context tests where the model/context supports them.

## 4. Monitor memory during a test

In another terminal:

```bash
watch -n 1 'free -h; echo; ps -eo pid,comm,rss,%mem --sort=-rss | head -15'
```

For AMD/Vulkan work, add the GPU/Vulkan monitoring tools available on the host.
Record observed peak memory in the experiment notes until automated telemetry is added.

## 5. Change one primary variable

Good experiment IDs:

- `threads-8`
- `ubatch-128`
- `gpu-layers-8`
- `kv-q4`
- `ctx-16k`
- `reasoning-off`
- `flash-attn-on`

Bad experiment IDs:

- `faster-settings`
- `new-config`
- `everything-tuned`

## 6. Keep/reject rule

Do not keep a tuning change because one number improved.

For interactive models, a change is normally worth keeping when:

- correctness/auto-score does not regress,
- HTTP/runtime stability does not regress,
- median TTFT or total latency improves materially, or
- usable context increases materially without making latency unacceptable.

As a starting threshold, treat <5% changes as noise unless repeated runs are very stable.
A 10–15% improvement is usually large enough to investigate seriously.

For long-context/RAG workloads, prompt ingestion and retrieval correctness can outweigh raw decode speed.

## 7. Cold vs warm behavior

`benchmark_llama_swap.py` logs warm-up requests but excludes them from summary medians.

This is intentional:

- warmed runs estimate steady-state inference,
- warm-up/cold-ish rows expose model-switch and first-use cost.

Because llama-swap TTL can keep a model resident, a warm-up row is not guaranteed to be a true cold load. If true cold-load benchmarking is needed, explicitly unload/restart using a method supported by the installed llama-swap version and document it.

## 8. Quality evaluation

Automatic validators only test narrow objective properties.

For coding/IT work, manually review representative responses for:

- correctness,
- dangerous/destructive commands,
- invented flags,
- failure to diagnose before changing things,
- code quality,
- instruction following.

Do not trade a meaningful quality regression for a small speed increase.
