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

## 9. Multimodal and task-coverage cases

`benchmark_cases.json` also includes:

- Text-only NLP task cases (`text_classification_sentiment`,
  `zero_shot_topic_classification`, `extractive_qa`, `summarization_short`,
  `translation_en_es`, `table_question_answering`) covering common
  HuggingFace text-generation-style pipeline tags.
- Synthetic multimodal cases (`synthetic_image_grid_vqa`,
  `synthetic_image_sequence_tracking`, `synthetic_audio_tone_count`) that
  exercise image/audio request plumbing and latency with deterministic,
  dependency-free generated content (see `synthetic_media.py`). These have
  reliable auto-validators since the expected answer is known exactly.
- Real-asset cases (`real_image_vqa`, `real_document_qa`,
  `real_audio_transcribe`) that use real photo/document/speech content for
  more representative (if non-deterministic) vision/audio quality checks.
  Run `python3 fetch_assets.py` once to download these (pinned, checksum
  verified, CC-BY-NC-SA-4.0 licensed samples from
  `huggingface/documentation-images`); the benchmark skips these cases with
  a clear message if the assets aren't present. `real_document_qa` and
  `real_audio_transcribe` have no automatic validator -- review responses
  manually, since the exact expected text wasn't independently verified.

`benchmark_llama_swap.py` only runs a case against a model if the model's
detected capabilities cover the case's required modalities. Capabilities are
resolved from `MODEL_CAPABILITIES_JSON`/`MODEL_DEFAULTS_JSON` (see
`.env.example`), a local cache, or a live, content-agnostic probe against the
endpoint (pass `--skip-capability-check` to force every case to run anyway).

Out of scope: pipeline tags that an OpenAI-compatible chat/completions
endpoint cannot serve (image/video/3D generation, object detection/
segmentation, depth estimation, tabular, reinforcement learning, graph ML,
text-to-speech output, etc.) are intentionally not covered here.

Do not trade a meaningful quality regression for a small speed increase.
