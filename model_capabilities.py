"""Detect which input modalities a model/endpoint actually accepts.

Capabilities are resolved in order:
1. Declared in config (`MODEL_CAPABILITIES_JSON`, or inferred from `MMPROJ`
   inside `MODEL_DEFAULTS_JSON`) -- trusted without a live check.
2. A cached result from a previous live probe.
3. A fresh live probe: a minimal request per modality against the real
   endpoint, judged only by whether the server accepts it (HTTP 200), never
   by response content.

A model is always assumed to accept "text".
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Set, Tuple

import synthetic_media

ALL_MODALITIES = ("text", "image", "audio")
DEFAULT_CACHE_PATH = Path(__file__).with_name("model_capabilities_cache.json")

PostChat = Callable[[Dict[str, Any]], Tuple[int, Dict[str, Any]]]


def _split_modalities(raw: Any) -> Set[str]:
    if isinstance(raw, list):
        return {str(m).strip() for m in raw if str(m).strip()}
    return {m.strip() for m in str(raw).split(",") if m.strip()}


def declared_capabilities(model: str, environment: Mapping[str, str]) -> Optional[Set[str]]:
    """Return explicitly configured capabilities, or None if nothing is declared."""
    raw_capabilities = environment.get("MODEL_CAPABILITIES_JSON", "")
    if raw_capabilities:
        try:
            parsed = json.loads(raw_capabilities)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict) and model in parsed:
            modalities = _split_modalities(parsed[model])
            modalities.add("text")
            return modalities

    raw_defaults = environment.get("MODEL_DEFAULTS_JSON", "")
    if raw_defaults:
        try:
            parsed = json.loads(raw_defaults)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            settings = parsed.get(model)
            if isinstance(settings, dict):
                if settings.get("MODALITIES"):
                    modalities = _split_modalities(settings["MODALITIES"])
                    modalities.add("text")
                    return modalities
                if settings.get("MMPROJ"):
                    return {"text", "image"}

    return None


def _cache_key(base_url: str, model: str) -> str:
    return f"{base_url.rstrip('/')}|{model}"


def _read_cache(cache_path: Path) -> Dict[str, Any]:
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_cache(cache_path: Path, data: Dict[str, Any]) -> None:
    try:
        cache_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        pass


def _probe_modality(post_chat: PostChat, model: str, content_part: Dict[str, Any]) -> bool:
    payload = {
        "model": model,
        "max_tokens": 1,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "ok"}, content_part]},
        ],
    }
    try:
        status, _body = post_chat(payload)
    except Exception:
        return False
    return status == 200


def probe_image_support(post_chat: PostChat, model: str) -> bool:
    import base64

    data_uri = "data:image/png;base64," + base64.b64encode(synthetic_media.tiny_probe_png()).decode("ascii")
    return _probe_modality(post_chat, model, {"type": "image_url", "image_url": {"url": data_uri}})


def probe_audio_support(post_chat: PostChat, model: str) -> bool:
    import base64

    data = base64.b64encode(synthetic_media.tiny_probe_wav()).decode("ascii")
    return _probe_modality(post_chat, model, {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}})


def detect_capabilities(
    *,
    model: str,
    base_url: str,
    environment: Mapping[str, str],
    post_chat: PostChat,
    cache_path: Path = DEFAULT_CACHE_PATH,
    use_cache: bool = True,
    probe: bool = True,
) -> Dict[str, Any]:
    declared = declared_capabilities(model, environment)
    if declared is not None:
        return {"modalities": sorted(declared), "source": "declared"}

    key = _cache_key(base_url, model)
    if use_cache:
        cached = _read_cache(cache_path).get(key)
        if isinstance(cached, dict) and isinstance(cached.get("modalities"), list):
            return {"modalities": sorted(cached["modalities"]), "source": "cached"}

    if not probe:
        return {"modalities": ["text"], "source": "default"}

    modalities = {"text"}
    if probe_image_support(post_chat, model):
        modalities.add("image")
    if probe_audio_support(post_chat, model):
        modalities.add("audio")

    if use_cache:
        cache = _read_cache(cache_path)
        cache[key] = {
            "modalities": sorted(modalities),
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        _write_cache(cache_path, cache)

    return {"modalities": sorted(modalities), "source": "probed"}
