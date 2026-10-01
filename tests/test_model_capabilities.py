import json
import tempfile
import unittest
from pathlib import Path

import model_capabilities as capabilities


class DeclaredCapabilitiesTests(unittest.TestCase):
    def test_returns_none_when_nothing_declared(self):
        self.assertIsNone(capabilities.declared_capabilities("model-a", {}))

    def test_model_capabilities_json_takes_precedence_over_mmproj(self):
        environment = {
            "MODEL_CAPABILITIES_JSON": json.dumps({"model-a": ["text", "audio"]}),
            "MODEL_DEFAULTS_JSON": json.dumps({"model-a": {"MMPROJ": "/path/to/proj"}}),
        }
        result = capabilities.declared_capabilities("model-a", environment)
        self.assertEqual(result, {"text", "audio"})

    def test_mmproj_presence_implies_text_and_image(self):
        environment = {"MODEL_DEFAULTS_JSON": json.dumps({"model-a": {"MMPROJ": "/path/to/proj"}})}
        self.assertEqual(capabilities.declared_capabilities("model-a", environment), {"text", "image"})

    def test_model_defaults_modalities_key_is_honored(self):
        environment = {"MODEL_DEFAULTS_JSON": json.dumps({"model-a": {"MODALITIES": ["image", "audio"]}})}
        self.assertEqual(capabilities.declared_capabilities("model-a", environment), {"text", "image", "audio"})

    def test_malformed_json_is_treated_as_undeclared(self):
        environment = {"MODEL_CAPABILITIES_JSON": "{not json"}
        self.assertIsNone(capabilities.declared_capabilities("model-a", environment))


class CacheTests(unittest.TestCase):
    def test_cache_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.json"
            capabilities._write_cache(cache_path, {"k": {"modalities": ["text", "image"]}})
            self.assertEqual(capabilities._read_cache(cache_path), {"k": {"modalities": ["text", "image"]}})

    def test_missing_cache_file_returns_empty_dict(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "missing.json"
            self.assertEqual(capabilities._read_cache(cache_path), {})


class ProbeTests(unittest.TestCase):
    def test_probe_image_support_true_on_http_200(self):
        self.assertTrue(capabilities.probe_image_support(lambda payload: (200, {}), "model-a"))

    def test_probe_image_support_false_on_non_200(self):
        self.assertFalse(capabilities.probe_image_support(lambda payload: (400, {}), "model-a"))

    def test_probe_audio_support_false_when_post_chat_raises(self):
        def boom(payload):
            raise RuntimeError("network down")

        self.assertFalse(capabilities.probe_audio_support(boom, "model-a"))


class DetectCapabilitiesTests(unittest.TestCase):
    def test_declared_short_circuits_probe_and_cache(self):
        environment = {"MODEL_CAPABILITIES_JSON": json.dumps({"model-a": ["text", "image"]})}

        def fail_if_called(payload):
            raise AssertionError("post_chat should not be called when declared")

        result = capabilities.detect_capabilities(
            model="model-a",
            base_url="http://localhost",
            environment=environment,
            post_chat=fail_if_called,
        )
        self.assertEqual(result, {"modalities": ["image", "text"], "source": "declared"})

    def test_uses_cache_when_available(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.json"
            key = capabilities._cache_key("http://localhost", "model-a")
            capabilities._write_cache(cache_path, {key: {"modalities": ["text", "audio"]}})

            def fail_if_called(payload):
                raise AssertionError("post_chat should not be called when cached")

            result = capabilities.detect_capabilities(
                model="model-a",
                base_url="http://localhost",
                environment={},
                post_chat=fail_if_called,
                cache_path=cache_path,
            )
            self.assertEqual(result, {"modalities": ["audio", "text"], "source": "cached"})

    def test_probes_and_writes_cache_when_nothing_declared_or_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.json"

            def post_chat(payload):
                content = payload["messages"][0]["content"]
                has_image = any(part.get("type") == "image_url" for part in content)
                return (200, {}) if has_image else (400, {})

            result = capabilities.detect_capabilities(
                model="model-a",
                base_url="http://localhost",
                environment={},
                post_chat=post_chat,
                cache_path=cache_path,
            )
            self.assertEqual(result, {"modalities": ["image", "text"], "source": "probed"})
            cached = capabilities._read_cache(cache_path)
            key = capabilities._cache_key("http://localhost", "model-a")
            self.assertEqual(sorted(cached[key]["modalities"]), ["image", "text"])

    def test_probe_disabled_defaults_to_text_only(self):
        def fail_if_called(payload):
            raise AssertionError("post_chat should not be called when probe=False")

        result = capabilities.detect_capabilities(
            model="model-a",
            base_url="http://localhost",
            environment={},
            post_chat=fail_if_called,
            use_cache=False,
            probe=False,
        )
        self.assertEqual(result, {"modalities": ["text"], "source": "default"})


if __name__ == "__main__":
    unittest.main()
