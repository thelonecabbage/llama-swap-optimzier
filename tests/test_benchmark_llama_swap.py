import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import benchmark_llama_swap as benchmark


class FakeResponse:
    def __init__(self, body: bytes, status: int = 200):
        self.lines = body.splitlines(keepends=True)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return b"".join(self.lines)

    def __iter__(self):
        return iter(self.lines)


class HttpTests(unittest.TestCase):
    def test_auth_headers_omit_empty_key_and_add_bearer_key(self):
        self.assertNotIn("Authorization", benchmark.auth_headers(None))
        self.assertEqual(
            benchmark.auth_headers("secret")["Authorization"], "Bearer secret"
        )

    def test_http_json_posts_payload_and_decodes_response(self):
        response = FakeResponse(b'{"ok": true}')
        with mock.patch.object(benchmark.urllib.request, "urlopen", return_value=response) as urlopen:
            status, body = benchmark.http_json(
                "http://example.test/api", method="POST", payload={"x": 2}, api_key="abc"
            )

        self.assertEqual(status, 200)
        self.assertEqual(body, {"ok": True})
        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(json.loads(request.data), {"x": 2})
        self.assertEqual(request.get_header("Authorization"), "Bearer abc")

    def test_http_json_returns_structured_http_error_body(self):
        error_body = b'{"message": "denied"}'
        error = urllib.error.HTTPError(
            "http://example.test", 403, "Forbidden", {}, io.BytesIO(error_body)
        )
        with mock.patch.object(benchmark.urllib.request, "urlopen", side_effect=error):
            status, body = benchmark.http_json("http://example.test")

        self.assertEqual(status, 403)
        self.assertEqual(body, {"message": "denied"})

    def test_list_models_ignores_malformed_entries_and_rejects_http_errors(self):
        with mock.patch.object(
            benchmark,
            "http_json",
            return_value=(200, {"data": [{"id": "one"}, None, {"name": "no-id"}, {"id": 7}]}),
        ):
            self.assertEqual(benchmark.list_models("http://localhost/", None), ["one", "7"])

        with mock.patch.object(benchmark, "http_json", return_value=(503, {"error": "down"})):
            with self.assertRaisesRegex(RuntimeError, "HTTP 503"):
                benchmark.list_models("http://localhost", None)


class CaseTests(unittest.TestCase):
    def test_load_cases_requires_a_json_array(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text('{"id": "not-an-array"}', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "JSON array"):
                benchmark.load_cases(path)

    def test_materialize_generator_is_deterministic_and_does_not_mutate_source(self):
        source = {
            "id": "needle-test",
            "generator": {"type": "needle_haystack", "needle": "ORCHID-42", "approx_tokens": 30},
        }

        first = benchmark.materialize_case(source)
        second = benchmark.materialize_case(source)

        self.assertEqual(first["messages"], second["messages"])
        self.assertIn("ORCHID-42", first["messages"][1]["content"])
        self.assertNotIn("messages", source)

    def test_materialize_rejects_unknown_generator_type(self):
        with self.assertRaisesRegex(ValueError, "Unsupported generator type"):
            benchmark.materialize_case({"generator": {"type": "unknown"}})

    def test_materialize_synthetic_image_grid_builds_image_content_part(self):
        case = {
            "id": "vqa",
            "generator": {
                "type": "synthetic_image_grid",
                "grid_size": 2,
                "cell_px": 4,
                "highlight_row": 0,
                "highlight_col": 1,
                "question": "where?",
            },
        }
        materialized = benchmark.materialize_case(case)
        content = materialized["messages"][1]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "where?"})
        self.assertEqual(content[1]["type"], "image_url")
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_materialize_synthetic_image_sequence_builds_one_part_per_frame(self):
        case = {
            "id": "seq",
            "generator": {
                "type": "synthetic_image_sequence",
                "grid_size": 2,
                "cell_px": 4,
                "frame_positions": [[0, 0], [1, 1]],
                "question": "track it",
            },
        }
        materialized = benchmark.materialize_case(case)
        content = materialized["messages"][1]["content"]
        image_parts = [c for c in content if c.get("type") == "image_url"]
        self.assertEqual(len(image_parts), 2)

    def test_materialize_synthetic_audio_tones_builds_audio_content_part(self):
        case = {
            "id": "tones",
            "generator": {"type": "synthetic_audio_tones", "tone_count": 2, "question": "how many?"},
        }
        materialized = benchmark.materialize_case(case)
        content = materialized["messages"][1]["content"]
        self.assertEqual(content[1]["type"], "input_audio")
        self.assertEqual(content[1]["input_audio"]["format"], "wav")

    def test_materialize_local_asset_image_marks_case_missing_when_file_absent(self):
        case = {
            "id": "real_vqa",
            "generator": {
                "type": "local_asset_image",
                "default_path": "assets/does-not-exist.jpg",
                "question": "what is it?",
            },
        }
        materialized = benchmark.materialize_case(case)
        self.assertTrue(materialized.get("_asset_missing"))
        self.assertNotIn("messages", materialized)

    def test_materialize_local_asset_image_reads_file_when_present(self):
        with tempfile.TemporaryDirectory() as directory:
            asset_path = Path(directory) / "cat.jpg"
            asset_path.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg-bytes")
            case = {
                "id": "real_vqa",
                "generator": {
                    "type": "local_asset_image",
                    "default_path": str(asset_path),
                    "question": "what is it?",
                },
            }
            materialized = benchmark.materialize_case(case)
        self.assertFalse(materialized.get("_asset_missing"))
        content = materialized["messages"][1]["content"]
        self.assertEqual(content[1]["type"], "image_url")

    def test_materialize_local_asset_honors_path_env_override(self):
        with tempfile.TemporaryDirectory() as directory:
            asset_path = Path(directory) / "override.wav"
            asset_path.write_bytes(b"RIFFfake-wav-bytes")
            case = {
                "id": "real_audio",
                "generator": {
                    "type": "local_asset_audio",
                    "default_path": "assets/missing.wav",
                    "path_env": "MY_AUDIO_PATH",
                    "question": "transcribe",
                },
            }
            materialized = benchmark.materialize_case(case, {"MY_AUDIO_PATH": str(asset_path)})
        self.assertFalse(materialized.get("_asset_missing"))

    def test_required_modalities_defaults_to_text(self):
        self.assertEqual(benchmark.required_modalities({}), {"text"})
        self.assertEqual(benchmark.required_modalities({"modalities": ["text", "image"]}), {"text", "image"})

    def test_case_supported_by_capabilities(self):
        case = {"modalities": ["text", "image"]}
        self.assertTrue(benchmark.case_supported_by_capabilities(case, {"text", "image", "audio"}))
        self.assertFalse(benchmark.case_supported_by_capabilities(case, {"text"}))


class StreamingTests(unittest.TestCase):
    def test_stream_chat_combines_sse_text_usage_and_nested_timings(self):
        events = b"".join(
            (
                b": keep-alive\n",
                b"data: not-json\n",
                b'data: {"choices":[{"delta":{"content":"hello "}}],"timings":{"prompt_per_second":12.5}}\n',
                b'data: {"choices":[{"delta":{"content":"world"}}]}\n',
                b'data: {"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":2}}\n',
                b"data: [DONE]\n",
            )
        )
        response = FakeResponse(events)
        with (
            mock.patch.object(benchmark.urllib.request, "urlopen", return_value=response),
            mock.patch.object(benchmark.time, "perf_counter", side_effect=[10.0, 10.25, 10.75]),
        ):
            result = benchmark.stream_chat(
                "http://example.test/", "key", {"model": "m", "messages": []}, timeout=15
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["text"], "hello world")
        self.assertEqual(result["usage"]["completion_tokens"], 2)
        self.assertEqual(result["server_timings"]["prompt_per_second"], 12.5)
        self.assertAlmostEqual(result["ttft_s"], 0.25)
        self.assertAlmostEqual(result["total_s"], 0.75)

    def test_stream_chat_returns_http_error_details(self):
        response_body = b'{"error": "unavailable"}'
        error = urllib.error.HTTPError(
            "http://example.test", 429, "Too Many Requests", {}, io.BytesIO(response_body)
        )
        with mock.patch.object(benchmark.urllib.request, "urlopen", side_effect=error):
            result = benchmark.stream_chat("http://example.test", None, {}, timeout=1)

        self.assertFalse(result["ok"])
        self.assertEqual(result["http_status"], 429)
        self.assertEqual(result["error"], {"error": "unavailable"})
        self.assertIsNone(result["ttft_s"])

    def test_stream_chat_converts_transport_exceptions_to_failed_outcomes(self):
        with mock.patch.object(
            benchmark.urllib.request, "urlopen", side_effect=TimeoutError("slow")
        ):
            result = benchmark.stream_chat("http://example.test", None, {}, timeout=1)

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["type"], "TimeoutError")
        self.assertEqual(result["error"]["message"], "slow")


class ResultTests(unittest.TestCase):
    def test_auto_score_supports_all_validator_kinds(self):
        self.assertEqual(
            benchmark.auto_score({"validator": {"type": "contains", "text": "OK"}}, "looks ok"),
            1.0,
        )
        self.assertEqual(
            benchmark.auto_score({"validator": {"type": "exact", "text": "yes"}}, " yes\n"),
            1.0,
        )
        self.assertEqual(
            benchmark.auto_score(
                {"validator": {"type": "contains_all", "texts": ["one", "two"]}},
                "ONE and TWO",
            ),
            1.0,
        )
        self.assertEqual(
            benchmark.auto_score(
                {"validator": {"type": "contains_any", "texts": ["one", "two"]}},
                "only TWO here",
            ),
            1.0,
        )
        self.assertEqual(
            benchmark.auto_score(
                {"validator": {"type": "contains_any", "texts": ["one", "two"]}},
                "neither here",
            ),
            0.0,
        )
        self.assertIsNone(benchmark.auto_score({"validator": {"type": "unknown"}}, "text"))
        self.assertIsNone(benchmark.auto_score({}, "text"))

    def test_append_jsonl_and_build_summary_exclude_warmups_and_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            results_path = Path(directory) / "nested" / "results.jsonl"
            summary_path = Path(directory) / "summary.csv"
            base = {
                "model": "model-a",
                "case": "case-a",
                "config_id": "baseline",
                "ok": True,
                "warmup": False,
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
                "server_timings": {"prompt_per_second": 20},
                "auto_score": 1.0,
                "timestamp": "2026-01-01T00:00:00+00:00",
            }
            benchmark.append_jsonl(results_path, {**base, "ttft_s": 0.2, "total_s": 1.0})
            benchmark.append_jsonl(
                results_path,
                {**base, "ttft_s": 0.4, "total_s": 2.0, "warmup": True},
            )
            benchmark.append_jsonl(
                results_path,
                {**base, "ttft_s": 9.0, "total_s": 10.0, "ok": False},
            )
            benchmark.append_jsonl(results_path, {**base, "ttft_s": 0.6, "total_s": 3.0})
            with results_path.open("a", encoding="utf-8") as output:
                output.write("not-json\n")

            benchmark.build_summary(results_path, summary_path)

            with summary_path.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["runs"], "2")
            self.assertEqual(float(rows[0]["median_ttft_s"]), 0.4)
            self.assertEqual(float(rows[0]["median_total_s"]), 2.0)
            self.assertEqual(float(rows[0]["median_server_prompt_tps"]), 20.0)


class MainTests(unittest.TestCase):
    def test_parser_uses_environment_configuration_as_defaults(self):
        parser = benchmark.make_parser(
            {
                "LLAMA_SWAP_BASE_URL": "http://configured.test:9000",
                "LLAMA_SWAP_API_KEY": "configured-key",
                "BENCHMARK_RUNS": "4",
            }
        )

        args = parser.parse_args(["--model", "model-a"])

        self.assertEqual(args.base_url, "http://configured.test:9000")
        self.assertEqual(args.api_key, "configured-key")
        self.assertEqual(args.runs, 4)

    def test_parser_uses_models_alias_when_set(self):
        parser = benchmark.make_parser({"MODELS": "model-a,model-b"})
        args = parser.parse_args([])

        self.assertEqual(args.models, ["model-a", "model-b"])

    def test_main_writes_warmup_and_measured_rows_without_live_http(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases_file = root / "cases.json"
            results_file = root / "results.jsonl"
            summary_file = root / "summary.csv"
            cases_file.write_text(
                json.dumps(
                    [
                        {
                            "id": "case-a",
                            "messages": [{"role": "user", "content": "question"}],
                            "max_tokens": 8,
                            "validator": {"type": "contains", "text": "answer"},
                        }
                    ]
                ),
                encoding="utf-8",
            )
            streamed = {
                "ok": True,
                "http_status": 200,
                "error": None,
                "ttft_s": 0.1,
                "total_s": 0.5,
                "text": "answer",
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                "server_timings": {},
            }
            argv = [
                "benchmark_llama_swap.py",
                "--base-url",
                "http://benchmark.test",
                "--model",
                "model-a",
                "--cases-file",
                str(cases_file),
                "--results",
                str(results_file),
                "--summary",
                str(summary_file),
                "--runs",
                "2",
                "--warmups",
                "1",
                "--skip-model-check",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(benchmark, "host_metadata", return_value={"hostname": "test"}),
                mock.patch.object(benchmark, "stream_chat", return_value=streamed) as stream_chat,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = benchmark.main()

            self.assertEqual(result, 0)
            self.assertEqual(stream_chat.call_count, 3)
            records = [json.loads(line) for line in results_file.read_text().splitlines()]
            self.assertEqual([row["warmup"] for row in records], [True, False, False])
            self.assertEqual(records[1]["auto_score"], 1.0)
            self.assertEqual(records[1]["decode_wall_tps"], 5.0)
            with summary_file.open(newline="", encoding="utf-8") as source:
                summary_rows = list(csv.DictReader(source))
            self.assertEqual(summary_rows[0]["runs"], "2")


if __name__ == "__main__":
    unittest.main()