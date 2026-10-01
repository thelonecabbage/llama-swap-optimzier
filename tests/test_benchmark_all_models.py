import contextlib
import io
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import benchmark_all_models as sweep
from benchmark_all_models import (
    BenchmarkConfigError,
    build_benchmark_command,
    build_direct_server_args,
    plan_suites,
    resolve_model_defaults,
)


class ModelDefaultTests(unittest.TestCase):
    def test_model_defaults_are_loaded_from_environment_json(self):
        defaults = resolve_model_defaults(
            "text-model",
            {
                "MODEL_DEFAULTS_JSON": json.dumps(
                    {
                        "text-model": {
                            "MODEL_PATH": "/models/text-model.gguf",
                            "GPU_LAYERS": "12",
                            "CACHE_TYPE_K": "q8_0",
                        }
                    }
                )
            },
        )

        self.assertEqual(defaults["MODEL_PATH"], "/models/text-model.gguf")
        self.assertEqual(defaults["GPU_LAYERS"], "12")
        self.assertEqual(defaults["CACHE_TYPE_K"], "q8_0")

    def test_unknown_model_requires_an_explicit_model_path(self):
        with self.assertRaisesRegex(BenchmarkConfigError, "MODEL_PATH"):
            resolve_model_defaults("custom-model", {})

    def test_environment_overrides_model_defaults(self):
        defaults = resolve_model_defaults(
            "custom-model", {"MODEL_PATH": "/custom/model.gguf", "GPU_LAYERS": "12"}
        )

        self.assertEqual(defaults["MODEL_PATH"], "/custom/model.gguf")
        self.assertEqual(defaults["GPU_LAYERS"], "12")


class DirectArgumentTests(unittest.TestCase):
    SERVER_HELP = """--alias --jinja --mmproj --device --no-op-offload
--cache-type-k --cache-type-v --flash-attn --reasoning --reasoning-effort
--spec-type --spec-draft-n-max --spec-draft-n-min --lazy-load --fit
--fit-target-mib --repack --no-repack --no-kv-offload --no-mmap --mlock
--cpu-moe --n-cpu-moe --n-cpu-ffn --spec-draft-model --reasoning-budget
--cache-ram --cpu-strict --cpu-strict-batch --poll --poll-batch
"""

    def test_direct_args_apply_overrides_mtp_and_first_supported_alias(self):
        args = build_direct_server_args(
            "text-model",
            {
                "MODEL_PATH": "/models/text-model.gguf",
                "DIRECT_HOST": "127.0.0.1",
                "CTX_SIZE": "16384",
                "GPU_LAYERS": "0",
                "MTP_N": "4",
                "LAZY_MODE": "resident",
                "DIRECT_EXTRA_ARGS": "--new-option value",
            },
            self.SERVER_HELP,
            port="18081",
        )

        self.assertIn("/models/text-model.gguf", args)
        self.assertEqual(args[args.index("--host") + 1], "127.0.0.1")
        self.assertIn("16384", args)
        self.assertIn("18081", args)
        self.assertIn("--device", args)
        self.assertIn("none", args)
        self.assertIn("--lazy-load", args)
        self.assertNotIn("--lazy", args)
        self.assertEqual(args[args.index("--spec-type") + 1], "draft-mtp")
        self.assertEqual(args[args.index("--spec-draft-n-max") + 1], "4")
        self.assertEqual(args[-2:], ["--new-option", "value"])

    def test_missing_requested_capability_fails_before_server_start(self):
        with self.assertRaisesRegex(BenchmarkConfigError, "CACHE_TYPE_K"):
            build_direct_server_args(
                "text-model",
                {
                    "MODEL_PATH": "/models/text-model.gguf",
                    "CACHE_TYPE_K": "q8_0",
                },
                "--jinja --device --no-op-offload --cache-type-v --flash-attn "
                "--reasoning --reasoning-effort",
            )

    def test_mtp_conflicts_with_another_speculative_mode(self):
        with self.assertRaisesRegex(BenchmarkConfigError, "conflicts"):
            build_direct_server_args(
                "text-model",
                {
                    "MODEL_PATH": "/models/text-model.gguf",
                    "MTP_N": "2",
                    "SPEC_TYPE": "draft-simple",
                },
                self.SERVER_HELP,
            )


class SuitePlanningTests(unittest.TestCase):
    def test_default_plan_has_core_short_long_and_vl_suites(self):
        suites = plan_suites({"CORE_MODELS": "text-a,text-b,text-c", "VL_MODELS": "vision-a"})

        self.assertEqual(len(suites), 7)
        self.assertEqual(suites[0].name, "short-core")
        self.assertEqual(suites[0].cases, ("quick_chat", "it_troubleshooting", "coding_bug", "rag_grounding"))
        self.assertEqual(suites[1].name, "long-context")
        self.assertEqual(suites[-1].name, "vl-text-subset")

    def test_selected_model_and_custom_cases_use_short_run_count(self):
        suites = plan_suites(
            {
                "MODEL": "custom-model",
                "MODEL_PATH": "/models/custom.gguf",
                "BENCH_CASES": "quick_chat, rag_grounding",
                "SHORT_RUNS": "5",
            }
        )

        self.assertEqual(len(suites), 1)
        self.assertEqual(suites[0].model, "custom-model")
        self.assertEqual(suites[0].name, "custom")
        self.assertEqual(suites[0].runs, 5)
        self.assertEqual(suites[0].cases, ("quick_chat", "rag_grounding"))

    def test_models_alias_is_the_unified_selection_list(self):
        suites = plan_suites({"MODELS": "text-a,text-b", "CORE_MODELS": "legacy-core"})

        self.assertEqual(len(suites), 4)
        self.assertEqual([suite.model for suite in suites], ["text-a", "text-a", "text-b", "text-b"])
        self.assertEqual([suite.name for suite in suites[:2]], ["short-core", "long-context"])

    def test_benchmark_command_preserves_repeated_case_arguments_and_direct_flags(self):
        suite = plan_suites({"MODEL": "text-model"})[0]
        command = build_benchmark_command(
            suite,
            benchmark=Path("benchmark_llama_swap.py"),
            cases_file=Path("benchmark_cases.json"),
            results=Path("benchmark_results.jsonl"),
            summary=Path("benchmark_summary.csv"),
            base_url="http://127.0.0.1:18080",
            config_id="threads-8",
            warmups=0,
            timeout=120,
            started_at="2026-10-01T12:00:00+00:00",
            direct=True,
            direct_args="/app/llama-server --threads 8",
        )

        self.assertEqual(command[command.index("--model") + 1], "text-model")
        self.assertEqual(command[command.index("--config-id") + 1], "threads-8")
        self.assertIn("--skip-model-check", command)
        self.assertEqual(command[command.index("--api-key") + 1], "")
        self.assertEqual(command.count("--case"), len(suite.cases))
        self.assertEqual(command[command.index("--base-url") + 1], "http://127.0.0.1:18080")


class SweepExecutionTests(unittest.TestCase):
    def test_swap_sweep_continues_after_a_suite_failure(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            bench = directory / "benchmark.py"
            cases = directory / "cases.json"
            bench.touch()
            cases.touch()
            log_dir = directory / "logs"

            with (
                mock.patch.object(sweep, "BENCH", bench),
                mock.patch.object(sweep, "CASES", cases),
                mock.patch.object(sweep, "RESULTS", directory / "results.jsonl"),
                mock.patch.object(sweep, "SUMMARY", directory / "summary.csv"),
                mock.patch.object(sweep, "LOG_DIR", log_dir),
                mock.patch.object(sweep.shutil, "which", return_value="/usr/bin/python3"),
                mock.patch.object(sweep, "discover_models", return_value=["text-model"]),
                mock.patch.object(sweep, "stream_process", side_effect=[7, 0]) as run_process,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = sweep.run_sweep(
                    {
                        "MODEL": "text-model",
                        "LLAMA_SWAP_BASE_URL": "http://benchmark.test",
                        "WARMUPS": "0",
                    }
                )

            self.assertEqual(result, 2)
            self.assertEqual(run_process.call_count, 2)
            short_command = run_process.call_args_list[0].args[0]
            long_command = run_process.call_args_list[1].args[0]
            self.assertIn("quick_chat", short_command)
            self.assertIn("long_context_8k", long_command)
            self.assertTrue(log_dir.is_dir())

    def test_swap_sweep_uses_local_mock_server_when_endpoint_is_unreachable(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            bench = directory / "benchmark.py"
            cases = directory / "cases.json"
            bench.touch()
            cases.touch()
            log_dir = directory / "logs"

            with (
                mock.patch.object(sweep, "BENCH", bench),
                mock.patch.object(sweep, "CASES", cases),
                mock.patch.object(sweep, "RESULTS", directory / "results.jsonl"),
                mock.patch.object(sweep, "SUMMARY", directory / "summary.csv"),
                mock.patch.object(sweep, "LOG_DIR", log_dir),
                mock.patch.object(sweep.shutil, "which", return_value="/usr/bin/python3"),
                mock.patch.object(sweep, "discover_models", side_effect=sweep.SweepError("connect failed")),
                mock.patch.object(
                    sweep,
                    "start_local_mock_server",
                    return_value=("http://127.0.0.1:8080", {"text-model"}, mock.Mock()),
                ) as fallback,
                mock.patch.object(sweep, "stream_process", return_value=0) as run_process,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = sweep.run_sweep({
                    "MODEL": "text-model",
                    "LLAMA_SWAP_BASE_URL": "http://localhost:8080",
                    "WARMUPS": "0",
                })

            self.assertEqual(result, 0)
            fallback.assert_called_once()
            self.assertEqual(run_process.call_count, 2)
            self.assertIn("quick_chat", run_process.call_args_list[0].args[0])
            self.assertIn("long_context_8k", run_process.call_args_list[1].args[0])

    def test_direct_sweep_requires_model_before_docker_actions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            bench = directory / "benchmark.py"
            cases = directory / "cases.json"
            bench.touch()
            cases.touch()
            with (
                mock.patch.object(sweep, "BENCH", bench),
                mock.patch.object(sweep, "CASES", cases),
                mock.patch.object(sweep, "LOG_DIR", directory / "logs"),
                mock.patch.object(sweep.shutil, "which", return_value="/usr/bin/python3"),
                mock.patch.object(sweep, "DirectServer") as direct_server,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                with self.assertRaisesRegex(sweep.SweepError, "requires MODEL"):
                    sweep.run_sweep({"DIRECT_MODE": "1"})
            direct_server.assert_not_called()


if __name__ == "__main__":
    unittest.main()