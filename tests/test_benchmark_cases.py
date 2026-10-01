import unittest
from pathlib import Path

import benchmark_llama_swap as benchmark


CASES_FILE = Path(__file__).resolve().parents[1] / "benchmark_cases.json"
REQUIRED_CASES = {
    "quick_chat",
    "it_troubleshooting",
    "coding_bug",
    "rag_grounding",
    "long_context_8k",
    "long_context_24k",
}


class BenchmarkCasesTests(unittest.TestCase):
    def test_default_sweep_cases_exist_and_have_valid_prompts(self):
        cases = benchmark.load_cases(CASES_FILE)
        by_id = {case.get("id"): case for case in cases}

        self.assertTrue(REQUIRED_CASES.issubset(by_id))
        for case_id in REQUIRED_CASES:
            with self.subTest(case=case_id):
                case = by_id[case_id]
                materialized = benchmark.materialize_case(case)
                self.assertTrue(materialized.get("messages"))
                self.assertIsInstance(materialized["messages"], list)
                self.assertIn("max_tokens", materialized)

        self.assertEqual(
            by_id["long_context_8k"]["generator"]["approx_tokens"], 8192
        )
        self.assertEqual(
            by_id["long_context_24k"]["generator"]["approx_tokens"], 24576
        )


if __name__ == "__main__":
    unittest.main()