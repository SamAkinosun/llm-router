import unittest
from pathlib import Path

from llm_router.bench import run_benchmark
from llm_router.classifier import ComplexityClassifier
from llm_router.clients import MockClient
from llm_router.data import load_jsonl, split_prompts_labels
from llm_router.router import Router

REPO_ROOT = Path(__file__).resolve().parent.parent


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = load_jsonl(REPO_ROOT / "data" / "training.jsonl")
        prompts, labels = split_prompts_labels(rows)
        cls.clf = ComplexityClassifier()
        cls.clf.fit(prompts, labels)

    def test_eval_set_routes_with_meaningful_savings(self):
        rows = load_jsonl(REPO_ROOT / "data" / "eval.jsonl")
        prompts, labels = split_prompts_labels(rows)
        router = Router(classifier=self.clf, client=MockClient())
        result = run_benchmark(router, prompts, labels)

        self.assertEqual(result.total_prompts, len(prompts))
        self.assertGreater(result.routing_accuracy, 0.5)
        self.assertGreater(result.savings_fraction, 0.0)
        # Sanity: router cost should be strictly less than baseline cost.
        self.assertLess(result.router_cost_usd, result.baseline_cost_usd)

    def test_mismatched_lengths_raise(self):
        router = Router(classifier=self.clf, client=MockClient())
        with self.assertRaises(ValueError):
            run_benchmark(router, ["a", "b"], ["simple"])


if __name__ == "__main__":
    unittest.main()
