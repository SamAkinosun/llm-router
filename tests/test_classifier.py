import os
import tempfile
import unittest
from pathlib import Path

from llm_router.classifier import CLASSES, ComplexityClassifier
from llm_router.data import load_jsonl, split_prompts_labels


REPO_ROOT = Path(__file__).resolve().parent.parent


class ClassifierBasicsTests(unittest.TestCase):
    def test_predict_before_fit_raises(self):
        clf = ComplexityClassifier()
        with self.assertRaises(RuntimeError):
            clf.predict("hello")

    def test_too_few_examples_raises(self):
        clf = ComplexityClassifier()
        with self.assertRaises(ValueError):
            clf.fit(["a", "b", "c"], ["simple", "medium", "complex"])

    def test_unknown_label_raises(self):
        clf = ComplexityClassifier()
        with self.assertRaises(ValueError):
            clf.fit(
                ["a", "b", "c", "d", "e", "f"],
                ["simple", "simple", "medium", "medium", "complex", "BOGUS"],
            )


class ClassifierTrainsAndPredictsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = load_jsonl(REPO_ROOT / "data" / "training.jsonl")
        cls.prompts, cls.labels = split_prompts_labels(rows)
        cls.clf = ComplexityClassifier()
        cls.train_result = cls.clf.fit(cls.prompts, cls.labels)

    def test_in_sample_accuracy_above_threshold(self):
        # In-sample accuracy on a small training set should be very high.
        self.assertGreater(self.train_result.accuracy, 0.85)

    def test_predict_returns_known_class(self):
        for prompt in [
            "What is the capital of Spain?",
            "Write a SQL join across three tables",
            "Design a distributed lock service with linearizable consistency and reason about the failure modes",
        ]:
            pred = self.clf.predict(prompt)
            self.assertIn(pred, CLASSES)

    def test_predict_proba_sums_to_one(self):
        proba = self.clf.predict_proba("anything")
        total = sum(proba.values())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_short_simple_prompt_is_not_classified_as_complex(self):
        pred = self.clf.predict("What is 5 plus 7?")
        self.assertNotEqual(pred, "complex")

    def test_long_reasoning_prompt_is_not_classified_as_simple(self):
        prompt = (
            "Design and analyze a distributed key-value store with linearizable consistency, "
            "covering the consensus protocol, failure detection, recovery semantics, and the "
            "operational implications of running across multiple regions with asymmetric latencies."
        )
        pred = self.clf.predict(prompt)
        self.assertNotEqual(pred, "simple")


class ClassifierPersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = load_jsonl(REPO_ROOT / "data" / "training.jsonl")
        prompts, labels = split_prompts_labels(rows)
        cls.clf = ComplexityClassifier()
        cls.clf.fit(prompts, labels)

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.json")
            self.clf.save(path)
            loaded = ComplexityClassifier.load(path)
            for prompt in ["hi", "write a regex", "design a database"]:
                self.assertEqual(self.clf.predict(prompt), loaded.predict(prompt))

    def test_save_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            ComplexityClassifier().save("/tmp/_unused.json")


if __name__ == "__main__":
    unittest.main()
