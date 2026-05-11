import unittest

from llm_router.classifier import ComplexityClassifier
from llm_router.clients import MockClient
from llm_router.data import load_jsonl, split_prompts_labels
from llm_router.router import DEFAULT_TIERS, Router

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def trained_classifier():
    rows = load_jsonl(REPO_ROOT / "data" / "training.jsonl")
    prompts, labels = split_prompts_labels(rows)
    clf = ComplexityClassifier()
    clf.fit(prompts, labels)
    return clf


class RouterDecisionTests(unittest.TestCase):
    def test_decide_returns_a_known_model(self):
        clf = trained_classifier()
        router = Router(classifier=clf, client=MockClient())
        decision = router.decide("Hello")
        self.assertIn(decision.chosen_model, DEFAULT_TIERS.values())

    def test_decide_does_not_send_prompt(self):
        clf = trained_classifier()
        client = MockClient()
        router = Router(classifier=clf, client=client)
        router.decide("anything")
        self.assertEqual(client.calls, [])

    def test_route_and_complete_calls_chosen_model(self):
        clf = trained_classifier()
        client = MockClient()
        router = Router(classifier=clf, client=client)
        decision = router.route_and_complete("Hello")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0][0], decision.chosen_model)

    def test_confidence_floor_escalates_one_tier(self):
        clf = trained_classifier()
        client = MockClient()
        # Confidence floor at 1.0 forces escalation on every prediction.
        router = Router(
            classifier=clf, client=client, confidence_floor=1.01
        )  # impossible threshold => always escalate
        decision = router.decide("Hello")
        # Predicted class should be escalated; if base was 'simple', escalated to 'medium'.
        # If base was already 'complex', it stays 'complex'.
        self.assertIn(decision.predicted_complexity, ("medium", "complex"))


class RouterCostTrackingTests(unittest.TestCase):
    def test_savings_summary_tracks_router_vs_baseline(self):
        clf = trained_classifier()
        router = Router(classifier=clf, client=MockClient())
        for prompt in ["What is 2+2?", "What time is it in Tokyo?", "Write a regex for emails"]:
            router.route_and_complete(prompt)
        s = router.savings_summary()
        self.assertEqual(s["calls"], 3)
        self.assertGreater(s["baseline_cost_usd"], 0.0)
        # Baseline is always Opus by default, so savings should be >= 0.
        self.assertGreaterEqual(s["savings_usd"], 0.0)

    def test_invalid_baseline_model_raises(self):
        clf = trained_classifier()
        with self.assertRaises(KeyError):
            Router(classifier=clf, client=MockClient(), baseline_model="nonexistent-model")

    def test_invalid_tier_model_raises(self):
        clf = trained_classifier()
        with self.assertRaises(KeyError):
            Router(
                classifier=clf,
                client=MockClient(),
                tiers={"simple": "nonexistent-model", "medium": "claude-sonnet-4-6", "complex": "claude-opus-4-7"},
            )


if __name__ == "__main__":
    unittest.main()
