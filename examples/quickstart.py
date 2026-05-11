"""Quickstart example.

Trains the classifier on the bundled data, routes a few prompts, and prints
the savings vs always-Opus. Uses MockClient so it runs without an API key.

Run from the repo root:

    PYTHONPATH=src python3 examples/quickstart.py
"""

from pathlib import Path

from llm_router.classifier import ComplexityClassifier
from llm_router.clients import MockClient
from llm_router.data import load_jsonl, split_prompts_labels
from llm_router.router import Router


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    rows = load_jsonl(repo_root / "data" / "training.jsonl")
    prompts, labels = split_prompts_labels(rows)

    clf = ComplexityClassifier()
    train_result = clf.fit(prompts, labels)
    print(f"trained classifier, in-sample accuracy={train_result.accuracy:.3f}")

    router = Router(classifier=clf, client=MockClient())

    examples = [
        "What is the capital of Norway?",
        "Write a Python function that returns the longest palindromic substring",
        "Design a global key-value store with strong consistency, walking through the consensus protocol, replication strategy, and failure modes",
    ]
    for prompt in examples:
        decision = router.route_and_complete(prompt)
        print(
            f"-> [{decision.predicted_complexity}] "
            f"{decision.chosen_model} "
            f"(confidence={decision.confidence:.2f})"
        )

    s = router.savings_summary()
    print()
    print(
        f"router cost: ${s['router_cost_usd']:.4f}, "
        f"baseline (always-Opus) cost: ${s['baseline_cost_usd']:.4f}"
    )
    print(
        f"savings: ${s['savings_usd']:.4f} "
        f"({100 * s['savings_fraction']:.1f}%) over {s['calls']} calls"
    )


if __name__ == "__main__":
    main()
