"""CLI for training, evaluating, and benchmarking the router."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .bench import run_benchmark
from .classifier import ComplexityClassifier
from .clients import MockClient
from .data import load_jsonl, split_prompts_labels
from .router import Router

DEFAULT_TRAINING = "data/training.jsonl"
DEFAULT_EVAL = "data/eval.jsonl"
DEFAULT_MODEL = "data/model.json"


def cmd_train(args) -> int:
    rows = load_jsonl(args.data)
    prompts, labels = split_prompts_labels(rows)
    clf = ComplexityClassifier()
    result = clf.fit(prompts, labels)
    clf.save(args.out)
    print(f"trained on {len(prompts)} examples")
    print(f"in-sample accuracy: {result.accuracy:.3f}")
    for cls, m in result.per_class.items():
        print(f"  {cls:>8s}  precision={m['precision']:.3f}  recall={m['recall']:.3f}  f1={m['f1']:.3f}  n={result.class_counts[cls]}")
    print(f"saved to {args.out}")
    return 0


def cmd_evaluate(args) -> int:
    clf = ComplexityClassifier.load(args.model)
    rows = load_jsonl(args.data)
    prompts, labels = split_prompts_labels(rows)
    result = clf.evaluate(prompts, labels)
    print(f"evaluated on {len(prompts)} examples")
    print(f"accuracy: {result.accuracy:.3f}")
    for cls, m in result.per_class.items():
        print(f"  {cls:>8s}  precision={m['precision']:.3f}  recall={m['recall']:.3f}  f1={m['f1']:.3f}  n={result.class_counts[cls]}")
    return 0


def cmd_decide(args) -> int:
    clf = ComplexityClassifier.load(args.model)
    router = Router(classifier=clf, client=MockClient())
    decision = router.decide(args.prompt)
    print(f"predicted complexity: {decision.predicted_complexity}")
    print(f"chosen model:         {decision.chosen_model}")
    print(f"confidence:           {decision.confidence:.3f}")
    return 0


def cmd_bench(args) -> int:
    clf = ComplexityClassifier.load(args.model)
    router = Router(classifier=clf, client=MockClient())
    rows = load_jsonl(args.data)
    prompts, labels = split_prompts_labels(rows)
    result = run_benchmark(router, prompts, labels)
    print(f"prompts:           {result.total_prompts}")
    print(f"routing accuracy:  {result.routing_accuracy:.3f}")
    print(f"router cost:       ${result.router_cost_usd:.4f}")
    print(f"baseline cost:     ${result.baseline_cost_usd:.4f}")
    print(f"savings:           ${result.savings_usd:.4f} ({100*result.savings_fraction:.1f}%)")
    print()
    print("by model:")
    for model, stats in result.by_model.items():
        print(f"  {model}: {stats['calls']} calls, ${stats['cost_usd']:.4f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    from . import __version__

    p = argparse.ArgumentParser(prog="llm-router", description="Route prompts to the cheapest viable model.")
    p.add_argument("--version", action="version", version=f"llm-router {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="train the complexity classifier")
    p_train.add_argument("--data", default=DEFAULT_TRAINING, help="training jsonl (prompt, label per line)")
    p_train.add_argument("--out", default=DEFAULT_MODEL, help="path to write the trained model")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="evaluate a trained model on a held-out set")
    p_eval.add_argument("--model", default=DEFAULT_MODEL)
    p_eval.add_argument("--data", default=DEFAULT_EVAL)
    p_eval.set_defaults(func=cmd_evaluate)

    p_decide = sub.add_parser("decide", help="show the routing decision for one prompt")
    p_decide.add_argument("prompt")
    p_decide.add_argument("--model", default=DEFAULT_MODEL)
    p_decide.set_defaults(func=cmd_decide)

    p_bench = sub.add_parser("bench", help="benchmark router vs always-baseline on a labeled set")
    p_bench.add_argument("--model", default=DEFAULT_MODEL)
    p_bench.add_argument("--data", default=DEFAULT_EVAL)
    p_bench.set_defaults(func=cmd_bench)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
