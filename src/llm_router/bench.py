"""Benchmark the router against an always-baseline strategy.

Replays a prompt set through both, then reports cost savings and the
accuracy of the routing decisions versus the labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .clients import Client
from .router import Router


@dataclass
class BenchResult:
    total_prompts: int
    correct_decisions: int
    routing_accuracy: float
    router_cost_usd: float
    baseline_cost_usd: float
    savings_usd: float
    savings_fraction: float
    by_model: Dict[str, Dict[str, float]]


def run_benchmark(router: Router, prompts: List[str], labels: List[str]) -> BenchResult:
    if len(prompts) != len(labels):
        raise ValueError("prompts and labels must be the same length")

    correct = 0
    for prompt, label in zip(prompts, labels):
        decision = router.route_and_complete(prompt)
        # A "correct" decision is one where the router did not over-spend
        # (chose a tier no higher than the labeled one) AND did not under-spend
        # (chose a tier no lower than the labeled one). Strict exact-match.
        if decision.predicted_complexity == label:
            correct += 1

    s = router.savings_summary()
    return BenchResult(
        total_prompts=len(prompts),
        correct_decisions=correct,
        routing_accuracy=correct / len(prompts) if prompts else 0.0,
        router_cost_usd=s["router_cost_usd"],
        baseline_cost_usd=s["baseline_cost_usd"],
        savings_usd=s["savings_usd"],
        savings_fraction=s["savings_fraction"],
        by_model=router.ledger.by_model(),
    )
