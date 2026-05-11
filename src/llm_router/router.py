"""Route a prompt to the cheapest model that should handle it well."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from .classifier import ComplexityClassifier
from .clients import Client, CompletionResult
from .cost import CostLedger, MODEL_PRICING, ModelPricing


# Default mapping: predicted complexity class -> model name.
DEFAULT_TIERS: Dict[str, str] = {
    "simple": "claude-haiku-4-5",
    "medium": "claude-sonnet-4-6",
    "complex": "claude-opus-4-7",
}


@dataclass
class RouteDecision:
    predicted_complexity: str
    chosen_model: str
    confidence: float
    completion: Optional[CompletionResult] = None


class Router:
    """Wraps a classifier + a client + a cost ledger."""

    def __init__(
        self,
        classifier: ComplexityClassifier,
        client: Client,
        tiers: Optional[Dict[str, str]] = None,
        pricing: Optional[Dict[str, ModelPricing]] = None,
        baseline_model: str = "claude-opus-4-7",
        confidence_floor: float = 0.0,
        ledger: Optional[CostLedger] = None,
        baseline_ledger: Optional[CostLedger] = None,
    ):
        self.classifier = classifier
        self.client = client
        self.tiers = dict(tiers) if tiers is not None else dict(DEFAULT_TIERS)
        self.pricing = pricing if pricing is not None else MODEL_PRICING
        self.baseline_model = baseline_model
        self.confidence_floor = confidence_floor
        self.ledger = ledger if ledger is not None else CostLedger()
        self.baseline_ledger = baseline_ledger if baseline_ledger is not None else CostLedger()

        # Validate that the tiers map only to models in the pricing table.
        for label, model in self.tiers.items():
            if model not in self.pricing:
                raise KeyError(f"tier '{label}' maps to model '{model}' which has no pricing")
        if baseline_model not in self.pricing:
            raise KeyError(f"baseline_model '{baseline_model}' has no pricing")

    def decide(self, prompt: str) -> RouteDecision:
        """Return what model would be chosen, without sending the prompt."""
        if not self.classifier.is_trained:
            raise RuntimeError("classifier must be trained before routing")
        proba = self.classifier.predict_proba(prompt)
        chosen_class = max(proba, key=proba.get)
        confidence = proba[chosen_class]

        # Confidence floor: if the classifier is uncertain, escalate one tier.
        if confidence < self.confidence_floor:
            chosen_class = _escalate_one_tier(chosen_class)

        chosen_model = self.tiers.get(chosen_class)
        if chosen_model is None:
            raise KeyError(f"no model mapped for class '{chosen_class}'")
        return RouteDecision(
            predicted_complexity=chosen_class,
            chosen_model=chosen_model,
            confidence=confidence,
        )

    def route_and_complete(self, prompt: str, max_tokens: int = 1024) -> RouteDecision:
        """Decide a model, send the prompt, log costs (router and baseline)."""
        decision = self.decide(prompt)
        result = self.client.complete(decision.chosen_model, prompt, max_tokens=max_tokens)
        decision.completion = result
        self.ledger.add(result.model, result.input_tokens, result.output_tokens, self.pricing)
        # Baseline: assume the same input/output token count, charged at baseline pricing.
        self.baseline_ledger.add(
            self.baseline_model, result.input_tokens, result.output_tokens, self.pricing
        )
        return decision

    def savings_summary(self) -> Dict[str, float]:
        """Compare router cost vs baseline cost on the prompts seen so far."""
        router_cost = self.ledger.total_cost_usd
        baseline_cost = self.baseline_ledger.total_cost_usd
        savings = baseline_cost - router_cost
        ratio = (savings / baseline_cost) if baseline_cost else 0.0
        return {
            "router_cost_usd": router_cost,
            "baseline_cost_usd": baseline_cost,
            "savings_usd": savings,
            "savings_fraction": ratio,
            "calls": len(self.ledger.records),
        }


def _escalate_one_tier(label: str) -> str:
    order = ("simple", "medium", "complex")
    if label not in order:
        return label
    idx = order.index(label)
    return order[min(idx + 1, len(order) - 1)]
