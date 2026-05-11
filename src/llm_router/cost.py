"""Token and dollar accounting.

The pricing values below are reasonable placeholders; check Anthropic's
current pricing page before quoting these to anyone. The whole table is
overridable via Router(model_pricing=...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class ModelPricing:
    input_per_mtok_usd: float
    output_per_mtok_usd: float


# Defaults reflect public list prices as of mid-2026. Override when needed.
MODEL_PRICING: Dict[str, ModelPricing] = {
    "claude-haiku-4-5": ModelPricing(input_per_mtok_usd=0.80, output_per_mtok_usd=4.00),
    "claude-sonnet-4-6": ModelPricing(input_per_mtok_usd=3.00, output_per_mtok_usd=15.00),
    "claude-opus-4-7": ModelPricing(input_per_mtok_usd=15.00, output_per_mtok_usd=75.00),
}


@dataclass
class CostRecord:
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class CostLedger:
    records: list = field(default_factory=list)

    def add(self, model: str, input_tokens: int, output_tokens: int, pricing: Dict[str, ModelPricing] = None) -> CostRecord:
        cost = estimate_cost(model, input_tokens, output_tokens, pricing)
        record = CostRecord(model, input_tokens, output_tokens, cost)
        self.records.append(record)
        return record

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.records)

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.records)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.records)

    def by_model(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for r in self.records:
            slot = out.setdefault(
                r.model, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
            )
            slot["calls"] += 1
            slot["input_tokens"] += r.input_tokens
            slot["output_tokens"] += r.output_tokens
            slot["cost_usd"] += r.cost_usd
        return out


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    pricing: Dict[str, ModelPricing] = None,
) -> float:
    """Return the dollar cost of a single completion."""
    table = pricing if pricing is not None else MODEL_PRICING
    if model not in table:
        raise KeyError(f"no pricing for model '{model}'; pass pricing= explicitly")
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("token counts must be non-negative")
    p = table[model]
    return (input_tokens / 1_000_000.0) * p.input_per_mtok_usd + (
        output_tokens / 1_000_000.0
    ) * p.output_per_mtok_usd
