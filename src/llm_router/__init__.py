"""llm-router: route prompts to the cheapest model that can answer them."""

__version__ = "0.1.0"

from .features import extract_features, FEATURE_NAMES
from .classifier import ComplexityClassifier
from .router import Router, RouteDecision
from .cost import MODEL_PRICING, estimate_cost
from .clients import MockClient

__all__ = [
    "__version__",
    "extract_features",
    "FEATURE_NAMES",
    "ComplexityClassifier",
    "Router",
    "RouteDecision",
    "MODEL_PRICING",
    "estimate_cost",
    "MockClient",
]
