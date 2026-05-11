"""Complexity classifier: predict simple/medium/complex from a prompt.

The classifier is a small logistic regression trained on a hand-labeled
dataset shipped under data/. Keeping it small means:

  - Inference is microseconds, not milliseconds
  - The model is interpretable (you can dump the coefficients)
  - The whole training set + model fits in a git repo
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .features import extract_features, FEATURE_NAMES

CLASSES = ("simple", "medium", "complex")


@dataclass
class TrainResult:
    accuracy: float
    per_class: Dict[str, Dict[str, float]]
    class_counts: Dict[str, int]


class ComplexityClassifier:
    """Three-class logistic regression over hand-built numeric features."""

    def __init__(self, model=None, scaler=None):
        self._model = model
        self._scaler = scaler

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def fit(self, prompts: List[str], labels: List[str]) -> TrainResult:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        if len(prompts) != len(labels):
            raise ValueError("prompts and labels must be the same length")
        if len(prompts) < 6:
            raise ValueError("need at least 6 training examples")
        unknown = [l for l in labels if l not in CLASSES]
        if unknown:
            raise ValueError(f"unknown labels: {sorted(set(unknown))}")

        X = np.array([extract_features(p) for p in prompts], dtype=float)
        y = np.array(labels)

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        # lbfgs handles multinomial multi-class natively; the explicit
        # multi_class kwarg was deprecated in scikit-learn 1.5 and removed in 1.8.
        model = LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            C=1.0,
        )
        model.fit(Xs, y)
        self._model = model
        self._scaler = scaler

        preds = model.predict(Xs)
        return _summarize_training(y, preds)

    def evaluate(self, prompts: List[str], labels: List[str]) -> TrainResult:
        if not self.is_trained:
            raise RuntimeError("classifier is not trained")
        X = np.array([extract_features(p) for p in prompts], dtype=float)
        Xs = self._scaler.transform(X)
        preds = self._model.predict(Xs)
        return _summarize_training(np.array(labels), preds)

    def predict(self, prompt: str) -> str:
        if not self.is_trained:
            raise RuntimeError("classifier is not trained")
        X = np.array([extract_features(prompt)], dtype=float)
        Xs = self._scaler.transform(X)
        return self._model.predict(Xs)[0]

    def predict_proba(self, prompt: str) -> Dict[str, float]:
        if not self.is_trained:
            raise RuntimeError("classifier is not trained")
        X = np.array([extract_features(prompt)], dtype=float)
        Xs = self._scaler.transform(X)
        proba = self._model.predict_proba(Xs)[0]
        classes = list(self._model.classes_)
        return {c: float(p) for c, p in zip(classes, proba)}

    # Persistence: save coefficients + scaler stats as JSON. Avoids pickle
    # so the saved files are inspectable and not version-fragile.
    def save(self, path: str | Path) -> None:
        if not self.is_trained:
            raise RuntimeError("nothing to save: classifier is not trained")
        payload = {
            "format_version": 1,
            "feature_names": FEATURE_NAMES,
            "classes": list(self._model.classes_),
            "coef": self._model.coef_.tolist(),
            "intercept": self._model.intercept_.tolist(),
            "scaler_mean": self._scaler.mean_.tolist(),
            "scaler_scale": self._scaler.scale_.tolist(),
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ComplexityClassifier":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        payload = json.loads(Path(path).read_text())
        if payload.get("format_version") != 1:
            raise ValueError("unsupported model file format version")

        coef = np.array(payload["coef"])
        intercept = np.array(payload["intercept"])
        classes = np.array(payload["classes"])

        model = LogisticRegression()
        model.coef_ = coef
        model.intercept_ = intercept
        model.classes_ = classes
        # Required attributes for predict() to work without re-fitting.
        model.n_features_in_ = coef.shape[1]

        scaler = StandardScaler()
        scaler.mean_ = np.array(payload["scaler_mean"])
        scaler.scale_ = np.array(payload["scaler_scale"])
        scaler.n_features_in_ = scaler.mean_.shape[0]

        return cls(model=model, scaler=scaler)


def _summarize_training(y_true: np.ndarray, y_pred: np.ndarray) -> TrainResult:
    correct = (y_true == y_pred).sum()
    total = len(y_true)
    accuracy = float(correct / total) if total else 0.0
    per_class: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}
    for cls in CLASSES:
        mask_true = y_true == cls
        mask_pred = y_pred == cls
        tp = int((mask_true & mask_pred).sum())
        fp = int((~mask_true & mask_pred).sum())
        fn = int((mask_true & ~mask_pred).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        counts[cls] = int(mask_true.sum())
    return TrainResult(accuracy=accuracy, per_class=per_class, class_counts=counts)
