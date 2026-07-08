from __future__ import annotations

from typing import Any

import numpy as np


class MetricRegistry:
    """Local registry (replaces the assumed quanthelion.metrics.MetricRegistry). OCP: register new
    metrics without editing callers (SPEC.md §6.2, §22, §26)."""

    def __init__(self) -> None:
        self._metrics: dict[str, Any] = {}

    def register(self, name: str, fn: Any) -> None:
        if name in self._metrics:
            raise ValueError(f"metric already registered: {name}")
        self._metrics[name] = fn

    def compute(self, name: str, *args: Any, **kwargs: Any) -> Any:
        return self._metrics[name](*args, **kwargs)


def classification_report(
    probs: Any,
    labels: Any,
    *,
    class_names: tuple[str, ...] | None = None,
    n_confidence_bins: int = 5,
) -> dict[str, Any]:
    """Return precision/recall/F1, confusion matrix, and confidence bucket accuracy."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=int).reshape(-1)
    if p.ndim != 2:
        raise ValueError(f"probs must be [N, C]; got shape {p.shape}")
    if p.shape[0] != y.shape[0]:
        raise ValueError("probs and labels must share the same first dimension")
    if y.size == 0:
        return {
            "samples": 0,
            "confusion_matrix": [],
            "per_class": {},
            "macro_f1": 0.0,
            "accuracy": 0.0,
            "confidence_buckets": [],
        }
    n_classes = p.shape[1]
    names = class_names or tuple(str(i) for i in range(n_classes))
    if len(names) != n_classes:
        raise ValueError("class_names length must match probs.shape[1]")
    pred = p.argmax(axis=1)
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for actual, predicted in zip(y, pred, strict=False):
        if 0 <= actual < n_classes:
            confusion[actual, predicted] += 1

    per_class: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for idx, name in enumerate(names):
        tp = float(confusion[idx, idx])
        fp = float(confusion[:, idx].sum() - confusion[idx, idx])
        fn = float(confusion[idx, :].sum() - confusion[idx, idx])
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        support = float(confusion[idx, :].sum())
        f1_values.append(f1)
        per_class[str(name)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    conf = p.max(axis=1)
    correct = pred == y
    bins = np.linspace(0.0, 1.0, n_confidence_bins + 1)
    buckets: list[dict[str, float]] = []
    for lo, hi in zip(bins[:-1], bins[1:], strict=False):
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if not bool(mask.any()):
            continue
        buckets.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "n": float(mask.sum()),
                "mean_confidence": float(conf[mask].mean()),
                "accuracy": float(correct[mask].mean()),
            }
        )

    return {
        "samples": int(y.size),
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "accuracy": float(correct.mean()),
        "confidence_buckets": buckets,
    }


__all__ = ["MetricRegistry", "classification_report"]
