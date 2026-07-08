"""Sample uniqueness weights for overlapping triple-barrier labels (SPEC.md §11.2).

Overlapping label windows are NOT independent — consecutive bars whose [t, exit_t]
spans overlap share signal.  Uniqueness ū_i = mean_{τ∈[t_i,exit_i]} 1/c_τ, where
c_τ = number of active labels at bar τ.  Training and CV use weights proportional
to ū_i.  SRP: uniqueness computation only.
"""

from __future__ import annotations

import numpy as np

from helion_risk_world.schemas.label_schema import LabelRecord


def compute_uniqueness(records: list[LabelRecord]) -> np.ndarray:
    """Return ū_i in [0,1] for each label record.

    Complexity O(T * avg_label_length) — acceptable for thousands of bars.
    """
    if not records:
        return np.array([], dtype=float)

    n = len(records)
    # Records are emitted chronologically, one decision bar per record, so the start index is the
    # record position. exit_t is a decision-relative exit offset in bars.
    t_start = np.arange(n, dtype=int)
    t_end = np.array([i + r.exit_t for i, r in enumerate(records)], dtype=int)

    # For bar τ, concurrency c_τ = number of labels with t_start[i] <= τ <= t_end[i]
    max_bar = int(t_end.max()) + 1
    concurrency = np.zeros(max_bar, dtype=float)
    for i in range(n):
        concurrency[t_start[i]: t_end[i] + 1] += 1.0

    uniqueness = np.empty(n, dtype=float)
    for i in range(n):
        bars = concurrency[t_start[i]: t_end[i] + 1]
        uniqueness[i] = float(np.mean(1.0 / np.maximum(bars, 1.0)))

    return uniqueness


def apply_uniqueness_weights(records: list[LabelRecord]) -> list[LabelRecord]:
    """Attach uniqueness_weight ū_i to each record.  Returns a new list (records are immutable)."""
    if not records:
        return []
    weights = compute_uniqueness(records)
    return [r.model_copy(update={"uniqueness_weight": float(w)}) for r, w in zip(records, weights)]


__all__ = ["compute_uniqueness", "apply_uniqueness_weights"]
