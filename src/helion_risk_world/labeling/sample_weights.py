"""Sample weight utilities for uniqueness-weighted training (SPEC.md §11.2).

Converts uniqueness values to training weights usable by DataLoader samplers
and loss weighting.  SRP: weight extraction only.
"""

from __future__ import annotations

import numpy as np

from helion_risk_world.schemas.label_schema import LabelRecord


def extract_sample_weights(
    records: list[LabelRecord],
    normalise: bool = True,
) -> np.ndarray:
    """Return uniqueness_weight as a 1-D float array.

    Raises ``ValueError`` if any record has ``uniqueness_weight is None`` —
    call ``apply_uniqueness_weights`` first.
    """
    if not records:
        return np.array([], dtype=float)

    weights = np.array(
        [r.uniqueness_weight for r in records], dtype=float
    )
    if np.isnan(weights).any():
        raise ValueError(
            "some records have uniqueness_weight=None; call apply_uniqueness_weights first"
        )
    if normalise:
        total = weights.sum()
        if total > 0:
            weights = weights / total
    return weights


__all__ = ["extract_sample_weights"]
