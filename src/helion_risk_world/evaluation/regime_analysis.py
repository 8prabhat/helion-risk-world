from __future__ import annotations

from typing import Any

import numpy as np


def compute(*args: Any, **kwargs: Any) -> dict[str, float]:
    """Regime-wise performance breakdown."""
    regimes = kwargs.get("regimes", [])
    values = np.asarray(kwargs.get("values", []), dtype=float)
    if not regimes:
        return {}
    out: dict[str, float] = {}
    for regime in sorted(set(regimes)):
        mask = np.asarray([r == regime for r in regimes])
        out[f"{regime}_mean"] = float(values[mask].mean()) if mask.any() else 0.0
        out[f"{regime}_count"] = float(mask.sum())
    return out
