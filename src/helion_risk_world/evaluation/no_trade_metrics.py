from __future__ import annotations

from typing import Any

import numpy as np


def compute(*args: Any, **kwargs: Any) -> dict[str, float]:
    """No-trade quality: avoided losses, regret of declined winners (SPEC.md §22)."""
    decisions = np.asarray(kwargs.get("actions", []))
    realized = np.asarray(kwargs.get("realized_returns", []), dtype=float)
    no_trade = decisions == "no_trade"
    if decisions.size == 0:
        return {"no_trade_rate": 0.0, "avoided_loss_rate": 0.0, "missed_winner_regret": 0.0}
    avoided = no_trade & (realized < 0)
    missed = no_trade & (realized > 0)
    return {
        "no_trade_rate": float(no_trade.mean()),
        "avoided_loss_rate": float(avoided.sum() / max(no_trade.sum(), 1)),
        "missed_winner_regret": float(realized[missed].mean()) if missed.any() else 0.0,
    }
