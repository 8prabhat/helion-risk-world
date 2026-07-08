from __future__ import annotations

from typing import Any

import numpy as np


def compute(*args: Any, **kwargs: Any) -> dict[str, float]:
    """Risk: VaR, CVaR, drawdown duration, breach counts, shield interventions."""
    returns = np.asarray(kwargs.get("step_returns", []), dtype=float)
    equity = np.asarray(kwargs.get("equity", []), dtype=float)
    alpha = float(kwargs.get("alpha", 0.05))
    interventions = int(kwargs.get("risk_shield_interventions", 0))
    if returns.size == 0:
        return {
            "var": 0.0,
            "cvar": 0.0,
            "max_drawdown": 0.0,
            "drawdown_duration": 0.0,
            "risk_shield_interventions": float(interventions),
        }
    var = float(-np.quantile(returns, alpha))
    tail = returns[returns <= np.quantile(returns, alpha)]
    cvar = float(-tail.mean()) if tail.size else 0.0
    if equity.size:
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / np.where(peak == 0, 1.0, peak)
        max_dd = float(dd.max())
        duration = 0
        cur = 0
        for val in dd:
            if val > 0:
                cur += 1
                duration = max(duration, cur)
            else:
                cur = 0
    else:
        max_dd = 0.0
        duration = 0
    return {
        "var": var,
        "cvar": cvar,
        "max_drawdown": max_dd,
        "drawdown_duration": float(duration),
        "risk_shield_interventions": float(interventions),
    }
