"""Shared causal barrier-width utilities for labels, training, and runtime."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_EPS = 1e-6
_DEFAULT_SIGMA = 0.01


@dataclass(frozen=True)
class BarrierSpec:
    """Fixed barrier geometry used to turn decision-time sigma into stop/target returns."""

    stop_mult: float = 2.0
    target_mult: float = 2.0
    vol_span: int = 50
    # Minimum barrier half-width as a return fraction (feature/label overhaul Phase 1):
    # without this, a purely vol-scaled barrier can be arbitrarily tiny in low-vol
    # regimes — well below round-trip transaction cost — labeling sub-cost noise as a
    # real directional "target hit"/"stop hit". Use
    # execution.cost_model.round_trip_cost_frac() to derive this from the project's own
    # documented cost assumptions rather than guessing a constant. 0.0 preserves the
    # old purely-vol-scaled behavior.
    cost_floor_frac: float = 0.0
    # Horizon-extension scope (feature/label overhaul Phase 4a): sigma from
    # ewma_barrier_sigma() is a PER-BAR estimate. Under an i.i.d.-returns assumption,
    # cumulative volatility over a holding period scales with sqrt(horizon_bars), so
    # a fixed `mult * sigma` barrier width is only calibrated to horizon_bars=1 bar.
    # Confirmed empirically before this fix: at horizon_bars=12 (60 min), median
    # exit_bars was 5 and only 24% of labels ever reached the H=12 timeout -- barriers
    # were already too tight for even the pre-existing horizon. Left at 1 so any
    # caller not yet passing horizon_bars keeps exact prior behavior (sqrt(1) = 1).
    horizon_bars: int = 1

    def __post_init__(self) -> None:
        if self.stop_mult <= 0.0:
            raise ValueError("stop_mult must be > 0")
        if self.target_mult <= 0.0:
            raise ValueError("target_mult must be > 0")
        if self.vol_span < 1:
            raise ValueError("vol_span must be >= 1")
        if self.cost_floor_frac < 0.0:
            raise ValueError("cost_floor_frac must be >= 0")
        if self.horizon_bars < 1:
            raise ValueError("horizon_bars must be >= 1")


@dataclass(frozen=True)
class BarrierContext:
    """Point-in-time barrier width expressed as sigma and relative stop/target returns."""

    sigma: float
    stop_return: float
    target_return: float

    def __post_init__(self) -> None:
        if self.sigma < 0.0:
            raise ValueError("sigma must be non-negative")
        if self.stop_return > 0.0:
            raise ValueError("stop_return must be <= 0")
        if self.target_return < 0.0:
            raise ValueError("target_return must be >= 0")


def ewma_barrier_sigma(close: np.ndarray, span: int = 50) -> np.ndarray:
    """Decision-time EWMA log-return sigma used by the barrier labeler."""
    arr = np.asarray(close, dtype=float)
    if arr.ndim != 1:
        raise ValueError("close must be a 1D array")
    series = pd.Series(arr)
    lr = np.log(series / series.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    sigma = lr.ewm(span=span, adjust=False).std().fillna(_DEFAULT_SIGMA).to_numpy(dtype=float)
    return np.maximum(sigma, _EPS)


def barrier_context_from_sigma(
    sigma: float,
    *,
    spec: BarrierSpec | None = None,
) -> BarrierContext:
    """Convert a decision-time sigma into explicit stop/target returns."""
    resolved = spec or BarrierSpec()
    sigma_value = max(float(sigma), _EPS)
    horizon_scale = np.sqrt(float(resolved.horizon_bars))
    stop_return = max(resolved.stop_mult * sigma_value * horizon_scale, resolved.cost_floor_frac)
    target_return = max(resolved.target_mult * sigma_value * horizon_scale, resolved.cost_floor_frac)
    return BarrierContext(
        sigma=sigma_value,
        stop_return=-stop_return,
        target_return=target_return,
    )


def barrier_context_series(
    close: np.ndarray,
    *,
    spec: BarrierSpec | None = None,
) -> np.ndarray:
    """Return per-bar ``[sigma, stop_return, target_return]`` barrier context rows."""
    resolved = spec or BarrierSpec()
    sigma = ewma_barrier_sigma(close, span=resolved.vol_span)
    horizon_scale = np.sqrt(float(resolved.horizon_bars))
    stop_return = np.maximum(resolved.stop_mult * sigma * horizon_scale, resolved.cost_floor_frac)
    target_return = np.maximum(resolved.target_mult * sigma * horizon_scale, resolved.cost_floor_frac)
    return np.stack(
        [
            sigma,
            -stop_return,
            target_return,
        ],
        axis=1,
    ).astype(np.float32, copy=False)


__all__ = [
    "BarrierContext",
    "BarrierSpec",
    "barrier_context_from_sigma",
    "barrier_context_series",
    "ewma_barrier_sigma",
]
