"""Market-feature primitives — hybrid shim (generic math migrated; project regime
labeling stays local).

The generic, model-agnostic primitives (returns/vol/ATR/OI-change/volume-z and the
F=19/Phase-2 stabilized set) are now the reusable quanthelion implementation
(re-exported here, identical behavior). ``regime_label``/``state_regime_label`` return
this project's own ``Regime`` enum and encode a project-specific labeling policy — they
do NOT belong in quanthelion and remain defined here.

Original implementation backed up in .pre_quanthelion_migration_backup/.
"""

from __future__ import annotations

from quanthelion.data.transforms.primitives import (
    SESSION_CLOSE,
    SESSION_MINUTES,
    SESSION_OPEN,
    atr,
    atr_pct,
    bb_position,
    d_oi_pct,
    day_of_week,
    dmi,
    dow_cos,
    dow_sin,
    first_window_return,
    high_low_pos,
    hl_range,
    log_returns,
    momentum_norm,
    oi_change,
    oi_norm,
    open_close_norm,
    opening_range_position,
    realized_vol,
    realized_vol_rs,
    rsi,
    session_boundary_mask,
    session_return,
    simple_returns,
    time_of_day,
    tod_cos,
    tod_sin,
    variance_ratio,
    volume_zscore,
)

from helion_risk_world.schemas.market_schema import Regime


def regime_label(
    forward_return: float,
    realized_vol: float,
    *,
    high_vol: float = 0.02,
    low_vol: float = 0.005,
    trend: float = 0.01,
    range_band: float = 0.002,
) -> Regime:
    """Heuristic regime target from realised return + volatility (market-derived; SPEC.md §17, §24).

    A simple, deterministic supervised label for the regime head (EVENT is flagged from the event
    calendar elsewhere, not derivable here). High vol dominates; otherwise a strong move is a TREND,
    a tight move a RANGE, a low-vol quiet bar LOW_VOL, and the rest CHOP.
    """
    move = abs(forward_return)
    if realized_vol >= high_vol:
        return Regime.HIGH_VOL
    if move >= trend:
        return Regime.TREND
    if realized_vol <= low_vol:
        return Regime.LOW_VOL if move <= range_band else Regime.RANGE
    return Regime.RANGE if move <= range_band else Regime.CHOP


def state_regime_label(
    trailing_return: float,
    trailing_vol: float,
    *,
    event: bool = False,
    high_vol: float = 0.02,
    low_vol: float = 0.005,
    trend: float = 0.01,
    range_band: float = 0.002,
) -> Regime:
    """Point-in-time regime heuristic from trailing state only.

    Unlike ``regime_label``, this must be safe to compute at decision time. Event days dominate,
    then the same volatility/move thresholds classify the current state into trend/range/vol/chop.
    """
    if event:
        return Regime.EVENT
    return regime_label(
        trailing_return,
        trailing_vol,
        high_vol=high_vol,
        low_vol=low_vol,
        trend=trend,
        range_band=range_band,
    )


__all__ = [
    "SESSION_OPEN",
    "SESSION_CLOSE",
    "SESSION_MINUTES",
    "log_returns",
    "simple_returns",
    "realized_vol",
    "atr",
    "oi_change",
    "volume_zscore",
    "time_of_day",
    "day_of_week",
    "session_boundary_mask",
    "hl_range",
    "atr_pct",
    "open_close_norm",
    "oi_norm",
    "d_oi_pct",
    "tod_sin",
    "tod_cos",
    "dow_sin",
    "dow_cos",
    "bb_position",
    "rsi",
    "momentum_norm",
    "session_return",
    "high_low_pos",
    "dmi",
    "variance_ratio",
    "realized_vol_rs",
    "opening_range_position",
    "first_window_return",
    "regime_label",
    "state_regime_label",
]
