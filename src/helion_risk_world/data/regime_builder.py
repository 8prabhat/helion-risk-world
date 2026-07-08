"""Regime/event context featurisation (SPEC.md §14, §16; torch-free, DRY).

K = 15 numeric + len(EventType) one-hot = 22 total.

Scaling conventions:
  vix           / 20      → O(0.5-1.5)
  atm_iv        / 100     → O(0.1-0.3)   (stored as percentage e.g. 15.0%)
  iv_skew       / 10      → O(0.1-0.5)   (stored as pct-points e.g. 2.1)
  fii_dii_net_z            O(-3, 3)      rolling 60-day z-score (see
                           data/daily_context_loader.py::_add_derived_macro_columns) —
                           NOT a fixed-divisor rescale: FII flow has no stable scale
                           over a multi-year window as the market grows, so a z-score
                           against its own trailing history is the stationary quantity.
  usdinr_ret_5d            O(-0.02, 0.02) 5-trading-day log return, not the raw level —
                           USD/INR is a trending macro driver (drifted 82→95 over this
                           project's training window); the level's "meaning" isn't fixed
                           over time, but a recent-shock return is.
  crude_ret_5d             same rationale, WTI crude.
  pc_oi_ratio_z            O(-3, 3)      rolling 60-day z-score, same rationale as
                           fii_dii_net_z (PCR is a mean-reverting sentiment ratio, not
                           naturally centered at 0 — z-scoring is more informative than
                           the previous fixed "-1.0" centering, which ignored PCR's own
                           dispersion).
  basis_daily   * 100      → O(0.1-0.5)   (basis ~0.003 → 0.3)
  usdinr_vol               O(0, 0.02)     rolling std of usdinr_ret_5d (feature/label
                           overhaul Phase 2) — cross-asset vol transmission signal
                           (FX vol spilling into equity vol), a distinct regime axis
                           from the candle-plane's own realized-vol features.
  crude_vol                same rationale, rolling std of crude_ret_5d.
  regime_missing_mask      → 1.0 when atm_iv/iv_skew/pc_oi_ratio/basis_daily are ALL
                           unavailable (review Idea #5), else 0.0. Lets the model
                           distinguish "genuinely zero/neutral" from "no signal
                           here" instead of conflating both into the same 0.0 the
                           numeric features already fall back to below. Uses the RAW
                           field's availability, not the derived z-score/return's — the
                           latter can be transiently NaN during the derived series'
                           own rolling-window warmup even when the raw value exists,
                           which is a normalization artifact, not a data-missing fact.
"""

from __future__ import annotations

import numpy as np

from helion_risk_world.schemas.market_schema import EventContext, EventType, RegimeContext

_NUMERIC: tuple[str, ...] = (
    "vix", "vix_pct", "atm_iv", "iv_skew", "expiry_flag", "event_day_flag",
    "blackout_active", "fii_dii_net_z", "usdinr_ret_5d", "crude_ret_5d",
    "pc_oi_ratio_z", "basis_daily", "usdinr_vol", "crude_vol", "regime_missing_mask",
)
EVENT_TYPE_ONEHOT: tuple[str, ...] = tuple(f"event_{t.value}" for t in EventType)
REGIME_CONTEXT_FEATURES: tuple[str, ...] = _NUMERIC + EVENT_TYPE_ONEHOT


def featurize_regime(regime: RegimeContext, event: EventContext) -> np.ndarray:
    """``RegimeContext`` + ``EventContext`` -> [K=22] float32."""
    missing_mask = 1.0 if (
        regime.atm_iv is None
        and regime.iv_skew is None
        and event.pc_oi_ratio is None
        and event.basis_daily is None
    ) else 0.0
    numeric = [
        (regime.vix or 0.0) / 20.0,
        regime.vix_pct,
        (regime.atm_iv or 0.0) / 100.0,         # stored as %, e.g. 15.0 → 0.15
        (regime.iv_skew or 0.0) / 10.0,          # stored as pct-pts, e.g. 2.0 → 0.20
        1.0 if event.expiry_flag else 0.0,
        1.0 if event.event_day_flag else 0.0,
        1.0 if event.blackout_active else 0.0,
        event.fii_dii_net_z or 0.0,              # rolling 60d z-score (review: was /50000)
        event.usdinr_ret_5d or 0.0,               # 5d log return (review: was raw level/100)
        event.crude_ret_5d or 0.0,                 # 5d log return (review: was raw level/100)
        event.pc_oi_ratio_z or 0.0,               # rolling 60d z-score (review: was -1.0 centering)
        (event.basis_daily or 0.0) * 100.0,       # basis fraction → percentage units
        event.usdinr_vol or 0.0,                  # rolling std of usdinr_ret_5d
        event.crude_vol or 0.0,                    # rolling std of crude_ret_5d
        missing_mask,
    ]
    onehot = [1.0 if event.event_type is t else 0.0 for t in EventType]
    return np.array(numeric + onehot, dtype=np.float32)
