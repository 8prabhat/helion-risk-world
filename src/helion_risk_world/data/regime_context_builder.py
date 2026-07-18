"""Assemble Upstox-only RegimeContext + EventContext.

Active training/runtime paths may use only:
  VIX        → data/ohlcv/INDIAVIX_5min.parquet  (Upstox)
  Expiry     → disabled here unless supplied by an Upstox-derived caller
  Events     → disabled here; hard-coded macro/event calendars are not Upstox data

The legacy daily context loader still exists for historical experiments, but this builder
rejects it by default so model artifacts cannot silently depend on non-Upstox files.

Usage:
    builder = RegimeContextBuilder.from_paths(
        vix_path=Path("data/ohlcv/INDIAVIX_5min.parquet"),
    )
    regime, event = builder.build(ts)
    vec = featurize_regime(regime, event)  # [K=22] float32 → RegimeEncoder input
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from helion_risk_world.data.daily_context_loader import DailyContextLoader
from quanthelion.calendars.expiry_calendar import ROLL_FLAG_DAYS
from helion_risk_world.schemas.market_schema import EventContext, EventType, RegimeContext

# Rolling window for VIX percentile (in 5-min bars; 252 trading days × 75 bars/day ≈ 18900)
_VIX_PERCENTILE_WINDOW = 18_900

# How close to expiry counts as an "expiry flag" (same as roll_flag threshold)
_EXPIRY_FLAG_DTE = ROLL_FLAG_DAYS


@dataclass
class _VixLoader:
    """Loads INDIAVIX parquet and computes rolling percentile rank (point-in-time safe)."""

    _df: pd.DataFrame = field(repr=False)
    _pct: pd.Series = field(repr=False, init=False)

    def __post_init__(self) -> None:
        close = self._df["close"].astype(float)
        self._pct = close.rolling(_VIX_PERCENTILE_WINDOW, min_periods=20).rank(pct=True)

    @classmethod
    def from_parquet(cls, path: Path) -> "_VixLoader":
        df = pd.read_parquet(path)
        if df.index.name != "datetime" and "datetime" in df.columns:
            df = df.set_index("datetime")
        df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("Asia/Kolkata")
        return cls(_df=df.sort_index())

    def get(self, ts: datetime) -> tuple[float | None, float]:
        """Return (vix_level, vix_pct) at or before ts. vix_pct in [0, 1]."""
        ts_pd = pd.Timestamp(ts)
        if ts_pd.tz is None:
            ts_pd = ts_pd.tz_localize("Asia/Kolkata")
        end = self._df.index.searchsorted(ts_pd, side="right")
        if end == 0:
            return None, 0.5
        idx = self._df.index[end - 1]
        vix = float(self._df.loc[idx, "close"])
        pct = float(self._pct.loc[idx]) if idx in self._pct.index and not pd.isna(self._pct.loc[idx]) else 0.5
        return vix, pct


@dataclass
class RegimeContextBuilder:
    """Assemble (RegimeContext, EventContext) at any intraday timestamp.

    All active default sources are point-in-time safe and Upstox-only:
    - VIX: from parquet, using only bars with index <= ts
    - daily context / macro / calendar events / hard-coded corporate blackouts:
      disabled unless a caller explicitly opts into legacy non-Upstox behavior

    Args:
        vix_loader:          Loaded INDIAVIX series (may be None → vix=15 fallback)
        daily_ctx:           Loaded daily context (may be None → FX/crude = None)
        symbol:              Primary underlying (used for point-in-time checks)
        atm_iv:              ATM implied vol override (None → use historical daily_ctx)
        iv_skew:             Put-call skew override (same)
        require_live_iv:     If True, build() raises RuntimeError when set_live_iv()
                              was never called, instead of silently falling back to
                              the historical daily_ctx value. Off by default since no
                              caller currently wires a live feed; opt in only once one
                              exists and genuinely needs to fail loud on its absence.
    """

    vix_loader: Optional[_VixLoader]
    daily_ctx: Optional[DailyContextLoader]
    symbol: str = "BANKNIFTY"
    atm_iv: Optional[float] = None
    iv_skew: Optional[float] = None
    require_live_iv: bool = False
    allow_non_upstox_context: bool = False

    @classmethod
    def from_paths(
        cls,
        vix_path: Path | None = None,
        daily_context_path: Path | None = None,
        symbol: str = "BANKNIFTY",
        allow_non_upstox_context: bool = False,
    ) -> "RegimeContextBuilder":
        """Construct from file paths (None → source unavailable, uses fallbacks)."""
        vix_loader = None
        if vix_path is not None and Path(vix_path).exists():
            vix_loader = _VixLoader.from_parquet(Path(vix_path))

        daily_ctx = None
        if daily_context_path is not None:
            if not allow_non_upstox_context:
                raise ValueError(
                    "daily_context_path is not Upstox-sourced; pass no daily context "
                    "for production/research-constrained artifacts"
                )
            daily_ctx = DailyContextLoader(Path(daily_context_path))

        return cls(
            vix_loader=vix_loader,
            daily_ctx=daily_ctx,
            symbol=symbol,
            allow_non_upstox_context=allow_non_upstox_context,
        )

    def build(self, ts: datetime) -> tuple[RegimeContext, EventContext]:
        """Return (RegimeContext, EventContext) for decision time ts."""
        dt = ts.date()
        ts_naive = ts

        # ── VIX ──────────────────────────────────────────────────────────────
        if self.vix_loader is not None:
            vix_level, vix_pct = self.vix_loader.get(ts)
        else:
            vix_level, vix_pct = 15.0, 0.5  # neutral fallback when no data

        # ── Daily macro context ───────────────────────────────────────────────
        daily = (
            self.daily_ctx.get(ts)
            if self.allow_non_upstox_context and self.daily_ctx is not None
            else {}
        )
        usdinr       = daily.get("usdinr")
        crude        = daily.get("crude")
        fii_dii_net  = daily.get("fii_dii_net")
        atm_iv_hist  = daily.get("atm_iv_pct")    # stored as %, e.g. 15.0
        iv_skew_hist = daily.get("iv_skew_pct")
        pc_oi_ratio  = daily.get("pc_oi_ratio")
        basis_daily  = daily.get("basis")
        # Stabilized derivatives (see daily_context_loader.py::_add_derived_macro_columns).
        fii_dii_net_z  = daily.get("fii_dii_net_z")
        pc_oi_ratio_z  = daily.get("pc_oi_ratio_z")
        usdinr_ret_5d  = daily.get("usdinr_ret_5d")
        crude_ret_5d   = daily.get("crude_ret_5d")
        usdinr_vol     = daily.get("usdinr_vol")
        crude_vol      = daily.get("crude_vol")
        realized_vol_vix_ratio = daily.get("realized_vol_vix_ratio")
        breadth_index_divergence = daily.get("breadth_index_divergence")

        # Live inference values override historical; fall back to None (-> 0 in featurizer)
        if self.require_live_iv and self.atm_iv is None and self.iv_skew is None:
            raise RuntimeError(
                "RegimeContextBuilder(require_live_iv=True) but set_live_iv() was never "
                "called before build() — no live option-chain client is wired into this "
                "process, so live IV/skew cannot be provided. Either call set_live_iv() "
                "first, or construct with require_live_iv=False to use the historical "
                "daily_context fallback."
            )
        atm_iv_use  = self.atm_iv  if self.atm_iv  is not None else atm_iv_hist
        iv_skew_use = self.iv_skew if self.iv_skew is not None else iv_skew_hist

        regime = RegimeContext(
            symbol=self.symbol,
            ts=ts_naive,
            available_at=ts_naive,
            vix=vix_level or 15.0,
            vix_pct=vix_pct,
            atm_iv=atm_iv_use,
            iv_skew=iv_skew_use,
        )

        # Calendar/event/corporate-action fields are hard-disabled in the default
        # Upstox-only path. Futures DTE/roll features are derived separately from
        # Upstox continuous-futures segment metadata when available.
        expiry_flag = False
        event_day_flag = False
        ev_type = EventType.NONE
        blackout_active = False

        event = EventContext(
            symbol=self.symbol,
            ts=ts_naive,
            available_at=ts_naive,
            expiry_flag=expiry_flag,
            event_day_flag=event_day_flag,
            blackout_active=blackout_active,
            event_type=ev_type,
            fii_dii_net=fii_dii_net,
            usdinr=usdinr,
            crude=crude,
            pc_oi_ratio=pc_oi_ratio,
            basis_daily=basis_daily,
            fii_dii_net_z=fii_dii_net_z,
            pc_oi_ratio_z=pc_oi_ratio_z,
            usdinr_ret_5d=usdinr_ret_5d,
            crude_ret_5d=crude_ret_5d,
            usdinr_vol=usdinr_vol,
            crude_vol=crude_vol,
            realized_vol_vix_ratio=realized_vol_vix_ratio,
            breadth_index_divergence=breadth_index_divergence,
        )
        return regime, event

    def set_live_iv(self, atm_iv: float | None, iv_skew: float | None) -> None:
        """Override IV fields for live inference from a real-time option chain.

        NOTE: no caller in this codebase currently invokes this method — there is
        no Upstox option-chain client implemented yet (see the module docstring).
        This is a hook for that future integration, not a wired data source today.
        Call before build() once such a feed exists; until then atm_iv/iv_skew are
        always sourced from the historical, lagged daily_context parquet.
        """
        self.atm_iv  = atm_iv
        self.iv_skew = iv_skew


def build_regime_tensor(
    builder: RegimeContextBuilder,
    ts: datetime,
) -> np.ndarray:
    """Convenience: build (RegimeContext, EventContext) and featurize to [K] float32."""
    from helion_risk_world.data.regime_builder import featurize_regime
    regime, event = builder.build(ts)
    return featurize_regime(regime, event)


__all__ = ["RegimeContextBuilder", "build_regime_tensor"]
