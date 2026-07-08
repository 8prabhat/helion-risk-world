"""Build the per-symbol temporal candle/OI window [L, F=30] (SPEC.md §10).

All features are price-normalized or dimensionless so they are comparable across assets
with very different price scales (BANKNIFTY ~50K vs HDFCBANK ~1.6K).

``rel_log_return`` (column 18), ``breadth`` (column 25), and ``dispersion`` (column 26)
are filled with zeros here; FeatureBuilder overwrites them after stacking all assets —
rel_log_return per-asset-differentiated, breadth/dispersion as market-wide scalars
broadcast identically into every asset's row (see feature_builder.py).

realized_vol_short/realized_vol_long (columns 3-4) use the Rogers-Satchell OHLC
estimator (feature/label overhaul Phase 2) rather than close-to-close realized vol —
strictly more information-efficient (uses all 4 OHLC prices, drift-independent) at the
same two window slots; a replacement, not an addition.

kalman_trend/kalman_innovation_norm/kalman_trend_uncertainty (columns 27-29, feature/
label overhaul Phase 3) come from data/kalman_trend.py::local_linear_trend_filter — the
only stateful/recursive feature in this file; see that module's docstring for the
segment-reset discipline it applies.
"""

from __future__ import annotations

import numpy as np

from helion_risk_world.data import primitives as P
from helion_risk_world.data.kalman_trend import local_linear_trend_filter
from helion_risk_world.schemas.market_schema import MarketCandle

CANDLE_FEATURE_NAMES: tuple[str, ...] = (
    "log_return",             # 0   bar log return
    "hl_range",               # 1   (H-L)/C intrabar range fraction
    "open_close_norm",        # 2   (C-O)/(ATR%*C) bar direction
    "realized_vol_short",     # 3   12-bar Rogers-Satchell OHLC vol
    "realized_vol_long",      # 4   60-bar Rogers-Satchell OHLC vol
    "atr_pct",                # 5   ATR/close price-normalized
    "bb_position",            # 6   Bollinger Band position (C-SMA20)/(2*std20)
    "rsi_14",                 # 7   RSI(14)/100 in [0,1]
    "momentum_norm",          # 8   12-bar momentum / (ATR%*C*sqrt(12))
    "session_return",         # 9   (C - first_bar_close_today) / first_bar_close
    "high_low_pos",           # 10  position in 12-bar rolling H/L range [0,1]
    "volume_zscore",          # 11  20-bar volume z-score (0 for NSE indices)
    "oi_norm",                # 12  OI / 96-bar mean OI (0 for indices)
    "d_oi_pct",               # 13  fractional OI change (0 for indices)
    "tod_sin",                # 14  sin(2π * time_of_day_fraction)
    "tod_cos",                # 15  cos(2π * time_of_day_fraction)
    "dow_sin",                # 16  sin(2π * weekday/5)
    "dow_cos",                # 17  cos(2π * weekday/5)
    "rel_log_return",         # 18  log_return - primary_asset_log_return (FeatureBuilder fills this)
    "adx_14",                 # 19  trend strength (magnitude), ADX(14)/100 in [0,1]
    "dmi_diff_14",            # 20  trend direction (sign), (+DI - -DI)/100 in [-1,1]
    "variance_ratio_20",      # 21  Lo-MacKinlay VR(20) - 1.0; 0=random walk
    "vol_ratio_short_long",   # 22  realized_vol_short / realized_vol_long — vol-of-vol regime
    "opening_range_position", # 23  position within the causal 15-min opening range
    "first_15min_return",     # 24  cumulative return over the first 15 min, frozen after
    "breadth",                # 25  fraction of universe with positive N-bar return (FeatureBuilder fills this)
    "dispersion",             # 26  cross-sectional std of universe N-bar returns (FeatureBuilder fills this)
    "kalman_trend",             # 27  Kalman-filtered log-price trend/slope
    "kalman_innovation_norm",   # 28  Kalman innovation, normalized by its predicted variance
    "kalman_trend_uncertainty", # 29  posterior std of the Kalman trend state
)

_VOL_SHORT = 12    # bars (1 hour at 5-min)
_VOL_LONG  = 60    # bars (5 hours)
_ATR_WIN   = 14
_BB_WIN    = 20
_RSI_WIN   = 14
_MOM_WIN   = 12
_HL_WIN    = 12
_VOL_Z_WIN = 20
_OI_WIN    = 96    # bars (8 hours — stable normalization base)
_OR_WINDOW_MINUTES = 15  # opening-range / first-window-return causal window


class MarketWindowBuilder:
    """Assemble a [L, F=30] feature matrix from a candle window. SRP: window assembly."""

    def _build_from_arrays(
        self,
        *,
        timestamps: list,
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        oi: np.ndarray,
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        atr_pct_arr = P.atr_pct(high, low, close, _ATR_WIN)
        vol_short = P.realized_vol_rs(open_, high, low, close, _VOL_SHORT)
        vol_long = P.realized_vol_rs(open_, high, low, close, _VOL_LONG)
        with np.errstate(invalid="ignore", divide="ignore"):
            vol_ratio = vol_short / vol_long
        vol_ratio[~np.isfinite(vol_ratio)] = np.nan
        adx, dmi_diff = P.dmi(high, low, close, _ATR_WIN)
        kalman_trend, kalman_innovation, kalman_uncertainty = local_linear_trend_filter(
            close, timestamps
        )

        columns = {
            "log_return":              P.log_returns(close),
            "hl_range":                P.hl_range(high, low, close),
            "open_close_norm":         P.open_close_norm(open_, close, atr_pct_arr),
            "realized_vol_short":      vol_short,
            "realized_vol_long":       vol_long,
            "atr_pct":                 atr_pct_arr,
            "bb_position":             P.bb_position(close, _BB_WIN),
            "rsi_14":                  P.rsi(close, _RSI_WIN),
            "momentum_norm":           P.momentum_norm(close, atr_pct_arr, _MOM_WIN),
            "session_return":          P.session_return(close, timestamps),
            "high_low_pos":            P.high_low_pos(close, high, low, _HL_WIN),
            "volume_zscore":           P.volume_zscore(volume, _VOL_Z_WIN),
            "oi_norm":                 P.oi_norm(oi, _OI_WIN),
            "d_oi_pct":                P.d_oi_pct(oi),
            "tod_sin":                 P.tod_sin(timestamps),
            "tod_cos":                 P.tod_cos(timestamps),
            "dow_sin":                 P.dow_sin(timestamps),
            "dow_cos":                 P.dow_cos(timestamps),
            "rel_log_return":          np.zeros(len(timestamps), dtype=float),
            "adx_14":                  adx,
            "dmi_diff_14":             dmi_diff,
            "variance_ratio_20":       P.variance_ratio(close, q=_BB_WIN),
            "vol_ratio_short_long":    vol_ratio,
            "opening_range_position": P.opening_range_position(
                close, high, low, timestamps, window_minutes=_OR_WINDOW_MINUTES
            ),
            "first_15min_return":     P.first_window_return(
                close, timestamps, window_minutes=_OR_WINDOW_MINUTES
            ),
            "breadth":                 np.zeros(len(timestamps), dtype=float),
            "dispersion":              np.zeros(len(timestamps), dtype=float),
            "kalman_trend":              kalman_trend,
            "kalman_innovation_norm":    kalman_innovation,
            "kalman_trend_uncertainty":  kalman_uncertainty,
        }

        feats = np.stack([columns[name] for name in CANDLE_FEATURE_NAMES], axis=1)  # [L, 30]
        feats = np.nan_to_num(feats, nan=0.0)
        return feats, CANDLE_FEATURE_NAMES

    def build(self, candles: list[MarketCandle]) -> tuple[np.ndarray, tuple[str, ...]]:
        """Return (features [L, 30], CANDLE_FEATURE_NAMES). Candles sorted ascending by ts.

        NaN warm-up entries are filled with 0.0 so the array is encoder-ready.
        OI and volume are structurally 0 for NSE index instruments.
        rel_log_return (col 18), breadth (col 25), dispersion (col 26) are 0.0 here;
        FeatureBuilder replaces them after stacking all symbols.
        """
        if len(candles) < 2:
            raise ValueError(f"need >= 2 candles, got {len(candles)}")
        ts_sorted = [c.ts for c in candles]
        if ts_sorted != sorted(ts_sorted):
            raise ValueError("candles must be sorted ascending by ts")

        timestamps = [c.ts for c in candles]
        open_  = np.array([c.open  for c in candles], dtype=float)
        high   = np.array([c.high  for c in candles], dtype=float)
        low    = np.array([c.low   for c in candles], dtype=float)
        close  = np.array([c.close for c in candles], dtype=float)
        volume = np.array([c.volume for c in candles], dtype=float)
        oi     = np.array([c.oi if c.oi is not None else 0.0 for c in candles], dtype=float)

        return self._build_from_arrays(
            timestamps=timestamps,
            open_=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
            oi=oi,
        )

    def build_frame(self, frame) -> tuple[np.ndarray, tuple[str, ...]]:
        """Return features directly from an OHLCV DataFrame window."""
        if len(frame) < 2:
            raise ValueError(f"need >= 2 candles, got {len(frame)}")
        timestamps = [ts.to_pydatetime() for ts in frame.index]
        if timestamps != sorted(timestamps):
            raise ValueError("candles must be sorted ascending by ts")
        return self._build_from_arrays(
            timestamps=timestamps,
            open_=frame["open"].to_numpy(dtype=float),
            high=frame["high"].to_numpy(dtype=float),
            low=frame["low"].to_numpy(dtype=float),
            close=frame["close"].to_numpy(dtype=float),
            volume=frame["volume"].to_numpy(dtype=float),
            oi=(
                frame["oi"].to_numpy(dtype=float)
                if "oi" in frame.columns
                else np.zeros(len(frame), dtype=float)
            ),
        )
