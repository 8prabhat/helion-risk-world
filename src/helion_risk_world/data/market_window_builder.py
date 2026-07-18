"""Candle/OI feature schema (SPEC.md §10).

The actual per-bar computation this module used to own now lives in alpha_data (Phase
2 migration, see alpha_data/docs/DATA_CATALOG.md's "Three-repo feature audit") --
``data/alpha_features.py::AlphaDataMarketWindowBuilder`` reads alpha_data's precomputed
technical-feature parquets instead of computing this tensor from raw OHLCV. This module
now only keeps the ``CANDLE_FEATURE_NAMES`` schema (the [L, F=30] column contract
``FeatureBuilder``/``ModelInputContract``/every encoder still key off of by name).

All features are price-normalized or dimensionless so they are comparable across assets
with very different price scales (BANKNIFTY ~50K vs HDFCBANK ~1.6K).

``rel_log_return`` (column 18), ``breadth`` (column 25), and ``dispersion`` (column 26)
are filled with zeros by the adapter; ``FeatureBuilder`` overwrites them after stacking
all assets -- rel_log_return per-asset-differentiated, breadth/dispersion as
market-wide scalars broadcast identically into every asset's row (see
feature_builder.py).
"""

from __future__ import annotations

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
    # Columns 27-29 were kalman_trend/kalman_innovation_norm/kalman_trend_uncertainty until
    # 2026-07-14: the saved IC report (runs/feature_ic_report_full.csv) showed
    # kalman_trend_uncertainty at IC=0.0 on every horizon/fold (a steady-state artifact of the
    # filter's fixed-noise-variance design, verified constant against the live parquet after
    # warm-up), kalman_innovation_norm likewise ~0.000 everywhere, and kalman_trend itself weak
    # (IC 0.02) and ~0.93-correlated with a simpler ema_dist feature per alpha_data's own
    # correlation-pruning log. Replaced with alpha_data's cross_asset_features.py rolling-window
    # relationship to BANKNIFTY_FUT (rolling_beta/rolling_corr/relative_strength,
    # data/alpha_cross_pair_context.py) -- CrossAssetEncoder mean-pools the time axis before
    # attending across assets (encoders/cross_asset_encoder.py), so it cannot reconstruct rolling
    # beta/correlation itself; these give it that relational signal directly.
    "cross_pair_beta",               # 27  60-bar rolling beta vs BANKNIFTY_FUT (clipped [-5,5])
    "cross_pair_corr",                # 28  60-bar rolling correlation vs BANKNIFTY_FUT, in [-1,1]
    "cross_pair_relative_strength",   # 29  60-bar relative-strength vs BANKNIFTY_FUT
)

__all__ = ["CANDLE_FEATURE_NAMES"]
