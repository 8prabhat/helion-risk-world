"""Compatibility adapter: reads alpha_data's precomputed per-symbol technical-feature
parquets and reshapes them into helion's own ``CANDLE_FEATURE_NAMES`` convention, so
``FeatureBuilder`` can use this in place of ``MarketWindowBuilder`` with no change to
downstream model code -- the shared ``FeatureBuilder``/``MarketBatch``/
``MarketFeatureHistory`` layer sees the same [A, L, F] / [T, A, F] tensor either way.

This is Phase 2 of the alpha_data migration (see ``alpha_data/docs/DATA_CATALOG.md`` --
"Three-repo feature audit"). alpha_data's ``pipelines/technical_features.py`` was
extended this same session specifically to close the gaps a column-name diff against
``CANDLE_FEATURE_NAMES`` found, and to add BANKNIFTY/NIFTY/FINNIFTY as first-class
assets (previously only BANKNIFTY_FUT continuous futures was computed there).

REMAINING KNOWN DISCREPANCIES -- genuine formula/convention differences, not bugs,
flagged for the Phase 2 parity check and human review before this becomes the default:

* ``bb_position``: helion's own formula is ``(close - SMA20) / (2 * std20)`` (a
  z-score-like measure, roughly bounded [-3, 3] after clipping). alpha_data's
  ``bollinger_position`` is ``(close - lower_band) / (upper_band - lower_band)`` (a
  position-within-band fraction, bounded [0, 1]). Genuinely different conventions,
  not a bug -- same category as Phase 1's RSI/ATR EWM-convention differences.
* ``dmi_diff_14``: helion's own formula is ``(+DI - -DI) / 100`` (flat /100 scale).
  alpha_data's ``dmi_diff`` is ``(+DI - -DI) / (+DI + -DI + eps)`` (normalized by the
  DI sum, not a flat divisor) -- both land in a similar [-1, 1] range but are not
  numerically identical.
* ``rsi_14``/``adx_14``: pure scale conversions (alpha_data's ``rsi``/``adx`` are
  0-100; helion wants 0-1), handled here by dividing by 100 -- not a discrepancy.

Columns with no alpha_data equivalent needed (filled in downstream by
``FeatureBuilder`` after stacking every symbol, matching ``MarketWindowBuilder``'s own
convention): ``rel_log_return``, ``breadth``, ``dispersion``.

``oi_norm``/``d_oi_pct`` are only meaningful for a real futures/OI-bearing instrument;
alpha_data only computes them for BANKNIFTY_FUT. helion's own comment on these columns
already says "(0 for NSE indices)" -- since every symbol in helion's default universe
is a spot index or cash equity (not BANKNIFTY_FUT itself), filling 0 here matches
helion's own documented expectation, not a gap.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from alpha_data.io.paths import DataPaths as AlphaDataPaths
from helion_risk_world.data.alpha_cross_pair_context import aligned_cross_pair
from helion_risk_world.data.market_window_builder import CANDLE_FEATURE_NAMES

# CANDLE_FEATURE_NAMES column -> alpha_data technical-feature column. Columns absent
# from this dict are handled specially (see module docstring): filled with 0.0 by this
# adapter (oi_norm/d_oi_pct when missing) or by FeatureBuilder itself after stacking
# (rel_log_return/breadth/dispersion).
_COLUMN_MAP: dict[str, str] = {
    "log_return": "log_return_1",
    "hl_range": "hl_range",
    "open_close_norm": "open_close_norm",
    "realized_vol_short": "realized_vol_short",
    "realized_vol_long": "realized_vol_long",
    "atr_pct": "atr_pct",
    "bb_position": "bollinger_position",       # NOTE: different formula, see docstring
    "momentum_norm": "momentum_norm",
    "session_return": "session_cumulative_return",
    "high_low_pos": "high_low_pos",
    "volume_zscore": "volume_zscore",
    "oi_norm": "oi_norm",
    "d_oi_pct": "d_oi_pct",
    "tod_sin": "tod_sin",
    "tod_cos": "tod_cos",
    "dow_sin": "dow_sin",
    "dow_cos": "dow_cos",
    "dmi_diff_14": "dmi_diff",                 # NOTE: different normalization, see docstring
    "variance_ratio_20": "variance_ratio",
    "vol_ratio_short_long": "vol_ratio_short_long",
    "opening_range_position": "opening_range_position",
    "first_15min_return": "first_15min_return",
    # cross_pair_beta/cross_pair_corr/cross_pair_relative_strength (columns 27-29) are NOT
    # in this map -- they come from a separate alpha_data file
    # (alpha_cross_pair_context.py::aligned_cross_pair), joined in below after this loop,
    # since kalman_trend/kalman_innovation_norm/kalman_trend_uncertainty were replaced
    # (see market_window_builder.py's column 27-29 comment for why).
}

# Columns that are a pure /100 scale conversion (alpha_data: 0-100; helion: 0-1).
_SCALE_BY_100 = {"rsi_14": "rsi", "adx_14": "adx"}

# Filled with 0.0 unconditionally -- FeatureBuilder overwrites these after stacking
# every symbol's window (matching MarketWindowBuilder's own convention exactly).
_ZERO_FILLED = {"rel_log_return", "breadth", "dispersion"}

# Sourced from alpha_cross_pair_context.py rather than the technical parquet's _COLUMN_MAP.
_CROSS_PAIR_COLUMNS = ("cross_pair_beta", "cross_pair_corr", "cross_pair_relative_strength")


def load_alpha_data_technical(
    symbol: str, *, interval: str = "5min", paths: AlphaDataPaths | None = None,
) -> pd.DataFrame:
    """Load alpha_data's precomputed technical-feature parquet for ``symbol``, renamed
    and reshaped to helion's own ``CANDLE_FEATURE_NAMES`` column convention (in order).

    Missing alpha_data columns (correlation-pruned away for this specific symbol, or
    genuinely not computed e.g. oi_norm/d_oi_pct for a non-futures symbol) are
    filled with 0.0 rather than raising -- matches ``MarketWindowBuilder``'s own
    NaN-fill-to-0.0 encoder-readiness convention.
    """
    paths = paths or AlphaDataPaths()
    path = paths.features / f"{symbol}_technical_{interval}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"load_alpha_data_technical: {path} not found. Run "
            f"`alpha-data compute-technical --interval {interval}` in alpha_data first."
        )
    tech = pd.read_parquet(path)
    # alpha_data's technical parquet is tz-aware; helion's own ParquetMarketDataSource
    # index is tz-naive (same underlying instants, numerically UTC). A reindex across
    # a tz-aware/tz-naive mismatch matches nothing at all and silently produces an
    # all-NaN -> all-zero result -- the same class of tz bug found repeatedly in this
    # migration (Phase 1's macro z-score loader, Phase 2's HDFC blackout mask, and
    # Phase 2's futures-microstructure adapter, which additionally needed a tz
    # CONVERSION, not just a strip, since alpha_data isn't internally consistent about
    # which tz its own parquets carry -- spot/technical/options files are tz-aware
    # UTC, but futures-continuous files are tz-aware Asia/Kolkata. tz_convert("UTC")
    # first makes this correct regardless of which convention this particular file
    # happens to use, rather than assuming it's always already UTC.
    if tech.index.tz is not None:
        tech.index = tech.index.tz_convert("UTC").tz_localize(None)

    out = pd.DataFrame(index=tech.index)
    for helion_col in CANDLE_FEATURE_NAMES:
        if helion_col in _ZERO_FILLED:
            out[helion_col] = 0.0
        elif helion_col in _SCALE_BY_100:
            src = _SCALE_BY_100[helion_col]
            out[helion_col] = tech[src] / 100.0 if src in tech.columns else 0.0
        elif helion_col in _CROSS_PAIR_COLUMNS:
            out[helion_col] = 0.0  # overwritten below via aligned_cross_pair
        else:
            src = _COLUMN_MAP.get(helion_col, helion_col)
            out[helion_col] = tech[src] if src in tech.columns else 0.0

    beta, corr, rel_strength = aligned_cross_pair(symbol, out.index, interval=interval, paths=paths)
    out["cross_pair_beta"] = beta
    out["cross_pair_corr"] = corr
    out["cross_pair_relative_strength"] = rel_strength
    return out[list(CANDLE_FEATURE_NAMES)]


class AlphaDataMarketWindowBuilder:
    """Drop-in replacement for ``MarketWindowBuilder`` that reads alpha_data's
    precomputed technical features instead of computing them from raw OHLCV.

    Exposes ``build_frame_for_symbol``/``build_for_symbol`` (a symbol-aware superset
    of ``MarketWindowBuilder``'s ``build_frame``/``build`` -- neither of which alone
    identifies which symbol's data it's processing). ``FeatureBuilder`` duck-type
    checks for these and passes the symbol through when present, falling back to the
    plain ``build``/``build_frame`` call for the original ``MarketWindowBuilder``.
    """

    def __init__(self, *, interval: str = "5min", paths: AlphaDataPaths | None = None) -> None:
        self._interval = interval
        self._paths = paths or AlphaDataPaths()
        self._cache: dict[str, pd.DataFrame] = {}

    def _technical(self, symbol: str) -> pd.DataFrame:
        cached = self._cache.get(symbol)
        if cached is None:
            cached = load_alpha_data_technical(symbol, interval=self._interval, paths=self._paths)
            self._cache[symbol] = cached
        return cached

    def build_frame_for_symbol(
        self, symbol: str, frame: pd.DataFrame,
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        """Return (features [T, 30], CANDLE_FEATURE_NAMES) for ``frame``'s timestamps."""
        tech = self._technical(symbol)
        window = tech.reindex(frame.index)
        feats = np.nan_to_num(window.to_numpy(dtype=np.float32), nan=0.0)
        return feats, CANDLE_FEATURE_NAMES

    def build_for_symbol(
        self, symbol: str, candles: list, ts: datetime | None = None,
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        """Return (features [L, 30], CANDLE_FEATURE_NAMES) for a candle-list window."""
        if len(candles) < 2:
            raise ValueError(f"need >= 2 candles, got {len(candles)}")
        index = pd.DatetimeIndex([c.ts for c in candles])
        tech = self._technical(symbol)
        window = tech.reindex(index)
        feats = np.nan_to_num(window.to_numpy(dtype=np.float32), nan=0.0)
        return feats, CANDLE_FEATURE_NAMES


__all__ = ["load_alpha_data_technical", "AlphaDataMarketWindowBuilder"]
