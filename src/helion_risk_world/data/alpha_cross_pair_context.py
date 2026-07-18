"""Compatibility adapter: per-bar rolling beta/correlation/relative-strength vs
BANKNIFTY_FUT, backed by alpha_data's real ``pipelines/cross_asset_features.py`` pairwise
relationship pipeline (2026-07-14).

This is a genuine CAPABILITY UPGRADE, not a parity port: ``CrossAssetEncoder``
(``encoders/cross_asset_encoder.py``) mean-pools the candle tensor's time axis before
running self-attention across the asset axis, so it structurally cannot reconstruct a
rolling covariance/beta/lead-lag relationship between assets from the candle tensor alone
-- the time-series shape that calculation needs is destroyed before attention ever runs.
alpha_data already computes this explicitly (60-bar rolling window) for each bank
constituent vs BANKNIFTY_FUT, plus BANKNIFTY_FUT vs NIFTY; this loader reads it directly
(no local recomputation) and replaces the three previously dead/weak Kalman-derived
candle-tensor columns (see ``market_window_builder.py``'s column 27-29 comment).

Reference-pair mapping (fixed, matches what alpha_data actually materializes -- see
``compute_all`` in ``alpha_data/pipelines/cross_asset_features.py``):
  - the 5 bank constituents each have their own ``{symbol}_vs_BANKNIFTY_FUT_{interval}.parquet``
  - BANKNIFTY itself uses ``BANKNIFTY_FUT_vs_NIFTY_{interval}.parquet`` (its own relationship to
    the broader index -- the closest available proxy for "BANKNIFTY's cross-asset row")
  - NIFTY/FINNIFTY have no direct pair file; zero-filled, matching the existing documented
    precedent for oi_norm/d_oi_pct ("0 for NSE indices") -- an honest gap, not a fabricated value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from quanthelion.utils.logging import get_logger

from alpha_data.io.paths import DataPaths as AlphaDataPaths

log = get_logger(__name__)

_BETA_CLIP = 5.0

_COLUMNS = ("rolling_beta", "rolling_corr", "relative_strength")

# symbol -> materialized cross-pair filename stem (see module docstring). Symbols absent
# from this map have no direct pair file computed by alpha_data.
_REFERENCE_STEM: dict[str, str] = {
    "HDFCBANK": "HDFCBANK_vs_BANKNIFTY_FUT",
    "ICICIBANK": "ICICIBANK_vs_BANKNIFTY_FUT",
    "SBIN": "SBIN_vs_BANKNIFTY_FUT",
    "AXISBANK": "AXISBANK_vs_BANKNIFTY_FUT",
    "KOTAKBANK": "KOTAKBANK_vs_BANKNIFTY_FUT",
    "BANKNIFTY": "BANKNIFTY_FUT_vs_NIFTY",
}


def load_alpha_data_cross_pair(
    symbol: str, *, interval: str = "5min", paths: AlphaDataPaths | None = None,
) -> pd.DataFrame | None:
    """Load alpha_data's precomputed cross-pair relationship parquet for ``symbol``.

    Returns a DataFrame with columns ``rolling_beta``/``rolling_corr``/``relative_strength``
    (UTC-naive index, ``rolling_beta`` clipped to +-5.0 -- the raw series has a thin tail of
    extreme values from near-zero-variance-denominator moments, ~0.2% of rows beyond +-5 on
    real data). Returns ``None`` when ``symbol`` has no materialized reference pair (NIFTY,
    FINNIFTY, or any symbol outside the default universe) -- callers zero-fill in that case,
    matching ``oi_norm``/``d_oi_pct``'s existing "0 for NSE indices" convention.
    """
    stem = _REFERENCE_STEM.get(symbol)
    if stem is None:
        return None
    paths = paths or AlphaDataPaths()
    path = paths.features / f"{stem}_{interval}.parquet"
    if not path.exists():
        log.warning("no alpha_data cross-pair context found", symbol=symbol, path=str(path))
        return None
    df = pd.read_parquet(path)[list(_COLUMNS)]
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    df.index = idx
    df = df.sort_index()
    df["rolling_beta"] = df["rolling_beta"].clip(lower=-_BETA_CLIP, upper=_BETA_CLIP)
    return df


def aligned_cross_pair(
    symbol: str, index: pd.DatetimeIndex, *, interval: str = "5min", paths: AlphaDataPaths | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (rolling_beta, rolling_corr, relative_strength) reindexed onto ``index``.

    Zero-filled where ``symbol`` has no reference pair or a given timestamp isn't covered.
    """
    n = len(index)
    df = load_alpha_data_cross_pair(symbol, interval=interval, paths=paths)
    if df is None:
        zeros = np.zeros(n, dtype=np.float32)
        return zeros, zeros.copy(), zeros.copy()
    target = pd.DatetimeIndex(index)
    if target.tz is not None:
        target = target.tz_convert("UTC").tz_localize(None)
    aligned = df.reindex(target)
    beta = aligned["rolling_beta"].fillna(0.0).to_numpy(dtype=np.float32)
    corr = aligned["rolling_corr"].fillna(0.0).to_numpy(dtype=np.float32)
    rel_strength = aligned["relative_strength"].fillna(0.0).to_numpy(dtype=np.float32)
    return beta, corr, rel_strength


__all__ = ["load_alpha_data_cross_pair", "aligned_cross_pair"]
