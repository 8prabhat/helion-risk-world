"""Compatibility adapter: reads alpha_data's precomputed futures-microstructure
parquet instead of computing it from the assembled parquet
(``scripts/assemble_data.py``) -- drop-in replacement for ``FuturesWindowBuilder``'s
``build_window``/``build_history`` (the only two methods ``FeatureBuilder`` and
callers of the precomputed-history path actually use).

alpha_data's ``pipelines/futures_microstructure.py`` was built this same session
(Phase 2 migration) using the EXACT same 14-column layout and ordering as helion's own
``FuturesWindowBuilder`` docstring (basis, oi_norm, d_oi, volume_zscore,
calendar_spread, dte_norm, roll_flag, d_oi_mag, oi_available, oi_basis_interaction,
long_buildup, short_covering, short_buildup, long_unwinding) -- no column renaming
needed, only a tz-naive/tz-aware index reconciliation (alpha_data's technical/
microstructure parquets are tz-aware UTC; helion's own assembled data is tz-naive).

``build_barrier_context[_history]`` ARE reproduced here (unlike an earlier draft of
this module's plan): ``ModelInputBuilder.build``/``build_many`` call them directly on
whatever object ``FeatureBuilder.futures_builder`` holds, so a drop-in replacement
needs them too, not just ``build_window``/``build_history``. They call
``quanthelion.labels.barrier_context`` (ported verbatim from helion's own
``barrier_context.py``, Phase 2, verified byte-identical) against alpha_data's own
continuous-futures close series -- the same underlying price series
``FuturesWindowBuilder._close_fut()`` reads, just sourced from alpha_data instead of
the assembled parquet.

``validate_window``/``eligible_positions`` are reproduced as simple, permissive
history-availability checks (enough real rows exist before ``ts``) rather than a
literal port of the original's segment_id/roll_gap/stale-price-run checks -- those
checks are specific to helion's own assembled-parquet construction issues (which
alpha_data's own pipeline doesn't have: no segment_id concept, and stale-price/
roll-gap handling already happens upstream in alpha_data's own ingestion QA). If
per-window eligibility gating beyond basic history-availability is still wanted
against alpha_data data, it needs its own review, not a blind port.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from quanthelion.labels.barrier_context import (
    BarrierContext,
    BarrierSpec,
    barrier_context_from_sigma,
    barrier_context_series,
    ewma_barrier_sigma,
)

from alpha_data.io.paths import DataPaths as AlphaDataPaths

FUTURES_FEATURE_DIM = 14


@dataclass(frozen=True)
class FuturesWindowQuality:
    """Sample eligibility for one futures lookback window (permissive history-availability
    check, per this module's docstring -- alpha_data's own ingestion QA already handles
    segment/roll-gap/stale-price issues upstream, so this doesn't re-check them)."""

    eligible: bool
    reasons: tuple[str, ...]
    n_bars: int

_EXPECTED_COLUMNS = (
    "basis", "oi_norm", "d_oi", "volume_zscore", "calendar_spread", "dte_norm",
    "roll_flag", "d_oi_mag", "oi_available", "oi_basis_interaction",
    "long_buildup", "short_covering", "short_buildup", "long_unwinding",
)


def _to_naive_utc(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Normalize any tz-aware index to naive-UTC-numeric, matching
    ``ParquetMarketDataSource``'s own storage convention (naive timestamps whose
    numeric value IS the UTC instant).

    alpha_data is NOT internally consistent about which tz its own parquets carry:
    spot/technical/options files are tz-aware UTC (so a blind ``tz_localize(None)``
    happens to already produce the right naive-UTC-numeric value), but
    ``BANKNIFTY_FUT_continuous``/futures-microstructure files are tz-aware
    Asia/Kolkata (a DIFFERENT numeric value for the same instant) -- blindly
    stripping tz there silently produces timestamps offset by 5:30 from every other
    source, which is exactly why an early version of this adapter's ``build_many``
    integration returned zero matches against the candle-feature side. Converting to
    UTC before stripping handles both conventions correctly regardless of which one
    a given alpha_data file happens to use.
    """
    if index.tz is None:
        return index
    return index.tz_convert("UTC").tz_localize(None)


def load_alpha_data_futures_microstructure(
    underlying: str = "BANKNIFTY", *, interval: str = "5min", paths: AlphaDataPaths | None = None,
) -> pd.DataFrame:
    """Load alpha_data's precomputed futures-microstructure parquet, tz-naive indexed
    and column-ordered to exactly match helion's own 14-column layout."""
    paths = paths or AlphaDataPaths()
    path = paths.features / f"{underlying}_FUT_futures_microstructure_{interval}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"load_alpha_data_futures_microstructure: {path} not found. Run "
            f"`alpha-data compute-futures-microstructure {underlying} --interval {interval}` "
            f"in alpha_data first."
        )
    df = pd.read_parquet(path)
    df.index = _to_naive_utc(df.index)
    missing = [c for c in _EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"alpha_data futures-microstructure parquet missing columns: {missing}")
    return df[list(_EXPECTED_COLUMNS)]


class AlphaDataFuturesWindowBuilder:
    """Drop-in replacement for ``FuturesWindowBuilder`` reading alpha_data's
    precomputed futures-microstructure features instead of the assembled parquet."""

    def __init__(
        self,
        underlying: str = "BANKNIFTY",
        *,
        interval: str = "5min",
        paths: AlphaDataPaths | None = None,
    ) -> None:
        self._paths = paths or AlphaDataPaths()
        self._df = load_alpha_data_futures_microstructure(underlying, interval=interval, paths=self._paths)
        fut_path = self._paths.ohlcv / f"{underlying}_FUT_continuous_{interval}.parquet"
        close = pd.read_parquet(fut_path)["close"]
        close.index = _to_naive_utc(close.index)
        self._close = close.reindex(self._df.index).ffill()
        self._sigma_cache: dict[int, np.ndarray] = {}

    def build_history(self) -> tuple[pd.DatetimeIndex, np.ndarray]:
        """Return the full per-bar futures feature history as (index, [T, 14])."""
        feats = np.nan_to_num(self._df.to_numpy(dtype=np.float32), nan=0.0)
        return self._df.index, feats

    def build_window(self, ts: datetime, lookback: int, *, strict: bool = False) -> np.ndarray:
        """Return [T=lookback, 14] float32 array at decision time ``ts``.

        Point-in-time safe: only bars with index <= ts are used. Returns zeros for the
        warm-up region if fewer than ``lookback`` bars are available before ``ts``.
        """
        end = self._df.index.searchsorted(ts, side="right")
        history = self._df.iloc[:end]
        if history.empty:
            return np.zeros((lookback, FUTURES_FEATURE_DIM), dtype=np.float32)

        feats = np.nan_to_num(history.to_numpy(dtype=np.float32), nan=0.0)[-lookback:]
        if len(feats) < lookback:
            pad = np.zeros((lookback - len(feats), FUTURES_FEATURE_DIM), dtype=np.float32)
            feats = np.concatenate([pad, feats], axis=0)
        assert feats.shape == (lookback, FUTURES_FEATURE_DIM), feats.shape
        return feats

    def validate_window(self, ts: datetime, lookback: int) -> None:
        """Raise if fewer than ``lookback`` real bars are available before ``ts``."""
        end = self._df.index.searchsorted(ts, side="right")
        if end < lookback:
            raise ValueError(
                f"invalid futures lookback at {ts}: only {end} bars available, need {lookback}"
            )

    def quality_for_window(self, ts: datetime, lookback: int) -> FuturesWindowQuality:
        """Permissive sample eligibility for the futures lookback ending at ``ts`` --
        see this module's/``FuturesWindowQuality``'s docstrings for why this doesn't
        reproduce the pre-migration segment_id/roll-gap/stale-price checks."""
        end = self._df.index.searchsorted(ts, side="right")
        n_bars = min(end, lookback)
        if end < lookback:
            return FuturesWindowQuality(False, ("incomplete_lookback",), n_bars)
        return FuturesWindowQuality(True, (), n_bars)

    def eligible_positions(self, lookback: int) -> np.ndarray:
        """Boolean mask over the full futures index: True once enough trailing
        history exists (position >= lookback - 1)."""
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        eligible = np.zeros(len(self._df), dtype=bool)
        eligible[lookback - 1:] = True
        return eligible

    def _sigma_series(self, vol_span: int) -> np.ndarray:
        cached = self._sigma_cache.get(int(vol_span))
        if cached is None:
            cached = ewma_barrier_sigma(self._close.to_numpy(dtype=float), span=vol_span)
            self._sigma_cache[int(vol_span)] = cached
        return cached

    def build_barrier_context_history(
        self,
        *,
        stop_mult: float = 2.0,
        target_mult: float = 2.0,
        vol_span: int = 50,
        horizon_bars: int = 1,
        cost_floor_frac: float = 0.0,
    ) -> tuple[pd.DatetimeIndex, np.ndarray]:
        """Return the full per-bar barrier context history as ``[sigma, stop, target]``."""
        spec = BarrierSpec(
            stop_mult=stop_mult, target_mult=target_mult, vol_span=vol_span,
            horizon_bars=horizon_bars, cost_floor_frac=cost_floor_frac,
        )
        rows = barrier_context_series(self._close.to_numpy(dtype=float), spec=spec)
        return self._df.index, rows

    def build_barrier_context(
        self,
        ts: datetime,
        *,
        stop_mult: float = 2.0,
        target_mult: float = 2.0,
        vol_span: int = 50,
        horizon_bars: int = 1,
        cost_floor_frac: float = 0.0,
    ) -> BarrierContext:
        """Return the explicit barrier geometry available at decision time ``ts``."""
        end = self._df.index.searchsorted(ts, side="right")
        if end <= 0:
            raise ValueError(f"no futures barrier context available at {ts}")
        spec = BarrierSpec(
            stop_mult=stop_mult, target_mult=target_mult, vol_span=vol_span,
            horizon_bars=horizon_bars, cost_floor_frac=cost_floor_frac,
        )
        sigma = self._sigma_series(spec.vol_span)[end - 1]
        return barrier_context_from_sigma(float(sigma), spec=spec)


__all__ = [
    "FUTURES_FEATURE_DIM",
    "FuturesWindowQuality",
    "load_alpha_data_futures_microstructure",
    "AlphaDataFuturesWindowBuilder",
]
