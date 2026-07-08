"""Build the [T, FUTURES_FEATURE_DIM] futures microstructure window (SPEC.md §13, §9.2).

This is the data-pipeline companion to ``encoders.futures_encoder.FuturesEncoder``.
It reads columns from the assembled parquet (output of ``scripts/assemble_data.py``) and
produces the 14-feature tensor that FuturesEncoder ingests.

Feature layout (must match ``encoders.futures_encoder.FUTURES_FEATURE_DIM = 14``):

  Scalars [10]:
    0  basis           (close_fut - close_spot) / close_spot
    1  oi_norm         open interest (normalised by rolling mean)
    2  d_oi            bar-to-bar change in OI (sign matters)
    3  volume_zscore   futures volume z-score (rolling 20-bar window)
    4  calendar_spread near - next month close (0 outside the near/next overlap
       window around a roll, or when the assembled parquet has no close_fut_next
       column — see continuous_futures.py::build_continuous, review Idea #6)
    5  dte_norm        days-to-expiry / 30, in [0, 1]
    6  roll_flag       1.0 when within ROLL_FLAG_DAYS of expiry
    7  d_oi_mag        |d_oi| magnitude
    8  oi_available    1.0 if the source OI column has a real value for this bar,
       0.0 if OI is missing entirely or NaN for this bar (review Idea #5) —
       distinguishes "OI is genuinely 0" from "no OI signal here"
    9  oi_basis_interaction  d_oi * sign(basis_t - basis_{t-1}) (feature/label overhaul
       Phase 2) — distinguishes genuine OI accumulation from short-covering-driven
       basis moves; belongs in this plane (not the candle plane) since d_oi/basis are
       structurally 0 for spot/index symbols and only genuinely populated here.

  OI-flow one-hot [4]:
    10  long_buildup   (price up  ∩ OI up)
    11  short_covering (price up  ∩ OI down)
    12  short_buildup  (price down ∩ OI up)
    13  long_unwinding (price down ∩ OI down)

SRP: feature engineering only — no model calls, no portfolio knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd

from helion_risk_world.barrier_context import (
    BarrierContext,
    BarrierSpec,
    barrier_context_series,
    barrier_context_from_sigma,
    ewma_barrier_sigma,
)
from quanthelion.calendars.expiry_calendar import ROLL_FLAG_DAYS
from quanthelion.calendars.expiry_calendar import dte_norm as _dte_norm
from quanthelion.calendars.expiry_calendar import roll_flag as _roll_flag
from helion_risk_world.data.primitives import volume_zscore
from helion_risk_world.encoders.futures_encoder import FUTURES_FEATURE_DIM

_OI_NORM_WINDOW = 20   # bars for rolling OI mean
_VOL_Z_WINDOW = 20     # bars for volume z-score
_EXPECTED_BAR_DELTA = pd.Timedelta(minutes=5)
_STALE_RUN_BARS = 3


def _calendar_spread(window: pd.DataFrame, close_fut: np.ndarray) -> np.ndarray:
    """(next-month close - near-month close) / near-month close (review Idea #6).

    ``close_fut_next`` is produced by ``continuous_futures.py::build_continuous``
    from the real, un-adjusted next-contract print during the near/next overlap
    window around each roll (NaN elsewhere — most bars, since a given contract
    only briefly overlaps its successor). 0.0 wherever the assembled parquet
    doesn't carry this column (older data) or the value is NaN for that bar,
    matching the previous hardcoded-0 fallback semantics.
    """
    if "close_fut_next" not in window.columns:
        return np.zeros(len(window), dtype=float)
    next_close = window["close_fut_next"].to_numpy(dtype=float)
    valid = ~np.isnan(next_close) & (close_fut != 0)
    safe_denom = np.where(close_fut != 0, close_fut, 1.0)
    return np.where(valid, (next_close - close_fut) / safe_denom, 0.0)


def _oi_availability(window: pd.DataFrame) -> np.ndarray:
    """1.0 where the source OI column has a real (non-NaN) value for that bar,
    0.0 where OI is missing entirely (no oi_fut/oi column at all) or NaN for that
    specific bar (review Idea #5). Lets the model distinguish "OI is genuinely
    zero/flat" from "OI data is unavailable here" instead of both silently
    collapsing to the same 0.0 that oi_norm/d_oi fall back to downstream.
    """
    if "oi_fut" in window.columns:
        raw = window["oi_fut"]
    elif "oi" in window.columns:
        raw = window["oi"]
    else:
        return np.zeros(len(window), dtype=float)
    return (~raw.isna()).to_numpy(dtype=float)


def _oi_basis_interaction(d_oi: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """d_oi * sign(basis_t - basis_{t-1}) (feature/label overhaul Phase 2).

    Distinguishes genuine OI accumulation from short-covering-driven basis moves —
    the same d_oi sign means something different depending on whether the futures
    premium is simultaneously widening or narrowing.
    """
    basis_change = np.zeros_like(basis)
    basis_change[1:] = np.diff(basis)
    return d_oi * np.sign(basis_change)


def _oi_flow_onehot(price_change: np.ndarray, d_oi: np.ndarray) -> np.ndarray:
    """Return [T, 4] one-hot OI-flow classification.

    Classes: long_buildup(0), short_covering(1), short_buildup(2), long_unwinding(3).
    """
    out = np.zeros((len(price_change), 4), dtype=np.float32)
    price_up = price_change > 0
    oi_up = d_oi > 0
    out[:, 0] = (price_up & oi_up).astype(float)      # long buildup
    out[:, 1] = (price_up & ~oi_up).astype(float)     # short covering
    out[:, 2] = (~price_up & oi_up).astype(float)     # short buildup
    out[:, 3] = (~price_up & ~oi_up).astype(float)    # long unwinding
    return out


@dataclass
class FuturesWindowQuality:
    """Eligibility summary for one futures lookback window."""

    eligible: bool
    reasons: tuple[str, ...]
    n_rows: int


@dataclass
class FuturesWindowBuilder:
    """Build [T, FUTURES_FEATURE_DIM] from the assembled parquet window.

    Usage::

        builder = FuturesWindowBuilder.from_parquet("data/processed/banknifty_5min.parquet")
        arr = builder.build_window(ts=datetime(2023, 6, 15, 10, 0), lookback=96)
        # arr.shape == (96, 14)
    """

    _df: pd.DataFrame = field(repr=False)
    _sigma_cache: dict[int, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _segment_end_cache: dict[object, date] | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_parquet(cls, path: str) -> FuturesWindowBuilder:
        """Load the assembled parquet and return a builder."""
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return cls(_df=df.sort_index())

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> FuturesWindowBuilder:
        """Wrap a pre-loaded DataFrame (columns from assemble_data.py)."""
        out = df.copy()
        out.index = pd.to_datetime(out.index)
        return cls(_df=out.sort_index())

    def _segment_end_dates(self) -> dict[object, date]:
        cached = self._segment_end_cache
        if cached is not None:
            return cached
        if "segment_id" not in self._df.columns:
            self._segment_end_cache = {}
            return {}
        mapping: dict[object, date] = {}
        for segment_id, frame in self._df.groupby("segment_id", sort=False):
            if frame.empty:
                continue
            mapping[segment_id] = pd.Timestamp(frame.index[-1]).date()
        self._segment_end_cache = mapping
        return mapping

    def _dte_roll_features(self, window: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return DTE/roll features using Upstox-derived segment metadata when present."""
        ts_dates: list[date] = [idx.date() for idx in window.index]
        if "segment_id" not in window.columns:
            dte_vals = np.array([_dte_norm(d) for d in ts_dates], dtype=float)
            roll_vals = np.array([float(_roll_flag(d)) for d in ts_dates], dtype=float)
            return dte_vals, roll_vals

        segment_ends = self._segment_end_dates()
        dte_days: list[float] = []
        roll_flags: list[float] = []
        for row_date, segment_id in zip(ts_dates, window["segment_id"].to_numpy(), strict=False):
            end_date = segment_ends.get(segment_id)
            if end_date is None:
                dte_days.append(0.0)
                roll_flags.append(0.0)
                continue
            days_left = max((end_date - row_date).days, 0)
            dte_days.append(min(days_left, 30) / 30.0)
            roll_flags.append(1.0 if days_left <= ROLL_FLAG_DAYS else 0.0)
        return np.asarray(dte_days, dtype=float), np.asarray(roll_flags, dtype=float)

    def _feature_matrix(self, window: pd.DataFrame) -> np.ndarray:
        if window.empty:
            return np.zeros((0, FUTURES_FEATURE_DIM), dtype=np.float32)

        close_fut = window["close_fut"].to_numpy(dtype=float) if "close_fut" in window else (
            window["close"].to_numpy(dtype=float)
        )
        close_spot = window.get("close_spot", pd.Series(close_fut)).to_numpy(dtype=float)

        basis = np.where(close_spot != 0, (close_fut - close_spot) / close_spot, 0.0)

        oi_raw = window.get(
            "oi_fut",
            window.get("oi", pd.Series(np.zeros(len(window)))),
        ).to_numpy(dtype=float)
        oi_mean = pd.Series(oi_raw).rolling(_OI_NORM_WINDOW, min_periods=1).mean().to_numpy(dtype=float)
        oi_norm = oi_raw / (oi_mean + 1e-8)

        d_oi_raw = np.zeros(len(window), dtype=float)
        d_oi_raw[1:] = np.diff(oi_raw)
        prev_oi = np.roll(oi_raw, 1)
        prev_oi[0] = oi_raw[0]
        d_oi = np.where(prev_oi != 0, d_oi_raw / (np.abs(prev_oi) + 1.0), 0.0)
        d_oi = np.clip(d_oi, -1.0, 1.0)

        vol_raw = window.get(
            "volume_fut",
            window.get("volume", pd.Series(np.ones(len(window)))),
        ).to_numpy(dtype=float)
        vol_z = volume_zscore(vol_raw, _VOL_Z_WINDOW)

        cal_spread = _calendar_spread(window, close_fut)
        dte_vals, roll_vals = self._dte_roll_features(window)
        d_oi_mag = np.abs(d_oi)
        oi_available = _oi_availability(window)
        oi_basis_interaction = _oi_basis_interaction(d_oi, basis)

        price_change = np.zeros(len(window), dtype=float)
        price_change[1:] = np.diff(close_fut)
        flow = _oi_flow_onehot(price_change, d_oi)

        scalars = np.stack(
            [
                basis, oi_norm, d_oi, vol_z, cal_spread, dte_vals, roll_vals,
                d_oi_mag, oi_available, oi_basis_interaction,
            ],
            axis=1,
        )
        feats = np.concatenate([scalars, flow], axis=1).astype(np.float32)
        return np.nan_to_num(feats, nan=0.0)

    def _close_fut(self) -> np.ndarray:
        if "close_fut" in self._df:
            return self._df["close_fut"].to_numpy(dtype=float)
        return self._df["close"].to_numpy(dtype=float)

    @staticmethod
    def _price_column(frame: pd.DataFrame, name: str) -> str | None:
        fut_name = f"{name}_fut"
        if fut_name in frame.columns:
            return fut_name
        if name in frame.columns:
            return name
        return None

    @classmethod
    def _invalid_ohlc_mask(cls, frame: pd.DataFrame) -> np.ndarray:
        cols = {name: cls._price_column(frame, name) for name in ("open", "high", "low", "close")}
        if any(value is None for value in cols.values()):
            return np.zeros(len(frame), dtype=bool)
        open_ = frame[cols["open"]].to_numpy(dtype=float)  # type: ignore[index]
        high = frame[cols["high"]].to_numpy(dtype=float)  # type: ignore[index]
        low = frame[cols["low"]].to_numpy(dtype=float)  # type: ignore[index]
        close = frame[cols["close"]].to_numpy(dtype=float)  # type: ignore[index]
        finite = np.isfinite(open_) & np.isfinite(high) & np.isfinite(low) & np.isfinite(close)
        positive = (open_ > 0.0) & (high > 0.0) & (low > 0.0) & (close > 0.0)
        ordered = (high >= low) & (high >= np.maximum(open_, close)) & (low <= np.minimum(open_, close))
        return ~(finite & positive & ordered)

    @classmethod
    def _stale_price_mask(cls, frame: pd.DataFrame) -> np.ndarray:
        close_col = cls._price_column(frame, "close")
        if close_col is None or len(frame) < _STALE_RUN_BARS:
            return np.zeros(len(frame), dtype=bool)
        volume_col = "volume_fut" if "volume_fut" in frame.columns else ("volume" if "volume" in frame.columns else None)
        close = frame[close_col].to_numpy(dtype=float)
        if volume_col is None:
            zero_volume = np.zeros(len(frame), dtype=bool)
        else:
            zero_volume = frame[volume_col].to_numpy(dtype=float) <= 0.0
        unchanged = np.zeros(len(frame), dtype=bool)
        unchanged[1:] = np.isclose(np.diff(close), 0.0, atol=0.0)
        stale = np.zeros(len(frame), dtype=bool)
        run = 0
        for idx, same in enumerate(unchanged & zero_volume):
            run = run + 1 if same else 0
            if run >= _STALE_RUN_BARS - 1:
                stale[max(0, idx - _STALE_RUN_BARS + 1) : idx + 1] = True
        return stale

    def quality_for_window(
        self,
        ts: datetime,
        lookback: int,
        *,
        require_full_lookback: bool = True,
    ) -> FuturesWindowQuality:
        """Return strict sample eligibility for the futures lookback ending at ``ts``."""
        end = self._df.index.searchsorted(ts, side="right")
        start = max(0, end - lookback)
        window = self._df.iloc[start:end]
        reasons: list[str] = []
        if require_full_lookback and len(window) < lookback:
            reasons.append("incomplete_lookback")
        if window.empty:
            reasons.append("empty_window")
            return FuturesWindowQuality(False, tuple(reasons), 0)
        if window.index.has_duplicates:
            reasons.append("duplicate_timestamps")
        if not window.index.is_monotonic_increasing:
            reasons.append("non_monotonic_timestamps")
        if len(window) > 1:
            index_series = window.index.to_series()
            deltas = index_series.diff()
            same_session = index_series.dt.date == index_series.shift(1).dt.date
            intraday_deltas = deltas[same_session.fillna(False)]
            if bool((intraday_deltas.dropna() != _EXPECTED_BAR_DELTA).any()):
                reasons.append("non_contiguous_5min")
        if "segment_id" in window.columns and window["segment_id"].nunique(dropna=False) > 1:
            reasons.append("cross_segment_window")
        if "roll_gap" in window.columns and bool(window["roll_gap"].fillna(False).astype(bool).any()):
            reasons.append("roll_gap")
        if bool(self._invalid_ohlc_mask(window).any()):
            reasons.append("invalid_ohlc")
        if bool(self._stale_price_mask(window).any()):
            reasons.append("stale_price_zero_volume")
        return FuturesWindowQuality(not reasons, tuple(reasons), len(window))

    def validate_window(self, ts: datetime, lookback: int) -> None:
        quality = self.quality_for_window(ts, lookback)
        if not quality.eligible:
            raise ValueError(
                f"invalid futures lookback at {ts}: reasons={','.join(quality.reasons)} "
                f"n_rows={quality.n_rows} lookback={lookback}"
            )

    def eligible_positions(self, lookback: int) -> np.ndarray:
        """Boolean mask over full futures index: True when the trailing window is clean."""
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        eligible = np.zeros(len(self._df), dtype=bool)
        for pos, ts in enumerate(self._df.index):
            quality = self.quality_for_window(ts.to_pydatetime(), lookback)
            eligible[pos] = quality.eligible
        return eligible

    def _sigma_series(self, vol_span: int) -> np.ndarray:
        cached = self._sigma_cache.get(int(vol_span))
        if cached is not None:
            return cached
        sigma = ewma_barrier_sigma(self._close_fut(), span=vol_span)
        self._sigma_cache[int(vol_span)] = sigma
        return sigma

    def build_history(self) -> tuple[pd.DatetimeIndex, np.ndarray]:
        """Return the full per-bar futures feature history as (index, [T, 13])."""
        feats = self._feature_matrix(self._df)
        return self._df.index, feats

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
            stop_mult=stop_mult,
            target_mult=target_mult,
            vol_span=vol_span,
            horizon_bars=horizon_bars,
            cost_floor_frac=cost_floor_frac,
        )
        rows = barrier_context_series(self._close_fut(), spec=spec)
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
            stop_mult=stop_mult,
            target_mult=target_mult,
            vol_span=vol_span,
            horizon_bars=horizon_bars,
            cost_floor_frac=cost_floor_frac,
        )
        sigma = self._sigma_series(spec.vol_span)[end - 1]
        return barrier_context_from_sigma(float(sigma), spec=spec)

    def build_window(self, ts: datetime, lookback: int, *, strict: bool = False) -> np.ndarray:
        """Return [T=lookback, FUTURES_FEATURE_DIM=14] float32 array at decision time ``ts``.

        Point-in-time safe: only bars with index <= ts are used.  Returns zeros for the
        warm-up region if fewer than ``lookback`` bars are available before ``ts``.
        """
        if strict:
            self.validate_window(ts, lookback)
        end = self._df.index.searchsorted(ts, side="right")
        history = self._df.iloc[:end]
        if history.empty:
            return np.zeros((lookback, FUTURES_FEATURE_DIM), dtype=np.float32)

        feats = self._feature_matrix(history)[-lookback:]

        # Zero-pad if the window is shorter than lookback (start of history warm-up).
        if len(feats) < lookback:
            pad = np.zeros((lookback - len(feats), FUTURES_FEATURE_DIM), dtype=np.float32)
            feats = np.concatenate([pad, feats], axis=0)

        assert feats.shape == (lookback, FUTURES_FEATURE_DIM), feats.shape
        return feats


__all__ = ["FuturesWindowBuilder"]
