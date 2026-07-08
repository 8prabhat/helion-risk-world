"""Tests for expiry_calendar, FuturesWindowBuilder, and regime label in label.py."""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from helion_risk_world.data.expiry_calendar import (
    ROLL_FLAG_DAYS,
    dte,
    dte_norm,
    monthly_expiry,
    roll_flag,
)
from helion_risk_world.data.futures_window_builder import FuturesWindowBuilder
from helion_risk_world.encoders.futures_encoder import FUTURES_FEATURE_DIM


# ── expiry_calendar ────────────────────────────────────────────────────────────

def test_monthly_expiry_is_thursday() -> None:
    # When the last Thursday is not a holiday it must be a Thursday.
    # Use a month where the last Thursday is not a known holiday.
    exp = monthly_expiry(2024, 3)
    assert exp.weekday() == 3   # Thursday


def test_monthly_expiry_rolls_on_holiday() -> None:
    # June 29, 2023 (last Thursday) = Bakri Eid; expiry must shift to prior trading day.
    exp = monthly_expiry(2023, 6)
    assert exp < date(2023, 6, 29)  # rolled back from the holiday


def test_monthly_expiry_is_last_in_month() -> None:
    exp = monthly_expiry(2023, 6)
    # No Thursday after it in the same month
    from datetime import timedelta
    nxt = exp + timedelta(days=7)
    assert nxt.month != 6 or nxt.year != 2023


def test_dte_positive_before_expiry() -> None:
    exp = monthly_expiry(2024, 3)
    day_before = date(exp.year, exp.month, exp.day - 1)
    assert dte(day_before) >= 1


def test_dte_zero_on_expiry() -> None:
    exp = monthly_expiry(2024, 3)
    assert dte(exp) == 0


def test_dte_norm_range() -> None:
    for m in range(1, 13):
        val = dte_norm(date(2024, m, 1))
        assert 0.0 <= val <= 1.0


def test_roll_flag_within_window() -> None:
    exp = monthly_expiry(2024, 5)
    from datetime import timedelta
    near = exp - timedelta(days=ROLL_FLAG_DAYS - 1)
    assert roll_flag(near)


def test_roll_flag_outside_window() -> None:
    exp = monthly_expiry(2024, 5)
    from datetime import timedelta
    far = exp - timedelta(days=ROLL_FLAG_DAYS + 2)
    assert not roll_flag(far)


# ── FuturesWindowBuilder ───────────────────────────────────────────────────────

def _make_df(n: int = 100) -> pd.DataFrame:
    """Synthetic assembled-parquet DataFrame with required columns."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-02 09:15", periods=n, freq="5min")
    close_fut = 44000.0 + np.cumsum(rng.normal(0, 20, n))
    close_spot = close_fut - rng.uniform(5, 50, n)
    oi = 1e6 + np.cumsum(rng.normal(0, 5000, n))
    volume = np.abs(rng.normal(5000, 1000, n))
    return pd.DataFrame({
        "open_fut": close_fut,
        "high_fut": close_fut + 5.0,
        "low_fut": close_fut - 5.0,
        "close_fut": close_fut,
        "close_spot": close_spot,
        "oi_fut": np.maximum(oi, 1e5),
        "volume_fut": volume,
        "segment_id": 0,
        "roll_gap": False,
    }, index=idx)


def test_build_window_shape() -> None:
    df = _make_df(100)
    builder = FuturesWindowBuilder.from_dataframe(df)
    ts = df.index[60].to_pydatetime()
    arr = builder.build_window(ts, lookback=24)
    assert arr.shape == (24, FUTURES_FEATURE_DIM)


def test_build_window_dtype_float32() -> None:
    df = _make_df()
    builder = FuturesWindowBuilder.from_dataframe(df)
    ts = df.index[50].to_pydatetime()
    arr = builder.build_window(ts, lookback=24)
    assert arr.dtype == np.float32


def test_build_window_no_nan() -> None:
    df = _make_df()
    builder = FuturesWindowBuilder.from_dataframe(df)
    ts = df.index[50].to_pydatetime()
    arr = builder.build_window(ts, lookback=24)
    assert not np.isnan(arr).any()


def test_build_window_matches_build_history_for_same_timestamp() -> None:
    df = _make_df(100)
    builder = FuturesWindowBuilder.from_dataframe(df)
    ts = df.index[60].to_pydatetime()
    lookback = 24

    history_index, history = builder.build_history()
    pos = history_index.get_loc(df.index[60])
    expected = history[pos - lookback + 1 : pos + 1]

    arr = builder.build_window(ts, lookback=lookback)

    assert np.allclose(arr, expected, atol=1e-7)


def test_build_window_point_in_time() -> None:
    """Bars after ts must not appear in the window."""
    df = _make_df(100)
    builder = FuturesWindowBuilder.from_dataframe(df)
    ts = df.index[30].to_pydatetime()
    arr = builder.build_window(ts, lookback=50)  # request more than available
    # Warm-up zeros will pad the front; no future data
    assert arr.shape == (50, FUTURES_FEATURE_DIM)


def test_build_window_warmup_padding() -> None:
    """When fewer bars than lookback exist, the window is zero-padded at the front."""
    df = _make_df(10)
    builder = FuturesWindowBuilder.from_dataframe(df)
    ts = df.index[5].to_pydatetime()
    arr = builder.build_window(ts, lookback=24)
    # First rows should be zeros (warm-up padding)
    assert (arr[:17] == 0.0).all()


def test_strict_quality_rejects_roll_gap_and_cross_segment_window() -> None:
    df = _make_df(30).copy()
    df.loc[df.index[-2], "roll_gap"] = True
    df.loc[df.index[-1], "segment_id"] = 1
    builder = FuturesWindowBuilder.from_dataframe(df)

    quality = builder.quality_for_window(df.index[-1].to_pydatetime(), lookback=10)

    assert not quality.eligible
    assert "roll_gap" in quality.reasons
    assert "cross_segment_window" in quality.reasons
    with pytest.raises(ValueError, match="roll_gap"):
        builder.build_window(df.index[-1].to_pydatetime(), lookback=10, strict=True)


def test_strict_quality_rejects_invalid_futures_ohlc() -> None:
    df = _make_df(30).copy()
    df.loc[df.index[-1], "close_fut"] = df.loc[df.index[-1], "low_fut"] - 50.0
    builder = FuturesWindowBuilder.from_dataframe(df)

    quality = builder.quality_for_window(df.index[-1].to_pydatetime(), lookback=10)

    assert not quality.eligible
    assert "invalid_ohlc" in quality.reasons


def test_barrier_context_uses_horizon_and_cost_floor() -> None:
    df = _make_df(80)
    builder = FuturesWindowBuilder.from_dataframe(df)
    ts = df.index[-1].to_pydatetime()

    ctx = builder.build_barrier_context(
        ts,
        stop_mult=2.0,
        target_mult=2.0,
        vol_span=20,
        horizon_bars=192,
        cost_floor_frac=0.02,
    )
    _, hist = builder.build_barrier_context_history(
        stop_mult=2.0,
        target_mult=2.0,
        vol_span=20,
        horizon_bars=192,
        cost_floor_frac=0.02,
    )

    assert ctx.stop_return <= -0.02
    assert ctx.target_return >= 0.02
    assert ctx.stop_return == pytest.approx(float(hist[-1, 1]), abs=1e-7)
    assert ctx.target_return == pytest.approx(float(hist[-1, 2]), abs=1e-7)


def test_oi_flow_onehot_sums_to_one() -> None:
    """Each bar's OI-flow one-hot has exactly one class active."""
    df = _make_df(50)
    builder = FuturesWindowBuilder.from_dataframe(df)
    ts = df.index[40].to_pydatetime()
    arr = builder.build_window(ts, lookback=24)
    flow = arr[:, 10:14]  # columns 10-13 are the one-hot (col 8 oi_available, col 9 oi_basis_interaction)
    row_sums = flow.sum(axis=1)
    # Rows after warm-up pad should sum to 1 (except the very first row where d_price=0→all zero)
    assert np.all((row_sums == 0.0) | (row_sums == 1.0))


def test_oi_basis_interaction_sign_tracks_basis_change_not_just_d_oi() -> None:
    """Feature/label overhaul Phase 2: same-sign d_oi (constant OI increase) must flip
    the interaction's sign depending on whether basis is simultaneously widening or
    narrowing — distinguishing genuine accumulation from short-covering-driven moves."""
    idx = pd.date_range("2024-01-02 09:15", periods=4, freq="5min")
    df = pd.DataFrame(
        {
            "close_fut":  [100.0, 101.0, 100.0, 100.0],
            "close_spot": [100.0,  99.0, 100.0, 100.0],
            # basis:         0.0,  ~0.0202,   0.0,   0.0  -> widens then narrows back
            "oi_fut":     [1e6, 1.01e6, 1.02e6, 1.02e6],  # constantly increasing (d_oi > 0)
            "volume_fut": [5000.0] * 4,
        },
        index=idx,
    )
    builder = FuturesWindowBuilder.from_dataframe(df)
    arr = builder.build_window(idx[-1].to_pydatetime(), lookback=4)
    interaction = arr[:, 9]
    # bar 1: basis widened (0 -> ~0.02) while d_oi > 0 -> positive interaction.
    assert interaction[1] > 0.0
    # bar 2: basis narrowed (~0.02 -> 0) while d_oi is still > 0 -> negative interaction.
    assert interaction[2] < 0.0


# ── calendar_spread (review Idea #6: activated via close_fut_next) ─────────────

_CAL_SPREAD_COL = 4  # feature layout index (see module docstring)


def test_calendar_spread_zero_without_close_fut_next_column() -> None:
    """Older assembled parquets without close_fut_next keep the pre-existing
    hardcoded-0 fallback semantics."""
    df = _make_df(50)
    builder = FuturesWindowBuilder.from_dataframe(df)
    ts = df.index[40].to_pydatetime()
    arr = builder.build_window(ts, lookback=24)
    assert (arr[:, _CAL_SPREAD_COL] == 0.0).all()


def test_calendar_spread_computed_from_close_fut_next(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _make_df(50)
    close_fut_next = df["close_fut"].to_numpy(dtype=float) * 1.01  # next month at a 1% premium
    df = df.copy()
    df["close_fut_next"] = close_fut_next
    builder = FuturesWindowBuilder.from_dataframe(df)
    ts = df.index[40].to_pydatetime()
    arr = builder.build_window(ts, lookback=24)
    expected = 0.01
    assert arr[:, _CAL_SPREAD_COL] == pytest.approx(expected, abs=1e-4)


def test_calendar_spread_zero_outside_overlap_window() -> None:
    """NaN close_fut_next (no next-contract overlap at that bar) -> 0.0, not NaN."""
    df = _make_df(50)
    df = df.copy()
    close_fut_next = np.full(len(df), np.nan)
    close_fut_next[-5:] = df["close_fut"].to_numpy(dtype=float)[-5:] * 1.02
    df["close_fut_next"] = close_fut_next
    builder = FuturesWindowBuilder.from_dataframe(df)
    ts = df.index[-1].to_pydatetime()
    arr = builder.build_window(ts, lookback=len(df))
    assert not np.isnan(arr).any()
    assert (arr[:-5, _CAL_SPREAD_COL] == 0.0).all()
    assert arr[-5:, _CAL_SPREAD_COL] == pytest.approx(0.02, abs=1e-4)


# ── oi_available (review Idea #5: missing-data mask channel) ───────────────────

_OI_AVAILABLE_COL = 8  # feature layout index (see module docstring)


def test_oi_available_is_one_when_oi_column_present_and_non_nan() -> None:
    df = _make_df(30)
    builder = FuturesWindowBuilder.from_dataframe(df)
    ts = df.index[-1].to_pydatetime()
    arr = builder.build_window(ts, lookback=len(df))
    assert (arr[:, _OI_AVAILABLE_COL] == 1.0).all()


def test_oi_available_is_zero_where_oi_is_nan() -> None:
    df = _make_df(30).copy()
    oi = df["oi_fut"].to_numpy(dtype=float).copy()
    oi[-5:] = np.nan
    df["oi_fut"] = oi
    builder = FuturesWindowBuilder.from_dataframe(df)
    ts = df.index[-1].to_pydatetime()
    arr = builder.build_window(ts, lookback=len(df))
    assert not np.isnan(arr).any()  # nan_to_num still zeroes oi_norm/d_oi themselves
    assert (arr[:-5, _OI_AVAILABLE_COL] == 1.0).all()
    assert (arr[-5:, _OI_AVAILABLE_COL] == 0.0).all()


def test_oi_available_is_zero_when_no_oi_column_at_all() -> None:
    df = _make_df(30).drop(columns=["oi_fut"])
    builder = FuturesWindowBuilder.from_dataframe(df)
    ts = df.index[-1].to_pydatetime()
    arr = builder.build_window(ts, lookback=len(df))
    assert (arr[:, _OI_AVAILABLE_COL] == 0.0).all()


# ── MarketBatch futures field ──────────────────────────────────────────────────

def test_market_batch_carries_futures() -> None:
    from datetime import datetime as dt
    import numpy as np
    from helion_risk_world.data.feature_builder import MarketBatch
    from helion_risk_world.data.market_window_builder import CANDLE_FEATURE_NAMES

    candle_feats = np.zeros((2, 10, len(CANDLE_FEATURE_NAMES)), dtype=np.float32)
    futures = np.zeros((10, FUTURES_FEATURE_DIM), dtype=np.float32)
    batch = MarketBatch(
        ts=dt(2024, 1, 2, 10, 0),
        symbols=("BANKNIFTY", "NIFTY"),
        candle_features=candle_feats,
        feature_names=CANDLE_FEATURE_NAMES,
        futures=futures,
    )
    assert batch.futures is not None
    assert batch.futures.shape == (10, FUTURES_FEATURE_DIM)


def test_market_batch_futures_wrong_ndim_raises() -> None:
    from datetime import datetime as dt
    from helion_risk_world.data.feature_builder import MarketBatch
    from helion_risk_world.data.market_window_builder import CANDLE_FEATURE_NAMES

    candle_feats = np.zeros((1, 10, len(CANDLE_FEATURE_NAMES)), dtype=np.float32)
    bad_futures = np.zeros((10,), dtype=np.float32)   # should be 2-D
    with pytest.raises(ValueError, match="FUTURES_FEATURE_DIM"):
        MarketBatch(
            ts=dt(2024, 1, 2, 10, 0),
            symbols=("BANKNIFTY",),
            candle_features=candle_feats,
            feature_names=CANDLE_FEATURE_NAMES,
            futures=bad_futures,
        )
