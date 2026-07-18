"""Tests for helion_risk_world.data.alpha_labels.build_alpha_labels (Phase 2 migration
-- the adapter that replaced labeling/barrier_labeler.py + labeling/uniqueness.py +
scripts/assemble_data.py's local basis/segment assembly).

Uses the ``raw_ohlcv=`` test seam to construct exact synthetic scenarios, the same way
the pre-migration BarrierLabeler-based tests did via ``--data-path`` -- these are ported
regression tests for specific previously-fixed bugs (ambiguous same-bar dual touch,
cost-floor timeout, session-boundary exclusion, contiguity-gap skip), not new coverage.
"""

from __future__ import annotations

import pandas as pd
import pytest

from helion_risk_world.data.alpha_labels import build_alpha_labels
from helion_risk_world.schemas.label_schema import (
    BARRIER_COST_FLOOR_COLUMN,
    BARRIER_SIGMA_COLUMN,
    BARRIER_STOP_MULT_COLUMN,
    BARRIER_STOP_RETURN_COLUMN,
    BARRIER_TARGET_MULT_COLUMN,
    BARRIER_TARGET_RETURN_COLUMN,
    BARRIER_VOL_SPAN_COLUMN,
    Barrier,
    horizon_return_column,
)


def _ohlc_frame(index, close, open_=None, high=None, low=None) -> pd.DataFrame:
    close = list(close)
    open_ = list(open_) if open_ is not None else close
    high = list(high) if high is not None else close
    low = list(low) if low is not None else close
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=index)


def test_persists_fixed_horizon_targets() -> None:
    index = pd.date_range("2026-01-01 09:15:00", periods=6, freq="5min")
    frame = _ohlc_frame(
        index,
        close=[100.0, 100.0, 101.0, 102.0, 103.0, 104.0],
        high=[100.0, 100.2, 101.5, 102.5, 103.5, 104.5],
        low=[100.0, 99.8, 100.5, 101.5, 102.5, 103.5],
    )

    labels = build_alpha_labels(
        H=3, target_horizons=(2, 3), stop_mult=0.1, target_mult=0.1,
        cost_floor_frac=0.0, session_exclude_minutes=0, raw_ohlcv=frame,
    )

    assert horizon_return_column(2) in labels.columns
    assert horizon_return_column(3) in labels.columns
    first = labels.iloc[0]
    assert first[horizon_return_column(2)] == pytest.approx((101.0 / 100.0) - 1.0, abs=1e-6)
    assert first[horizon_return_column(3)] == pytest.approx((102.0 / 100.0) - 1.0, abs=1e-6)
    assert first["exit_return"] != pytest.approx(first[horizon_return_column(3)], abs=1e-6)
    assert first[BARRIER_SIGMA_COLUMN] > 0.0
    assert first[BARRIER_STOP_RETURN_COLUMN] < 0.0
    assert first[BARRIER_TARGET_RETURN_COLUMN] > 0.0
    assert first[BARRIER_STOP_MULT_COLUMN] == pytest.approx(0.1)
    assert first[BARRIER_TARGET_MULT_COLUMN] == pytest.approx(0.1)
    assert first[BARRIER_VOL_SPAN_COLUMN] == 50


def test_persists_ambiguous_barrier_rows() -> None:
    index = pd.date_range("2026-01-01 09:15:00", periods=5, freq="5min")
    frame = _ohlc_frame(
        index,
        close=[100.0, 100.0, 100.0, 100.0, 100.0],
        high=[100.0, 101.0, 100.5, 100.5, 100.5],
        low=[100.0, 99.0, 99.5, 99.5, 99.5],
    )

    labels = build_alpha_labels(
        H=2, target_horizons=(1, 2), stop_mult=0.1, target_mult=0.1,
        cost_floor_frac=0.0, session_exclude_minutes=0, raw_ohlcv=frame,
    )

    first = labels.iloc[0]
    assert first["barrier"] == Barrier.AMBIGUOUS.value
    assert bool(first["barrier_valid"]) is False
    assert first["entry_price"] == pytest.approx(100.0, abs=1e-6)
    assert first["exit_price"] == pytest.approx(100.0, abs=1e-6)


def test_skips_records_crossing_a_contiguity_gap() -> None:
    """A decision bar whose H-bar-ahead scan window would reach past a data gap
    must be dropped entirely, not silently labeled as if the gap didn't exist."""
    day1 = pd.date_range("2026-01-01 09:15:00", periods=5, freq="5min")
    day2 = pd.date_range("2026-01-20 09:15:00", periods=5, freq="5min")  # ~19-day gap
    index = day1.append(day2)
    frame = _ohlc_frame(index, close=[100.0] * 10)

    labels = build_alpha_labels(
        H=3, target_horizons=(3,), stop_mult=0.1, target_mult=0.1,
        cost_floor_frac=0.0, session_exclude_minutes=0, raw_ohlcv=frame,
    )

    # t=2,3,4 (day1) have an H=3 scan window that reaches into day2 and must be
    # dropped; t=0,1 (fully within day1) and t=5,6 (fully within day2) survive.
    surviving_ts = set(pd.Timestamp(ts) for ts in labels.index)
    assert len(labels) == 4
    for skipped in (2, 3, 4):
        assert index[skipped] not in surviving_ts
    for kept in (0, 1, 5, 6):
        assert index[kept] in surviving_ts


def test_cost_floor_times_out_sub_cost_moves() -> None:
    """A small enough move that a purely vol-scaled barrier would call TARGET must
    instead TIMEOUT once cost_floor_frac is set wide enough that the move can't clear
    it within H bars."""
    index = pd.date_range("2026-01-01 10:00:00", periods=6, freq="5min")
    # A small, steady drift: ~0.3% cumulative move by bar 3 -- enough to trip a tiny
    # vol-scaled barrier, not enough to clear a 2% cost floor within H=3 bars.
    close = [100.0, 100.05, 100.15, 100.30, 100.30, 100.30]
    frame = _ohlc_frame(index, close=close)

    labels_no_floor = build_alpha_labels(
        H=3, target_horizons=(3,), stop_mult=0.05, target_mult=0.05,
        cost_floor_frac=0.0, session_exclude_minutes=0, raw_ohlcv=frame,
    )
    assert labels_no_floor.iloc[0]["barrier"] == Barrier.TARGET.value

    labels_with_floor = build_alpha_labels(
        H=3, target_horizons=(3,), stop_mult=0.05, target_mult=0.05,
        cost_floor_frac=0.02, session_exclude_minutes=0, raw_ohlcv=frame,
    )
    assert labels_with_floor.iloc[0]["barrier"] == Barrier.TIMEOUT.value
    assert labels_with_floor.iloc[0][BARRIER_COST_FLOOR_COLUMN] == pytest.approx(0.02)


def test_excludes_session_boundary_bars() -> None:
    """Decision bars in the first/last N minutes of the NSE session are excluded
    from labeling entirely."""
    # 09:15 (open) through 09:35: bars at 09:15-09:29 fall in the excluded opening
    # window (exclude_minutes=15); 09:30+ do not.
    index = pd.date_range("2026-01-01 09:15:00", periods=8, freq="5min")
    close = [100.0 + 0.01 * i for i in range(8)]
    frame = _ohlc_frame(index, close=close)

    labels = build_alpha_labels(
        H=2, target_horizons=(2,), stop_mult=0.1, target_mult=0.1,
        cost_floor_frac=0.0, session_exclude_minutes=15, raw_ohlcv=frame,
    )
    surviving = set(pd.Timestamp(ts) for ts in labels.index)
    for excluded_i in (0, 1, 2):  # 09:15, 09:20, 09:25 -> minute_of_day < 555+15
        assert index[excluded_i] not in surviving
    for kept_i in (3, 4, 5):  # 09:30, 09:35, 09:40 -> outside the excluded window
        assert index[kept_i] in surviving


def test_target_horizons_cannot_exceed_h() -> None:
    index = pd.date_range("2026-01-01 09:15:00", periods=10, freq="5min")
    frame = _ohlc_frame(index, close=[100.0] * 10)
    with pytest.raises(ValueError, match="cannot exceed"):
        build_alpha_labels(H=3, target_horizons=(5,), raw_ohlcv=frame)


def test_regime_column_present_and_categorical() -> None:
    index = pd.date_range("2026-01-01 09:15:00", periods=20, freq="5min")
    close = [100.0 + 0.02 * i for i in range(20)]
    frame = _ohlc_frame(index, close=close)
    labels = build_alpha_labels(
        H=3, target_horizons=(3,), stop_mult=0.1, target_mult=0.1,
        cost_floor_frac=0.0, session_exclude_minutes=0, raw_ohlcv=frame,
    )
    assert "regime" in labels.columns
    assert labels["regime_source"].eq("point_in_time").all()


def test_meta_label_columns_present_and_consistent_with_primary_side() -> None:
    from helion_risk_world.labeling.meta_labels import meta_label_for_side

    index = pd.date_range("2026-01-01 09:15:00", periods=40, freq="5min")
    # Strong sustained uptrend -> positive trailing momentum -> primary_side long
    # almost everywhere it's computable, and exit_return should clear a zero cost floor.
    close = [100.0 + 0.5 * i for i in range(40)]
    frame = _ohlc_frame(index, close=close)

    labels = build_alpha_labels(
        H=5, target_horizons=(5,), stop_mult=5.0, target_mult=5.0,
        cost_floor_frac=0.0, session_exclude_minutes=0, meta_label_lookback=3,
        raw_ohlcv=frame,
    )
    assert "primary_side" in labels.columns
    assert "meta_label" in labels.columns
    assert set(labels["primary_side"].unique()).issubset({-1, 0, 1})

    computable = labels[labels["primary_side"] != 0]
    assert len(computable) > 0
    for _, row in computable.iterrows():
        expected = meta_label_for_side(
            int(row["primary_side"]), float(row["exit_return"]), float(row[BARRIER_COST_FLOOR_COLUMN])
        )
        assert int(row["meta_label"]) == expected

    # Sustained strong uptrend: primary_side should be long (1) almost everywhere.
    assert (computable["primary_side"] == 1).mean() > 0.8


def test_meta_label_none_becomes_nan_when_primary_side_flat() -> None:
    index = pd.date_range("2026-01-01 09:15:00", periods=10, freq="5min")
    # Perfectly flat price -> momentum is always an exact tie -> primary_side == 0 everywhere.
    frame = _ohlc_frame(index, close=[100.0] * 10)
    labels = build_alpha_labels(
        H=3, target_horizons=(3,), stop_mult=1.0, target_mult=1.0,
        cost_floor_frac=0.0, session_exclude_minutes=0, meta_label_lookback=3,
        raw_ohlcv=frame,
    )
    assert (labels["primary_side"] == 0).all()
    assert labels["meta_label"].isna().all()
