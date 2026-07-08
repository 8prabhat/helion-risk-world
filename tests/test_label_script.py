from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

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


_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
_SPEC = importlib.util.spec_from_file_location("label_script", _ROOT / "scripts" / "label.py")
assert _SPEC is not None and _SPEC.loader is not None
label_script = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = label_script
_SPEC.loader.exec_module(label_script)


def test_run_labeling_persists_fixed_horizon_targets(tmp_path) -> None:
    index = pd.date_range("2026-01-01 09:15:00", periods=6, freq="5min")
    frame = pd.DataFrame(
        {
            "open_fut": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "high_fut": [100.0, 100.2, 101.5, 102.5, 103.5, 104.5],
            "low_fut": [100.0, 99.8, 100.5, 101.5, 102.5, 103.5],
            "close_fut": [100.0, 100.0, 101.0, 102.0, 103.0, 104.0],
        },
        index=index,
    )
    data_path = tmp_path / "assembled.parquet"
    out_path = tmp_path / "labels.parquet"
    frame.to_parquet(data_path)

    labels = label_script.run_labeling(
        data_path,
        out_path,
        H=3,
        target_horizons=(2, 3),
        stop_mult=0.1,
        target_mult=0.1,
        cost_floor_frac=0.0,
        session_exclude_minutes=0,
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


def test_run_labeling_persists_ambiguous_barrier_rows(tmp_path) -> None:
    index = pd.date_range("2026-01-01 09:15:00", periods=5, freq="5min")
    frame = pd.DataFrame(
        {
            "open_fut": [100.0, 100.0, 100.0, 100.0, 100.0],
            "high_fut": [100.0, 101.0, 100.5, 100.5, 100.5],
            "low_fut": [100.0, 99.0, 99.5, 99.5, 99.5],
            "close_fut": [100.0, 100.0, 100.0, 100.0, 100.0],
        },
        index=index,
    )
    data_path = tmp_path / "assembled_ambiguous.parquet"
    out_path = tmp_path / "labels_ambiguous.parquet"
    frame.to_parquet(data_path)

    labels = label_script.run_labeling(
        data_path,
        out_path,
        H=2,
        target_horizons=(1, 2),
        stop_mult=0.1,
        target_mult=0.1,
        cost_floor_frac=0.0,
        session_exclude_minutes=0,
    )

    first = labels.iloc[0]
    assert first["barrier"] == Barrier.AMBIGUOUS.value
    assert bool(first["barrier_valid"]) is False
    assert first["entry_price"] == pytest.approx(100.0, abs=1e-6)
    assert first["exit_price"] == pytest.approx(100.0, abs=1e-6)


def test_run_labeling_skips_records_crossing_a_contiguity_gap(tmp_path) -> None:
    """Review findings H3, H4, M7: a decision bar whose H-bar-ahead scan window
    would reach past a data gap (e.g. a dropped corporate-action blackout window)
    must be dropped entirely, not silently labeled as if the gap didn't exist."""
    day1 = pd.date_range("2026-01-01 09:15:00", periods=5, freq="5min")
    day2 = pd.date_range("2026-01-20 09:15:00", periods=5, freq="5min")  # ~19-day gap
    index = day1.append(day2)
    close = [100.0] * 10
    frame = pd.DataFrame(
        {"open_fut": close, "high_fut": close, "low_fut": close, "close_fut": close},
        index=index,
    )
    data_path = tmp_path / "assembled_gap.parquet"
    out_path = tmp_path / "labels_gap.parquet"
    frame.to_parquet(data_path)

    labels = label_script.run_labeling(
        data_path,
        out_path,
        H=3,
        target_horizons=(3,),
        stop_mult=0.1,
        target_mult=0.1,
        cost_floor_frac=0.0,
        session_exclude_minutes=0,
    )

    # t=2,3,4 (day1) have an H=3 scan window that reaches into day2 and must be
    # dropped; t=0,1 (fully within day1) and t=5,6 (fully within day2) survive.
    surviving_ts = set(pd.Timestamp(ts) for ts in labels.index)
    assert len(labels) == 4
    for skipped in (2, 3, 4):
        assert index[skipped] not in surviving_ts
    for kept in (0, 1, 5, 6):
        assert index[kept] in surviving_ts


def test_run_labeling_cost_floor_times_out_sub_cost_moves(tmp_path) -> None:
    """Feature/label overhaul Phase 1: a small enough move that a purely vol-scaled
    barrier would call TARGET must instead TIMEOUT once cost_floor_frac is set wide
    enough that the move can't clear it within H bars."""
    index = pd.date_range("2026-01-01 10:00:00", periods=6, freq="5min")
    # A small, steady drift: ~0.3% cumulative move by bar 3 — enough to trip a tiny
    # vol-scaled barrier, not enough to clear a 2% cost floor within H=3 bars.
    close = [100.0, 100.05, 100.15, 100.30, 100.30, 100.30]
    frame = pd.DataFrame(
        {"open_fut": close, "high_fut": close, "low_fut": close, "close_fut": close},
        index=index,
    )
    data_path = tmp_path / "assembled_cost_floor.parquet"
    frame.to_parquet(data_path)

    labels_no_floor = label_script.run_labeling(
        data_path, tmp_path / "labels_no_floor.parquet",
        H=3, target_horizons=(3,), stop_mult=0.05, target_mult=0.05,
        cost_floor_frac=0.0, session_exclude_minutes=0,
    )
    assert labels_no_floor.iloc[0]["barrier"] == Barrier.TARGET.value

    labels_with_floor = label_script.run_labeling(
        data_path, tmp_path / "labels_with_floor.parquet",
        H=3, target_horizons=(3,), stop_mult=0.05, target_mult=0.05,
        cost_floor_frac=0.02, session_exclude_minutes=0,
    )
    assert labels_with_floor.iloc[0]["barrier"] == Barrier.TIMEOUT.value
    assert labels_with_floor.iloc[0][BARRIER_COST_FLOOR_COLUMN] == pytest.approx(0.02)


def test_run_labeling_excludes_session_boundary_bars(tmp_path) -> None:
    """Feature/label overhaul Phase 1: decision bars in the first/last N minutes of
    the NSE session are excluded from labeling entirely."""
    # 09:15 (open) through 09:35: bars at 09:15-09:29 fall in the excluded opening
    # window (exclude_minutes=15); 09:30+ do not.
    index = pd.date_range("2026-01-01 09:15:00", periods=8, freq="5min")
    close = [100.0 + 0.01 * i for i in range(8)]
    frame = pd.DataFrame(
        {"open_fut": close, "high_fut": close, "low_fut": close, "close_fut": close},
        index=index,
    )
    data_path = tmp_path / "assembled_boundary.parquet"
    frame.to_parquet(data_path)

    labels = label_script.run_labeling(
        data_path, tmp_path / "labels_boundary.parquet",
        H=2, target_horizons=(2,), stop_mult=0.1, target_mult=0.1,
        cost_floor_frac=0.0, session_exclude_minutes=15,
    )
    surviving = set(pd.Timestamp(ts) for ts in labels.index)
    for excluded_i in (0, 1, 2):  # 09:15, 09:20, 09:25 -> minute_of_day < 555+15
        assert index[excluded_i] not in surviving
    for kept_i in (3, 4, 5):  # 09:30, 09:35, 09:40 -> outside the excluded window
        assert index[kept_i] in surviving
