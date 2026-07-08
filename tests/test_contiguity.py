"""contiguous_segment_ids (review findings H3, H4, M7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from helion_risk_world.data.contiguity import DEFAULT_MAX_GAP, contiguous_segment_ids


def test_empty_and_single_row() -> None:
    assert contiguous_segment_ids(pd.DatetimeIndex([])).tolist() == []
    idx = pd.DatetimeIndex(["2024-01-01 09:15"])
    assert contiguous_segment_ids(idx).tolist() == [0]


def test_ordinary_weekend_is_not_flagged_as_a_gap() -> None:
    """A normal Friday-close -> Monday-open transition (~2.75 days) must not split
    the series into separate segments — that would be true of every single week."""
    idx = pd.DatetimeIndex(
        ["2024-01-05 15:25", "2024-01-05 15:30", "2024-01-08 09:15", "2024-01-08 09:20"]
    )
    seg = contiguous_segment_ids(idx)
    assert seg.tolist() == [0, 0, 0, 0]


def test_multi_day_blackout_gap_is_flagged() -> None:
    """A dropped multi-trading-day window (e.g. the HDFC merger blackout, ~10
    trading days) must split the series — bridging it would silently splice
    unrelated market history into one "adjacent" pair of rows."""
    idx = pd.DatetimeIndex(
        ["2023-06-26 15:25", "2023-06-26 15:30", "2023-07-10 09:15", "2023-07-10 09:20"]
    )
    seg = contiguous_segment_ids(idx)
    assert seg.tolist() == [0, 0, 1, 1]


def test_extra_gap_mask_flags_a_roll_bar_even_with_no_time_gap() -> None:
    """A futures-roll bar can be only one normal bar-interval from its neighbor in
    wall-clock time, yet still represents a real contract-price discontinuity."""
    idx = pd.date_range("2024-01-10 09:15", periods=4, freq="5min")
    roll_flag = np.array([False, True, False, False])  # row 1 is the last bar of its contract
    seg = contiguous_segment_ids(idx, extra_gap_mask=roll_flag)
    assert seg.tolist() == [0, 0, 1, 1]


def test_max_gap_is_configurable() -> None:
    idx = pd.date_range("2024-01-10 09:15", periods=3, freq="1h")
    # 1-hour gaps: not flagged under the (generous) default, but flagged under a
    # tight explicit threshold.
    assert contiguous_segment_ids(idx).tolist() == [0, 0, 0]
    assert contiguous_segment_ids(idx, max_gap=pd.Timedelta(minutes=30)).tolist() == [0, 1, 2]


def test_extra_gap_mask_length_mismatch_raises() -> None:
    idx = pd.date_range("2024-01-10 09:15", periods=3, freq="5min")
    with pytest.raises(ValueError):
        contiguous_segment_ids(idx, extra_gap_mask=np.array([False, True]))


def test_default_max_gap_value() -> None:
    assert DEFAULT_MAX_GAP == pd.Timedelta(days=4)
