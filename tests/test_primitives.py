"""New primitives added by the feature/label engineering overhaul (Phases 1-2):
dmi, variance_ratio, realized_vol_rs, session_boundary_mask, opening_range_position,
first_window_return. The last two carry an explicit look-ahead-safety requirement
(causal within the opening-range window) — tested with a differential mutation test,
the same style of check this project's own C1 finding would have caught.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from helion_risk_world.data.primitives import (
    dmi,
    first_window_return,
    opening_range_position,
    realized_vol_rs,
    session_boundary_mask,
    variance_ratio,
)


def _session_timestamps(n: int, start_hour: int = 9, start_minute: int = 15, step_minutes: int = 5):
    base = datetime(2026, 1, 5, start_hour, start_minute)
    return [base + timedelta(minutes=step_minutes * i) for i in range(n)]


# ── session_boundary_mask ─────────────────────────────────────────────────────

def test_session_boundary_mask_flags_open_and_close_windows() -> None:
    ts = [
        datetime(2026, 1, 5, 9, 15),   # open, excluded
        datetime(2026, 1, 5, 9, 25),   # still within 15min of open, excluded
        datetime(2026, 1, 5, 9, 30),   # just outside, kept
        datetime(2026, 1, 5, 12, 0),   # midday, kept
        datetime(2026, 1, 5, 15, 20),  # within 15min of close, excluded
        datetime(2026, 1, 5, 15, 29),  # last bar, excluded
    ]
    mask = session_boundary_mask(ts, exclude_minutes=15)
    assert mask.tolist() == [True, True, False, False, True, True]


# ── dmi ────────────────────────────────────────────────────────────────────────

def test_dmi_positive_direction_in_a_clean_uptrend() -> None:
    n = 60
    close = 100.0 + np.arange(n, dtype=float) * 0.5
    high = close + 0.2
    low = close - 0.2
    adx, dmi_diff = dmi(high, low, close, window=14)
    assert np.all(np.isnan(adx[:20]))  # compounded warm-up (DX feeds a 2nd rolling mean)
    tail_diff = dmi_diff[~np.isnan(dmi_diff)][-5:]
    assert np.all(tail_diff > 0.0)  # clean uptrend -> +DI dominates
    tail_adx = adx[~np.isnan(adx)][-5:]
    assert np.all(tail_adx > 0.0)


def test_dmi_output_bounds() -> None:
    rng = np.random.default_rng(3)
    n = 80
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.3, n))
    high = close + np.abs(rng.normal(0.1, 0.05, n))
    low = close - np.abs(rng.normal(0.1, 0.05, n))
    adx, dmi_diff = dmi(high, low, close, window=14)
    valid_adx = adx[~np.isnan(adx)]
    valid_diff = dmi_diff[~np.isnan(dmi_diff)]
    assert np.all((valid_adx >= 0.0) & (valid_adx <= 1.0))
    assert np.all((valid_diff >= -1.0) & (valid_diff <= 1.0))


# ── variance_ratio ───────────────────────────────────────────────────────────

def test_variance_ratio_positive_for_strong_trend() -> None:
    n = 200
    close = 100.0 + np.arange(n, dtype=float) * 0.3  # pure trend, no noise
    vr = variance_ratio(close, q=20, window=80)
    valid = vr[~np.isnan(vr)]
    assert len(valid) > 0
    assert np.all(valid > 0.0)  # trending -> VR > 1 -> (VR - 1) > 0


def test_variance_ratio_negative_for_oscillation() -> None:
    n = 200
    t = np.arange(n, dtype=float)
    close = 100.0 + 2.0 * np.sin(t / 3.0)  # tight mean-reverting oscillation
    vr = variance_ratio(close, q=20, window=80)
    valid = vr[~np.isnan(vr)]
    assert len(valid) > 0
    assert np.all(valid < 0.0)  # mean-reverting -> VR < 1 -> (VR - 1) < 0


def test_variance_ratio_output_bounds() -> None:
    rng = np.random.default_rng(11)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.5, 150))
    vr = variance_ratio(close, q=20)
    valid = vr[~np.isnan(vr)]
    assert np.all(valid >= -1.0) and np.all(valid <= 9.0)


# ── realized_vol_rs ──────────────────────────────────────────────────────────

def test_realized_vol_rs_hand_computed_single_bar_window() -> None:
    # A single, hand-computable bar: O=100, H=102, L=99, C=101.
    open_ = np.array([100.0] * 6)
    high = np.array([102.0] * 6)
    low = np.array([99.0] * 6)
    close = np.array([101.0] * 6)
    window = 3
    out = realized_vol_rs(open_, high, low, close, window)
    ho = np.log(102.0 / 100.0)
    hc = np.log(102.0 / 101.0)
    lo_ = np.log(99.0 / 100.0)
    lc = np.log(99.0 / 101.0)
    expected_bar_rs = max(ho * hc + lo_ * lc, 0.0)
    expected = np.sqrt(expected_bar_rs)  # constant bars -> rolling mean == the bar value
    assert out[-1] == pytest.approx(expected, rel=1e-6)
    assert np.isnan(out[: window - 1]).all()


def test_realized_vol_rs_nonnegative() -> None:
    rng = np.random.default_rng(5)
    n = 50
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.2, n))
    high = close + np.abs(rng.normal(0.1, 0.05, n))
    low = close - np.abs(rng.normal(0.1, 0.05, n))
    open_ = close + rng.normal(0.0, 0.05, n)
    out = realized_vol_rs(open_, high, low, close, window=10)
    valid = out[~np.isnan(out)]
    assert np.all(valid >= 0.0)


# ── opening_range_position / first_window_return: causality ──────────────────

def test_opening_range_position_is_causal_within_window() -> None:
    """A later bar's H/L inside the opening-range window must not change an earlier
    bar's feature value — the exact leak shape this project's own C1 finding covers."""
    ts = _session_timestamps(6)  # 09:15, 09:20, 09:25, 09:30, 09:35, 09:40
    close = np.array([100.0, 100.1, 100.2, 100.3, 100.4, 100.5])
    high_a = np.array([100.1, 100.2, 100.3, 100.4, 100.5, 100.6])
    low_a = np.array([99.9, 100.0, 100.1, 100.2, 100.3, 100.4])
    out_a = opening_range_position(close, high_a, low_a, ts, window_minutes=15)

    # Mutate bar 2 (09:25, still inside the opening window) to a huge spike.
    high_b = high_a.copy()
    high_b[2] = 500.0
    out_b = opening_range_position(close, high_b, low_a, ts, window_minutes=15)

    # Bars 0 and 1 (before the mutated bar) must be unchanged.
    assert out_a[0] == pytest.approx(out_b[0])
    assert out_a[1] == pytest.approx(out_b[1])
    # Bar 2 itself and everything after (still in/after the window) may differ.
    assert out_a[2] != pytest.approx(out_b[2])


def test_first_window_return_is_causal_and_freezes_after_window() -> None:
    ts = _session_timestamps(6)  # 09:15 .. 09:40, in/out of a 15-min window
    close_a = np.array([100.0, 100.5, 101.0, 101.5, 102.0, 102.5])
    out_a = first_window_return(close_a, ts, window_minutes=15)

    close_b = close_a.copy()
    close_b[2] = 999.0  # mutate the last in-window bar (09:25)
    out_b = first_window_return(close_b, ts, window_minutes=15)

    assert out_a[0] == pytest.approx(out_b[0])
    assert out_a[1] == pytest.approx(out_b[1])

    # After the window elapses (bars 3-5, 09:30+), the value must be frozen at the
    # last in-window bar's return, not keep evolving with later closes.
    assert out_a[3] == pytest.approx(out_a[2])
    assert out_a[4] == pytest.approx(out_a[2])
    assert out_a[5] == pytest.approx(out_a[2])


def test_first_window_return_resets_across_day_boundary() -> None:
    ts = _session_timestamps(3) + _session_timestamps(3, start_hour=9, start_minute=15)
    ts[3] = ts[3].replace(day=6)
    ts[4] = ts[4].replace(day=6)
    ts[5] = ts[5].replace(day=6)
    close = np.array([100.0, 101.0, 102.0, 200.0, 202.0, 204.0])
    out = first_window_return(close, ts, window_minutes=15)
    # Day 2's first bar resets relative to its own open (200.0), not day 1's.
    assert out[3] == pytest.approx(0.0, abs=1e-9)
