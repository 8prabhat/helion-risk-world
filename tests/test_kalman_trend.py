"""local_linear_trend_filter (feature/label overhaul Phase 3): the only stateful,
recursive primitive in the feature set. Tests focus on (a) basic filtering sanity
(converges toward a known drift, non-degenerate on flat input), and (b) the
segment-reset behavior — the highest-risk part of this module, since a missed reset
would let recursive belief silently bridge a genuine data gap (the same H3/H4 bug
class this project's review already found twice elsewhere).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from helion_risk_world.data.kalman_trend import local_linear_trend_filter


def _timestamps(n: int, step_minutes: int = 5, start: datetime | None = None) -> list[datetime]:
    base = start or datetime(2026, 1, 5, 9, 15)
    return [base + timedelta(minutes=step_minutes * i) for i in range(n)]


def test_trend_converges_toward_known_drift() -> None:
    rng = np.random.default_rng(0)
    n = 400
    true_drift = 0.001  # log-price units per bar
    noise = rng.normal(0.0, 0.0005, n)
    log_price = np.cumsum(np.full(n, true_drift) + noise)
    close = np.exp(log_price) * 100.0
    ts = _timestamps(n)

    trend, _innovation, _uncertainty = local_linear_trend_filter(close, ts)
    tail_trend = trend[-50:]
    assert np.mean(tail_trend) == pytest.approx(true_drift, abs=true_drift * 0.5)


def test_trend_near_zero_for_flat_series() -> None:
    n = 100
    close = np.full(n, 100.0)
    ts = _timestamps(n)
    trend, innovation, uncertainty = local_linear_trend_filter(close, ts)
    assert np.all(np.isfinite(trend))
    assert np.all(np.isfinite(innovation))
    assert np.all(np.isfinite(uncertainty))
    assert np.allclose(trend[-10:], 0.0, atol=1e-6)
    assert np.allclose(innovation, 0.0, atol=1e-6)


def test_output_shape_matches_input() -> None:
    n = 50
    rng = np.random.default_rng(1)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.5, n))
    ts = _timestamps(n)
    trend, innovation, uncertainty = local_linear_trend_filter(close, ts)
    assert trend.shape == (n,)
    assert innovation.shape == (n,)
    assert uncertainty.shape == (n,)


def test_empty_input_returns_empty_arrays() -> None:
    trend, innovation, uncertainty = local_linear_trend_filter(np.array([]), [])
    assert trend.shape == (0,)
    assert innovation.shape == (0,)
    assert uncertainty.shape == (0,)


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError):
        local_linear_trend_filter(np.array([100.0, 101.0, 102.0]), _timestamps(2))


# ── Segment-reset behavior (the highest-risk part of this module) ────────────────

def test_uncertainty_spikes_after_forced_segment_reset_then_decays() -> None:
    """A genuine multi-day gap (well beyond contiguous_segment_ids' default 4-day
    threshold) must trigger a state reset: trend_uncertainty jumps back up to the
    fresh-start level immediately after the gap, then decays again as the filter
    re-converges — proving state did NOT silently bridge the gap."""
    rng = np.random.default_rng(2)
    n_before, n_after = 60, 60
    before_close = 100.0 + np.cumsum(rng.normal(0.0, 0.3, n_before))
    after_close = 500.0 + np.cumsum(rng.normal(0.0, 0.3, n_after))  # unrelated price level
    close = np.concatenate([before_close, after_close])

    ts_before = _timestamps(n_before)
    # A 30-day gap after the last "before" bar — far beyond the 4-day default threshold.
    gap_start = ts_before[-1] + timedelta(days=30)
    ts_after = _timestamps(n_after, start=gap_start)
    ts = ts_before + ts_after

    trend, innovation, uncertainty = local_linear_trend_filter(close, ts)

    # Immediately after the gap: state reset -> trend=0, innovation=0, uncertainty at
    # the fresh-start level (same as bar 0's uncertainty).
    reset_idx = n_before
    assert trend[reset_idx] == pytest.approx(0.0, abs=1e-9)
    assert innovation[reset_idx] == pytest.approx(0.0, abs=1e-9)
    assert uncertainty[reset_idx] == pytest.approx(uncertainty[0], rel=1e-6)

    # Uncertainty must have decayed well below the fresh-start level by the bar just
    # before the gap (the filter had time to converge over 60 bars).
    assert uncertainty[reset_idx - 1] < uncertainty[reset_idx] * 0.9

    # And it decays again after the reset, given enough bars to re-converge.
    assert uncertainty[-1] < uncertainty[reset_idx] * 0.9


def test_no_reset_at_routine_overnight_gap() -> None:
    """An ordinary overnight/weekend gap (well within contiguous_segment_ids' 4-day
    tolerance) must NOT trigger a reset — state should carry across it normally."""
    rng = np.random.default_rng(3)
    n = 40
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.3, n))
    day1 = _timestamps(n // 2)
    day2_start = day1[-1] + timedelta(hours=18)  # overnight gap, not a data gap
    day2 = _timestamps(n - n // 2, start=day2_start)
    ts = day1 + day2

    trend, innovation, uncertainty = local_linear_trend_filter(close, ts)
    boundary = n // 2
    # No reset: uncertainty should NOT jump back up to the fresh-start level at the
    # overnight boundary (it should already have decayed from convergence).
    assert uncertainty[boundary] < uncertainty[0] * 0.9


def test_does_not_reset_at_a_pure_price_jump_with_no_time_gap() -> None:
    """A large price jump with NO time gap (e.g. a backward-adjusted roll boundary)
    must NOT reset state — only genuine TIME gaps do. Backward adjustment already
    makes rolls invisible to this filter by design; resetting on price jumps alone
    would defeat that."""
    n = 60
    close = np.concatenate([np.full(30, 100.0), np.full(30, 50.0)])  # instant halving
    ts = _timestamps(n)  # perfectly regular 5-min bars, no time gap anywhere
    trend, innovation, uncertainty = local_linear_trend_filter(close, ts)
    # No reset means uncertainty keeps decaying through the jump, rather than
    # snapping back to the bar-0 fresh-start level at the jump.
    assert uncertainty[30] < uncertainty[0]
