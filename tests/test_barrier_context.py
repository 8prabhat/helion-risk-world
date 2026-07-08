"""barrier_context.py cost-floor behavior (feature/label overhaul Phase 1).

A purely vol-scaled barrier (cost_floor_frac=0.0, the old default) can be arbitrarily
tiny in a low-vol regime, well below round-trip transaction cost. cost_floor_frac
guarantees the barrier never resolves finer than that floor.
"""

from __future__ import annotations

import numpy as np
import pytest

from helion_risk_world.barrier_context import (
    BarrierSpec,
    barrier_context_from_sigma,
    barrier_context_series,
)


def test_cost_floor_binds_in_low_sigma_regime() -> None:
    spec = BarrierSpec(stop_mult=2.0, target_mult=2.0, cost_floor_frac=0.01)
    ctx = barrier_context_from_sigma(0.0001, spec=spec)  # 2*sigma = 0.0002, far below floor
    assert ctx.stop_return == pytest.approx(-0.01)
    assert ctx.target_return == pytest.approx(0.01)


def test_cost_floor_does_not_bind_in_high_sigma_regime() -> None:
    spec = BarrierSpec(stop_mult=2.0, target_mult=2.0, cost_floor_frac=0.01)
    ctx = barrier_context_from_sigma(0.02, spec=spec)  # 2*sigma = 0.04, above floor
    assert ctx.stop_return == pytest.approx(-0.04)
    assert ctx.target_return == pytest.approx(0.04)


def test_cost_floor_symmetric() -> None:
    spec = BarrierSpec(stop_mult=1.5, target_mult=3.0, cost_floor_frac=0.01)
    ctx = barrier_context_from_sigma(0.0001, spec=spec)
    assert ctx.stop_return == pytest.approx(-0.01)
    assert ctx.target_return == pytest.approx(0.01)


def test_zero_cost_floor_preserves_old_behavior() -> None:
    spec = BarrierSpec(stop_mult=2.0, target_mult=2.0, cost_floor_frac=0.0)
    ctx = barrier_context_from_sigma(0.0001, spec=spec)
    assert ctx.stop_return == pytest.approx(-0.0002)
    assert ctx.target_return == pytest.approx(0.0002)


def test_negative_cost_floor_rejected() -> None:
    with pytest.raises(ValueError):
        BarrierSpec(cost_floor_frac=-0.01)


def test_horizon_bars_scales_barrier_width_by_sqrt_horizon() -> None:
    spec1 = BarrierSpec(stop_mult=2.0, target_mult=2.0, horizon_bars=1)
    spec4 = BarrierSpec(stop_mult=2.0, target_mult=2.0, horizon_bars=4)
    ctx1 = barrier_context_from_sigma(0.001, spec=spec1)
    ctx4 = barrier_context_from_sigma(0.001, spec=spec4)
    # sqrt(4) = 2 -> horizon_bars=4 barrier width is exactly 2x horizon_bars=1's.
    assert ctx4.stop_return == pytest.approx(2.0 * ctx1.stop_return)
    assert ctx4.target_return == pytest.approx(2.0 * ctx1.target_return)


def test_default_horizon_bars_preserves_prior_behavior() -> None:
    spec = BarrierSpec(stop_mult=2.0, target_mult=2.0)
    ctx = barrier_context_from_sigma(0.0001, spec=spec)
    assert ctx.stop_return == pytest.approx(-0.0002)
    assert ctx.target_return == pytest.approx(0.0002)


def test_negative_horizon_bars_rejected() -> None:
    with pytest.raises(ValueError):
        BarrierSpec(horizon_bars=0)


def test_barrier_context_series_scales_by_horizon_bars() -> None:
    close = np.full(60, 100.0)
    close[30:] += np.linspace(0.0, 0.05, 30)
    spec1 = BarrierSpec(stop_mult=2.0, target_mult=2.0, horizon_bars=1)
    spec9 = BarrierSpec(stop_mult=2.0, target_mult=2.0, horizon_bars=9)
    rows1 = barrier_context_series(close, spec=spec1)
    rows9 = barrier_context_series(close, spec=spec9)
    # sqrt(9) = 3.
    np.testing.assert_allclose(rows9[:, 1], 3.0 * rows1[:, 1], atol=1e-9)
    np.testing.assert_allclose(rows9[:, 2], 3.0 * rows1[:, 2], atol=1e-9)


def test_barrier_context_series_applies_floor_per_bar() -> None:
    # A near-flat close series produces a small EWMA sigma throughout.
    close = np.full(60, 100.0)
    close[30:] += np.linspace(0.0, 0.05, 30)  # tiny drift, still low vol
    spec = BarrierSpec(stop_mult=2.0, target_mult=2.0, cost_floor_frac=0.02)
    rows = barrier_context_series(close, spec=spec)
    stop_returns = rows[:, 1]
    target_returns = rows[:, 2]
    assert np.all(np.abs(stop_returns) >= 0.02 - 1e-6)
    assert np.all(target_returns >= 0.02 - 1e-6)
