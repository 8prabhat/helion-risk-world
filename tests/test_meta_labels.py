"""Meta-labeling primary-signal + cost-aware binary label construction (2026-07-18)."""

from __future__ import annotations

import numpy as np
import pytest

from helion_risk_world.labeling.meta_labels import (
    compute_meta_labels,
    meta_label_for_side,
    primary_side_from_close,
    primary_side_from_log_returns,
)


def test_primary_side_long_on_positive_momentum() -> None:
    close = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    assert primary_side_from_close(close, idx=4, lookback=5) == 1


def test_primary_side_short_on_negative_momentum() -> None:
    close = np.array([104.0, 103.0, 102.0, 101.0, 100.0])
    assert primary_side_from_close(close, idx=4, lookback=5) == -1


def test_primary_side_flat_on_exact_tie() -> None:
    close = np.array([100.0, 105.0, 100.0])
    assert primary_side_from_close(close, idx=2, lookback=3) == 0


def test_primary_side_never_reads_future_bars() -> None:
    """Causal check: appending future bars after idx must not change the signal at idx."""
    close = np.array([100.0, 101.0, 102.0, 103.0])
    truncated = close[:3]
    side_full = primary_side_from_close(close, idx=2, lookback=3)
    side_truncated = primary_side_from_close(truncated, idx=2, lookback=3)
    assert side_full == side_truncated

    # Now make the future move violently in the OPPOSITE direction -- if this changed
    # the idx=2 signal, that would prove future leakage.
    close_adversarial = np.array([100.0, 101.0, 102.0, 50.0, 10.0])
    side_adversarial = primary_side_from_close(close_adversarial, idx=2, lookback=3)
    assert side_adversarial == side_full


def test_primary_side_insufficient_history_is_flat() -> None:
    close = np.array([100.0])
    assert primary_side_from_close(close, idx=0, lookback=12) == 0


def test_primary_side_from_log_returns_matches_close_based() -> None:
    close = np.array([100.0, 102.0, 101.0, 105.0])
    log_returns = np.diff(np.log(close))
    side_from_close = primary_side_from_close(close, idx=3, lookback=4)
    side_from_logret = primary_side_from_log_returns(log_returns)
    assert side_from_close == side_from_logret == 1


def test_meta_label_none_when_primary_side_is_flat() -> None:
    assert meta_label_for_side(0, exit_return=0.05, cost_floor_frac=0.001) is None


def test_meta_label_long_profitable_above_cost_floor() -> None:
    # Long primary, price went up 2%, cost floor 0.13% -> clears cost -> label 1.
    assert meta_label_for_side(1, exit_return=0.02, cost_floor_frac=0.0013) == 1


def test_meta_label_long_unprofitable_below_cost_floor() -> None:
    # Long primary, price barely moved (0.05%) -- doesn't clear the cost floor.
    assert meta_label_for_side(1, exit_return=0.0005, cost_floor_frac=0.0013) == 0


def test_meta_label_short_flips_sign_of_exit_return() -> None:
    # Short primary: price FELL 2% (exit_return negative) -> profitable for the short.
    assert meta_label_for_side(-1, exit_return=-0.02, cost_floor_frac=0.0013) == 1
    # Short primary: price ROSE 2% -> a loss for the short, despite a "positive" exit_return.
    assert meta_label_for_side(-1, exit_return=0.02, cost_floor_frac=0.0013) == 0


def test_meta_label_exactly_at_cost_floor_is_not_profitable() -> None:
    """Strict '>' -- a trade that exactly breaks even on cost is not a worthwhile bet."""
    assert meta_label_for_side(1, exit_return=0.0013, cost_floor_frac=0.0013) == 0


def test_compute_meta_labels_batch_matches_row_by_row() -> None:
    close = np.array([100.0, 101.0, 102.5, 101.0, 99.0, 98.0, 100.0])
    exit_return = np.array([0.03, 0.02, -0.01, 0.0004, -0.02, 0.01, 0.0])
    cost_floor = 0.0013
    primary, meta = compute_meta_labels(close, exit_return, cost_floor, lookback=3)
    assert primary.shape == exit_return.shape
    assert meta.shape == exit_return.shape
    for i in range(len(exit_return)):
        expected_side = primary_side_from_close(close, i, lookback=3)
        assert primary[i] == expected_side
        if expected_side == 0:
            assert meta[i] == -1
        else:
            expected_label = meta_label_for_side(expected_side, exit_return[i], cost_floor)
            assert meta[i] == expected_label


def test_compute_meta_labels_accepts_per_row_cost_floor_array() -> None:
    close = np.array([100.0, 101.0, 102.0, 103.0])
    exit_return = np.array([0.01, 0.01, 0.01, 0.01])
    cost_floor = np.array([0.001, 0.02, 0.001, 0.001])  # row 1's floor exceeds the return
    primary, meta = compute_meta_labels(close, exit_return, cost_floor, lookback=2)
    assert primary[1] == 1  # momentum still positive
    assert meta[1] == 0     # but this row's floor (2%) is not cleared by a 1% return


def test_compute_meta_labels_rejects_short_close_array() -> None:
    with pytest.raises(ValueError):
        compute_meta_labels(np.array([1.0, 2.0]), np.array([0.01, 0.02, 0.03]), 0.001)
