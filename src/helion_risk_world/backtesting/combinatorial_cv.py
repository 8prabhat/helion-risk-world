"""Combinatorial purged cross-validation for predictive-metric distributions (SPEC.md §23.3, §24).

CPCV is used ONLY for the robustness distribution of predictive/calibration metrics
(quantile-coverage, latent-error, barrier ECE, PIT).  It is NOT used for the PnL
backtest, because a path-dependent account (running drawdown, management loop) cannot
be simulated coherently across time-discontiguous test blocks.

Reference: López de Prado (2018), Ch. 12.
SRP: CPCV split generation and metric aggregation only.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def make_cpcv_splits(
    n: int,
    n_splits: int = 6,
    test_size: int | None = None,
    embargo_bars: int = 12,
    t1: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate combinatorial purged CV (train, test) index pairs.

    n:           total number of samples
    n_splits:    number of folds (k); generates C(k, k//2) test combinations
    test_size:   samples per group; defaults to n // n_splits
    embargo_bars: bars to remove after each test block boundary
    t1:          optional array of label exit indices for purging; if None, no purge

    Returns a list of (train_idx, test_idx) pairs — test blocks are CONTIGUOUS within
    a group but the collection of all (train,test) pairs spans the full timeline.

    IMPORTANT: the caller is responsible for ensuring these are used only with
    metrics that are NOT path-dependent (do not use for PnL simulation).
    """
    if n < n_splits:
        raise ValueError(f"n={n} must be >= n_splits={n_splits}")
    size = test_size or (n // n_splits)
    # Assign each sample to a group
    groups = np.minimum(np.arange(n) // size, n_splits - 1)

    n_test_groups = max(1, n_splits // 2)
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    for test_groups in combinations(range(n_splits), n_test_groups):
        test_mask = np.isin(groups, test_groups)
        test_idx = np.where(test_mask)[0]
        if len(test_idx) == 0:
            continue

        # Purge: remove training samples whose label window overlaps ANY test bar
        if t1 is not None:
            test_start = int(test_idx[0])
            test_end = int(test_idx[-1])
            # training sample i is purged if t1[i] >= test_start or t0[i] <= test_end+embargo
            purge_mask = (t1 >= test_start)
        else:
            purge_mask = np.zeros(n, dtype=bool)

        # Embargo: remove bars within embargo_bars after test end
        embargo_end = int(test_idx[-1]) + embargo_bars
        embargo_mask = np.arange(n) <= embargo_end

        train_mask = ~test_mask & ~purge_mask & ~(embargo_mask & ~test_mask)
        # Simpler: exclude test + immediately-after-embargo
        embargo_indices = set(range(int(test_idx[-1]) + 1, min(int(test_idx[-1]) + embargo_bars + 1, n)))
        train_idx = np.array([
            i for i in range(n)
            if not test_mask[i] and i not in embargo_indices and (t1 is None or t1[i] < test_idx[0])
        ], dtype=int)

        splits.append((train_idx, test_idx))

    return splits


__all__ = ["make_cpcv_splits"]
