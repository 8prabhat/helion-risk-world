"""Purged + embargoed cross-validation splits (SPEC.md §11.3).

Wraps ``quanthelion.labels.embargo.make_purged_splits`` for the HRW label schema.
Purging removes every training sample whose label window [t, exit_t] overlaps the
test window.  Embargo adds at least ``embargo_bars`` bars after the test block to
prevent forward-looking contamination from recent labels.

Applied in BOTH train construction and CV — doing embargo only at one boundary is invalid.
SRP: split generation only.
"""

from __future__ import annotations

import numpy as np

try:
    from quanthelion.labels.embargo import make_purged_splits
    _HAS_QUANTHELION = True
except ImportError:
    _HAS_QUANTHELION = False

from helion_risk_world.schemas.label_schema import LabelRecord


def make_hrw_purged_splits(
    records: list[LabelRecord],
    n_splits: int = 4,
    embargo_bars: int = 12,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (train_idx, test_idx) pairs via purged+embargoed splits.

    Uses quanthelion's make_purged_splits when available, with a fallback
    implementation for environments where quanthelion is not installed.

    ``embargo_bars`` should be >= H (the label horizon) to prevent contamination.
    """
    n = len(records)
    if n == 0:
        return []

    # Build arrays quanthelion needs
    t0 = np.arange(n, dtype=int)
    # exit_t relative to the start of the records array
    t1 = np.array([r.exit_t for r in records], dtype=int)

    if _HAS_QUANTHELION:
        return list(make_purged_splits(t0, t1, n_splits=n_splits, embargo=embargo_bars))

    # Fallback: simple contiguous WF with manual purge/embargo
    return _fallback_purged_splits(t0, t1, n_splits, embargo_bars)


def _fallback_purged_splits(
    t0: np.ndarray,
    t1: np.ndarray,
    n_splits: int,
    embargo_bars: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pure-Python fallback for contiguous purged walk-forward splits."""
    n = len(t0)
    fold_size = n // (n_splits + 1)
    splits = []
    for fold in range(n_splits):
        test_start = (fold + 1) * fold_size
        test_end = test_start + fold_size
        test_idx = np.arange(test_start, min(test_end, n))

        test_start_bar = int(t0[test_idx[0]])
        test_end_bar = int(t0[test_idx[-1]]) + embargo_bars

        # Purge: remove training samples whose label window overlaps the test window
        train_mask = t1 < test_start_bar          # label resolved before test starts
        train_idx = np.where(train_mask)[0]
        train_idx = train_idx[train_idx < test_start]  # no future bar

        # Embargo (review finding M5): also exclude any candidate whose own start
        # bar falls inside [test_start_bar, test_end_bar) -- the test block or its
        # embargo buffer. The walk-forward filter above (train_idx < test_start)
        # already makes this a no-op for this fold's own test window, but keeps
        # the two-sided embargo this module's docstring promises intact should
        # this fallback ever be reused for a non-purely-sequential split.
        embargo_mask = (t0[train_idx] < test_start_bar) | (t0[train_idx] >= test_end_bar)
        train_idx = train_idx[embargo_mask]

        splits.append((train_idx, test_idx))
    return splits


__all__ = ["make_hrw_purged_splits"]
