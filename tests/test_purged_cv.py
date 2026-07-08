"""purged_cv._fallback_purged_splits (review finding M5: two-sided embargo)."""

from __future__ import annotations

import numpy as np

from helion_risk_world.labeling.purged_cv import _fallback_purged_splits


def test_fallback_purged_splits_train_precedes_test_and_respects_embargo() -> None:
    n = 40
    t0 = np.arange(n, dtype=int)
    t1 = t0 + 2  # each label resolves 2 bars after its own start
    embargo_bars = 3

    splits = _fallback_purged_splits(t0, t1, n_splits=3, embargo_bars=embargo_bars)
    assert len(splits) == 3

    for train_idx, test_idx in splits:
        assert len(test_idx) > 0
        test_start_bar = int(t0[test_idx[0]])
        test_end_bar = int(t0[test_idx[-1]]) + embargo_bars

        # Purge: no training sample's label resolves at/after the test window starts.
        assert bool((t1[train_idx] < test_start_bar).all())
        # Embargo (M5): no training sample's own bar falls inside
        # [test_start_bar, test_end_bar) -- the test block or its embargo buffer.
        in_embargo_zone = (t0[train_idx] >= test_start_bar) & (t0[train_idx] < test_end_bar)
        assert not bool(in_embargo_zone.any())
