"""Purged + embargoed walk-forward splits (SPEC.md §23, Day 7).

Delegates to ``quanthelion.labels.embargo.make_purged_splits`` (López de Prado, AFML Ch. 7):
each fold is chronologically ordered train -> val -> test with an embargo gap, NEVER a random
split. SRP: split generation only. ``embargo_bars`` is converted to minutes via the interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from helion_risk_world.integration.quanthelion_adapter import make_purged_splits

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd


class WalkForward:
    """Generate purged walk-forward folds via the framework's embargo logic (SPEC.md §23)."""

    def __init__(self, n_folds: int = 5, embargo_bars: int = 12, bar_minutes: int = 5) -> None:
        self._n_folds = n_folds
        self._embargo_bars = embargo_bars
        self._bar_minutes = bar_minutes

    @property
    def embargo_bars(self) -> int:
        return self._embargo_bars

    def splits(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        date_col: str = "trade_date",
        test_pct: float = 0.2,
    ) -> list[dict[str, Any]]:
        """Return ``n_folds`` dicts with chronological, embargoed 'train'/'val'/'test' frames."""
        if make_purged_splits is not None:
            return make_purged_splits(
                df,
                n_splits=self._n_folds,
                test_pct=test_pct,
                embargo_minutes=self._embargo_bars * self._bar_minutes,
                timestamp_col=timestamp_col,
                date_col=date_col,
            )
        folds = self.split_indices(len(df), test_size=max(1, int(len(df) * test_pct)))
        return [
            {
                "train": df.iloc[fold.train_start:fold.train_end].copy(),
                "val": df.iloc[fold.val_start:fold.val_end].copy(),
                "test": df.iloc[fold.test_start:fold.test_end].copy(),
            }
            for fold in folds
        ]

    def split_indices(
        self,
        n_samples: int,
        *,
        test_size: int | None = None,
        val_size: int | None = None,
        min_train_size: int | None = None,
    ) -> list[WalkForwardFold]:
        """Return chronological fold index ranges for sequence-based backtests.

        Test windows are contiguous and non-overlapping so out-of-sample reports can be
        stitched into one coherent chronological path.
        """
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        embargo = self._embargo_bars
        base_block = max(1, n_samples // max(self._n_folds + 3, 4))
        val = max(1, val_size or base_block)
        min_train = max(1, min_train_size or base_block * 2)
        first_test_start = min_train + val + 2 * embargo
        if first_test_start >= n_samples:
            raise ValueError(
                f"not enough samples ({n_samples}) for walk-forward with "
                f"min_train={min_train}, val={val}, embargo={embargo}"
            )

        remaining = n_samples - first_test_start
        t_size = max(1, test_size or max(1, remaining // self._n_folds))
        folds: list[WalkForwardFold] = []
        test_start = first_test_start

        for fold_id in range(self._n_folds):
            if test_start >= n_samples:
                break
            test_end = min(test_start + t_size, n_samples)
            val_end = max(0, test_start - embargo)
            val_start = max(0, val_end - val)
            train_end = max(0, val_start - embargo)
            if train_end <= 0 or test_end <= test_start:
                break
            folds.append(
                WalkForwardFold(
                    fold_id=fold_id,
                    train_start=0,
                    train_end=train_end,
                    val_start=val_start,
                    val_end=val_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )
            test_start = test_end

        if not folds:
            raise ValueError(f"unable to create walk-forward folds for n_samples={n_samples}")
        return folds


@dataclass(frozen=True)
class WalkForwardFold:
    """Index ranges for one chronological walk-forward fold."""

    fold_id: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int

    @property
    def train_slice(self) -> slice:
        return slice(self.train_start, self.train_end)

    @property
    def val_slice(self) -> slice:
        return slice(self.val_start, self.val_end)

    @property
    def test_slice(self) -> slice:
        return slice(self.test_start, self.test_end)

    def as_dict(self) -> dict[str, int]:
        return {
            "fold_id": self.fold_id,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "val_start": self.val_start,
            "val_end": self.val_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
        }


__all__ = ["WalkForward", "WalkForwardFold"]
