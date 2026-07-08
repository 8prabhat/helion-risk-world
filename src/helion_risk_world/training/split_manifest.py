"""Chronological split manifests derived from the current local label history."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class ChronoSplitManifest:
    """Chronological train/val/test boundaries persisted into model artifacts."""

    train_end: str
    val_end: str
    total_rows: int
    train_rows: int
    val_rows: int
    test_rows: int
    train_fraction: float
    val_fraction: float
    val_start: str | None = None
    test_start: str | None = None
    embargo_bars: int = 0

    @classmethod
    def from_labels(
        cls,
        labels: pd.DataFrame,
        *,
        train_fraction: float,
        val_fraction: float,
        embargo_bars: int = 0,
        bar_interval: str = "5min",
    ) -> "ChronoSplitManifest":
        if labels.empty:
            raise ValueError("cannot derive split manifest from an empty label frame")
        ordered = labels.sort_index()
        ts_index = pd.DatetimeIndex(pd.to_datetime(ordered.index))
        resolved_at = (
            pd.Series(pd.to_datetime(ordered["label_realized_at"]), index=ordered.index)
            if "label_realized_at" in ordered.columns
            else None
        )
        n = len(ordered)
        if n < 3:
            raise ValueError("need at least 3 labeled rows to derive train/val/test splits")
        train_cut = max(1, min(n - 2, int(n * train_fraction)))
        val_cut = max(train_cut + 1, min(n - 1, int(n * (train_fraction + val_fraction))))
        train_end = pd.Timestamp(ordered.index[train_cut - 1])
        val_end = pd.Timestamp(ordered.index[val_cut - 1])
        embargo = max(int(embargo_bars), 0)
        embargo_delta = pd.to_timedelta(bar_interval) * embargo
        val_start = _first_timestamp_after(ts_index, train_end + embargo_delta)
        test_start = _first_timestamp_after(ts_index, val_end + embargo_delta)

        train_mask = ts_index <= train_end
        if resolved_at is not None:
            train_mask &= (resolved_at <= train_end).to_numpy(dtype=bool, copy=False)
        val_mask = ts_index > train_end
        if val_start is not None:
            val_mask &= ts_index >= val_start
        val_mask &= ts_index <= val_end
        if resolved_at is not None:
            val_mask &= (resolved_at <= val_end).to_numpy(dtype=bool, copy=False)
        test_mask = ts_index > val_end
        if test_start is not None:
            test_mask &= ts_index >= test_start

        if val_cut > train_cut and not bool(val_mask.any()):
            raise ValueError("embargoed split leaves no validation rows; reduce embargo or adjust fractions")
        if n - val_cut > 0 and not bool(test_mask.any()):
            raise ValueError("embargoed split leaves no test rows; reduce embargo or adjust fractions")
        return cls(
            train_end=train_end.isoformat(),
            val_end=val_end.isoformat(),
            total_rows=n,
            train_rows=int(train_mask.sum()),
            val_rows=int(val_mask.sum()),
            test_rows=int(test_mask.sum()),
            train_fraction=float(train_fraction),
            val_fraction=float(val_fraction),
            val_start=val_start.isoformat() if val_start is not None else None,
            test_start=test_start.isoformat() if test_start is not None else None,
            embargo_bars=embargo,
        )

    @classmethod
    def from_metadata(cls, payload: dict[str, object]) -> "ChronoSplitManifest":
        return cls(
            train_end=str(payload["train_end"]),
            val_end=str(payload["val_end"]),
            total_rows=int(payload["total_rows"]),
            train_rows=int(payload["train_rows"]),
            val_rows=int(payload["val_rows"]),
            test_rows=int(payload["test_rows"]),
            train_fraction=float(payload["train_fraction"]),
            val_fraction=float(payload["val_fraction"]),
            val_start=(
                str(payload["val_start"])
                if payload.get("val_start") is not None
                else None
            ),
            test_start=(
                str(payload["test_start"])
                if payload.get("test_start") is not None
                else None
            ),
            embargo_bars=int(payload.get("embargo_bars", 0)),
        )

    def contains(self, ts: pd.Timestamp, split: str) -> bool:
        train_end = pd.Timestamp(self.train_end)
        val_end = pd.Timestamp(self.val_end)
        val_start = pd.Timestamp(self.val_start) if self.val_start is not None else None
        test_start = pd.Timestamp(self.test_start) if self.test_start is not None else None
        if split == "holdout":
            return ts >= val_start if val_start is not None else ts > train_end
        if split == "all":
            return True
        if split == "train":
            return ts <= train_end
        if split == "pretest":
            return ts <= val_end
        if split == "val":
            lower_ok = ts >= val_start if val_start is not None else ts > train_end
            return lower_ok and ts <= val_end
        if split == "test":
            return ts >= test_start if test_start is not None else ts > val_end
        raise ValueError(f"unsupported split: {split!r}")

    def mask(self, index: pd.Index, split: str) -> pd.Series:
        ts_index = pd.to_datetime(index)
        return pd.Series([self.contains(pd.Timestamp(ts), split) for ts in ts_index], index=index)

    def label_mask(self, labels: pd.DataFrame, split: str) -> pd.Series:
        if labels.empty:
            return pd.Series(dtype=bool, index=labels.index)
        mask = self.mask(labels.index, split)
        resolved_at = (
            pd.to_datetime(labels["label_realized_at"])
            if "label_realized_at" in labels.columns
            else None
        )
        if resolved_at is None:
            return mask
        if split == "train":
            mask &= resolved_at <= pd.Timestamp(self.train_end)
        elif split == "pretest":
            mask &= resolved_at <= pd.Timestamp(self.val_end)
        elif split == "val":
            mask &= resolved_at <= pd.Timestamp(self.val_end)
        return mask

    def filter_frame(self, frame: pd.DataFrame, split: str) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        return frame.loc[self.mask(frame.index, split)]

    def filter_labels(self, labels: pd.DataFrame, split: str) -> pd.DataFrame:
        if labels.empty:
            return labels.copy()
        return labels.loc[self.label_mask(labels, split)]

    def to_metadata(self) -> dict[str, object]:
        return asdict(self)


__all__ = ["ChronoSplitManifest"]


def _first_timestamp_after(index: pd.DatetimeIndex, cutoff: pd.Timestamp) -> pd.Timestamp | None:
    candidates = index[index > cutoff]
    if len(candidates) == 0:
        return None
    return pd.Timestamp(candidates[0])
