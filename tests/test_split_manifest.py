"""Chronological split-manifest tests."""

from __future__ import annotations

import pandas as pd

from helion_risk_world.training.split_manifest import ChronoSplitManifest


def test_split_manifest_is_chronological() -> None:
    index = pd.date_range("2026-01-01", periods=10, freq="D")
    labels = pd.DataFrame({"label_realized_at": index + pd.Timedelta(days=1)}, index=index)

    manifest = ChronoSplitManifest.from_labels(labels, train_fraction=0.6, val_fraction=0.2)

    assert pd.Timestamp(manifest.train_end) < pd.Timestamp(manifest.val_end)
    assert manifest.train_rows + manifest.val_rows + manifest.test_rows <= len(labels)
    assert manifest.train_rows == 5


def test_split_manifest_round_trips_and_filters_holdout() -> None:
    index = pd.date_range("2026-01-01", periods=10, freq="D")
    labels = pd.DataFrame({"value": range(10)}, index=index)

    manifest = ChronoSplitManifest.from_labels(labels, train_fraction=0.6, val_fraction=0.2)
    restored = ChronoSplitManifest.from_metadata(manifest.to_metadata())

    holdout = restored.filter_frame(labels, "holdout")
    test = restored.filter_frame(labels, "test")

    assert restored == manifest
    assert len(holdout) == manifest.val_rows + manifest.test_rows
    assert len(test) == manifest.test_rows
    boundary = pd.Timestamp(manifest.val_start) if manifest.val_start is not None else pd.Timestamp(manifest.train_end)
    assert holdout.index.min() >= boundary


def test_split_manifest_filters_pretest_rows() -> None:
    index = pd.date_range("2026-01-01", periods=10, freq="D")
    labels = pd.DataFrame({"label_realized_at": index}, index=index)

    manifest = ChronoSplitManifest.from_labels(labels, train_fraction=0.6, val_fraction=0.2)

    pretest = manifest.filter_labels(labels, "pretest")

    assert len(pretest) == manifest.train_rows + manifest.val_rows
    assert pretest.index.max() <= pd.Timestamp(manifest.val_end)


def test_split_manifest_applies_embargo_and_resolution_filters() -> None:
    index = pd.date_range("2026-01-01 09:15:00", periods=12, freq="5min")
    labels = pd.DataFrame(
        {
            "label_realized_at": [
                index[0],
                index[1],
                index[2],
                index[3],
                index[4] + pd.Timedelta(minutes=5),
                index[5] + pd.Timedelta(minutes=10),
                index[6],
                index[7],
                index[8],
                index[9],
                index[10],
                index[11],
            ]
        },
        index=index,
    )

    manifest = ChronoSplitManifest.from_labels(
        labels,
        train_fraction=0.5,
        val_fraction=0.25,
        embargo_bars=2,
        bar_interval="5min",
    )

    assert manifest.embargo_bars == 2
    assert pd.Timestamp(manifest.val_start) > pd.Timestamp(manifest.train_end)
    assert pd.Timestamp(manifest.test_start) > pd.Timestamp(manifest.val_end)
    assert manifest.train_rows == 5
    assert manifest.val_rows == 1
    assert manifest.test_rows == 1
    assert len(manifest.filter_labels(labels, "train")) == manifest.train_rows
    assert len(manifest.filter_labels(labels, "val")) == manifest.val_rows
    assert len(manifest.filter_labels(labels, "test")) == manifest.test_rows
