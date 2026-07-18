"""ParquetMarketDataSource over OHLCV parquet (SPEC.md §8, §27).

Self-contained: writes tiny synthetic parquet files to a temp dir, so it tests the source's
resample/align/point-in-time logic without depending on any real data location.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from helion_risk_world.data.parquet_source import ParquetMarketDataSource  # noqa: E402

UNIVERSE = ("AAA", "BBB")


def _write_parquet(data_dir: Path, symbol: str, n: int = 200, base: float = 100.0) -> None:
    start = datetime(2026, 1, 1, 9, 15)
    idx = pd.date_range(start, periods=n, freq="1min")
    px = base + pd.Series(range(n), index=idx) * 0.1
    df = pd.DataFrame(
        {"open": px, "high": px * 1.001, "low": px * 0.999, "close": px,
         "volume": 1000.0, "oi": 0.0},
        index=idx,
    )
    df.index.name = "datetime"
    (data_dir / "ohlcv").mkdir(parents=True, exist_ok=True)
    df.to_parquet(data_dir / "ohlcv" / f"{symbol}_1min.parquet")


def _write_native_interval_parquet(
    data_dir: Path,
    symbol: str,
    *,
    n: int = 200,
    base: float = 100.0,
    interval: str = "5min",
) -> None:
    start = datetime(2026, 1, 1, 9, 15)
    idx = pd.date_range(start, periods=n, freq=interval)
    px = base + pd.Series(range(n), index=idx) * 0.1
    df = pd.DataFrame(
        {"open": px, "high": px * 1.001, "low": px * 0.999, "close": px,
         "volume": 1000.0, "oi": 0.0},
        index=idx,
    )
    df.index.name = "datetime"
    (data_dir / "ohlcv").mkdir(parents=True, exist_ok=True)
    df.to_parquet(data_dir / "ohlcv" / f"{symbol}_{interval}.parquet")


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    for s in UNIVERSE:
        _write_parquet(tmp_path, s)
    return tmp_path


def test_resamples_1min_to_5min_with_right_label(data_dir: Path) -> None:
    src = ParquetMarketDataSource(data_dir=str(data_dir), universe=UNIVERSE, base_interval="5min")
    ts = src.timestamps()
    assert len(ts) > 0
    # 5-min bars labelled at the CLOSE (right edge): first 1-min at 09:15 -> first bar 09:20.
    assert ts[0] == datetime(2026, 1, 1, 9, 20)
    assert (ts[1] - ts[0]) == timedelta(minutes=5)


def test_candle_window_is_point_in_time(data_dir: Path) -> None:
    src = ParquetMarketDataSource(data_dir=str(data_dir), universe=UNIVERSE)
    ts = src.timestamps()
    window = src.candle_window("AAA", ts[10], lookback=4)
    assert len(window) == 4
    assert all(c.available_at <= ts[10] for c in window)        # nothing from the future
    assert all(c.available_at == c.ts for c in window)          # bar known at its close
    assert window[-1].ts == ts[10]


# NOTE: test_universe_aligned_to_common_index and test_build_history_window_matches_build_window
# were removed here (Phase 2 alpha_data migration): both pushed synthetic AAA/BBB OHLCV through
# ParquetMarketDataSource and expected FeatureBuilder to compute features from those made-up
# values -- but AlphaDataMarketWindowBuilder now looks up precomputed features by real symbol
# name + timestamp in alpha_data's actual store, so a fake symbol like "AAA" can never resolve
# there regardless of what synthetic parquet is written locally. ParquetMarketDataSource's own
# resample/align/point-in-time logic (tested by every other test in this file, none of which
# route through FeatureBuilder) is unaffected and still fully self-contained.


def test_missing_symbol_raises(data_dir: Path) -> None:
    src = ParquetMarketDataSource(data_dir=str(data_dir), universe=("AAA", "MISSING"))
    with pytest.raises(FileNotFoundError):
        src.timestamps()


def test_native_base_interval_is_shifted_to_bar_close(tmp_path: Path) -> None:
    for symbol in UNIVERSE:
        _write_native_interval_parquet(tmp_path, symbol)
    src = ParquetMarketDataSource(data_dir=str(tmp_path), universe=UNIVERSE, base_interval="5min")
    ts = src.timestamps()
    assert ts[0] == datetime(2026, 1, 1, 9, 20)
    window = src.candle_window("AAA", ts[3], lookback=2)
    assert window[-1].ts == ts[3]
    assert all(c.available_at == c.ts for c in window)


def _write_parquet_with_gap(
    data_dir: Path, symbol: str, gap_start_min: int, gap_len_min: int = 10,
    n: int = 200, base: float = 100.0,
) -> None:
    """Like _write_parquet, but with a contiguous block of 1-min rows removed —
    a genuine coverage gap, not just a shorter overall series."""
    start = datetime(2026, 1, 1, 9, 15)
    idx = pd.date_range(start, periods=n, freq="1min")
    px = base + pd.Series(range(n), index=idx) * 0.1
    df = pd.DataFrame(
        {"open": px, "high": px * 1.001, "low": px * 0.999, "close": px,
         "volume": 1000.0, "oi": 0.0},
        index=idx,
    )
    df = df.drop(index=idx[gap_start_min : gap_start_min + gap_len_min])
    df.index.name = "datetime"
    (data_dir / "ohlcv").mkdir(parents=True, exist_ok=True)
    df.to_parquet(data_dir / "ohlcv" / f"{symbol}_1min.parquet")


def test_universe_gap_shrinking_common_index_raises(tmp_path: Path) -> None:
    """Feature/label overhaul risk: breadth/dispersion depend on ALL non-primary universe
    symbols jointly, so a gap in one constituent silently shrinking the shared calendar
    for everyone must be logged loudly, not swallowed silently. Each symbol has its OWN
    gap at a DIFFERENT point in time (not just an overall shorter series) — the
    intersection loses bars from both gaps, dropping below either symbol's own row count."""
    _write_parquet_with_gap(tmp_path, "AAA", gap_start_min=50, gap_len_min=10, n=200)
    _write_parquet_with_gap(tmp_path, "BBB", gap_start_min=150, gap_len_min=10, n=200)
    src = ParquetMarketDataSource(data_dir=str(tmp_path), universe=("AAA", "BBB"))
    with pytest.raises(ValueError, match="universe_gap_shrinks_common_index"):
        src.timestamps()


def test_no_warning_when_universe_symbols_are_fully_aligned(tmp_path: Path, caplog) -> None:
    _write_parquet(tmp_path, "AAA", n=200)
    _write_parquet(tmp_path, "BBB", n=200)
    src = ParquetMarketDataSource(data_dir=str(tmp_path), universe=("AAA", "BBB"))
    with caplog.at_level("WARNING", logger="hrw.parquet_source"):
        src.timestamps()
    assert not any(
        "universe_gap_shrinks_common_index" in rec.getMessage() for rec in caplog.records
    )
