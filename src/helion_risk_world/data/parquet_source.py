"""Real ``MarketDataSource`` over a directory of per-symbol OHLCV parquet files (SPEC.md §8).

Supports either:
  * native ``<SYMBOL>_<base_interval>.parquet`` files fetched from Upstox, or
  * ``<SYMBOL>_1min.parquet`` files resampled up to ``base_interval``.

Upstox candle timestamps mark the *start* of the interval, so every loaded bar is converted to its
decision/availability time at the *close* of the interval before the feature builder sees it.
This keeps ``ts == available_at == bar close`` and avoids a 1-bar point-in-time leak on native
5-minute files.

No historical option chain is included (``option_chain`` returns None) — V1 does not require it.
SRP: read + align + serve point-in-time candles only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from collections.abc import Iterator
from typing import TYPE_CHECKING

from helion_risk_world.integration import get_logger
from helion_risk_world.schemas.market_schema import MarketCandle
from helion_risk_world.schemas.option_chain_schema import OptionContractSnapshot
from quanthelion.data.storage.wide_parquet import (
    infer_interval_from_path,
    load_ohlcv_parquet,
    prepare_ohlcv_frame,
    resolve_ohlcv_path,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

log = get_logger("hrw.parquet_source")


@dataclass
class ParquetMarketDataSource:
    """Point-in-time ``MarketDataSource`` backed by per-symbol OHLCV parquet files."""

    data_dir: str
    universe: tuple[str, ...]
    base_interval: str = "5min"
    _frames: dict[str, pd.DataFrame] = field(default_factory=dict, init=False)
    _index: pd.DatetimeIndex | None = field(default=None, init=False)

    @staticmethod
    def _row_to_candle(symbol: str, ts: datetime, row: pd.Series) -> MarketCandle:
        oi = None if row["oi"] != row["oi"] else float(row["oi"])  # NaN-safe
        return MarketCandle(
            symbol=symbol,
            ts=ts,
            available_at=ts,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
            oi=oi,
        )

    def _load_resample(self, symbol: str) -> pd.DataFrame:
        path, source_interval = resolve_ohlcv_path(self.data_dir, symbol, self.base_interval)
        df = load_ohlcv_parquet(path)
        return prepare_ohlcv_frame(
            df,
            source_interval=source_interval,
            target_interval=self.base_interval,
        )

    def _ensure(self) -> None:
        if self._index is not None:
            return
        frames = {s: self._load_resample(s) for s in self.universe}
        common = None
        for df in frames.values():
            common = df.index if common is None else common.intersection(df.index)
        if common is None or len(common) == 0:
            raise ValueError("no common timestamps across the universe")
        # Feature/label overhaul risk (breadth/dispersion depend on ALL non-primary
        # universe symbols jointly): a gap in any ONE constituent silently shrinks the
        # shared calendar for every symbol, not just the affected one. Treat this as a
        # hard data-quality failure instead of training on a biased common index.
        min_raw_len = min(len(df) for df in frames.values())
        if min_raw_len > 0 and len(common) < 0.999 * min_raw_len:
            raise ValueError(
                "parquet_source.universe_gap_shrinks_common_index "
                f"common={len(common)} min_constituent_raw={min_raw_len} "
                f"universe={self.universe}"
            )
        self._frames = {s: frames[s].loc[common] for s in self.universe}
        self._index = common

    def timestamps(self) -> list[datetime]:
        """The common decision timestamps (bar close times) shared by the whole universe."""
        self._ensure()
        assert self._index is not None
        return [ts.to_pydatetime() for ts in self._index]

    def timestamp_index(self) -> pd.DatetimeIndex:
        """The aligned common timestamp index across the configured universe."""
        self._ensure()
        assert self._index is not None
        return self._index

    def aligned_frames(self) -> dict[str, pd.DataFrame]:
        """The aligned OHLCV frames keyed by symbol on the shared timestamp index."""
        self._ensure()
        return dict(self._frames)

    def candle_window(self, symbol: str, end_ts: datetime, lookback: int) -> list[MarketCandle]:
        window = self.candle_frame(symbol, end_ts, lookback)
        out: list[MarketCandle] = []
        for ts, r in window.iterrows():
            py_ts = ts.to_pydatetime()
            out.append(self._row_to_candle(symbol, py_ts, r))
        return out

    def candle_frame(self, symbol: str, end_ts: datetime, lookback: int) -> pd.DataFrame:
        """Return the trailing aligned OHLCV frame for ``symbol`` up to ``end_ts``."""
        self._ensure()
        df = self._frames[symbol]
        end = df.index.searchsorted(end_ts, side="right")
        start = max(0, end - lookback)
        return df.iloc[start:end]

    def iter_candles(self) -> Iterator[MarketCandle]:
        """Yield every aligned candle once in timestamp order for validation/reporting."""
        self._ensure()
        rows: list[MarketCandle] = []
        for symbol, df in self._frames.items():
            for ts, record in df.iterrows():
                rows.append(self._row_to_candle(symbol, ts.to_pydatetime(), record))
        rows.sort(key=lambda row: (row.ts, row.symbol))
        yield from rows

    def option_chain(self, underlying: str, ts: datetime) -> list[OptionContractSnapshot] | None:
        return None  # no historical option chain in the V1 dataset

    def spot(self, symbol: str, ts: datetime) -> float:
        self._ensure()
        df = self._frames[symbol]
        end = df.index.searchsorted(ts, side="right")
        if end == 0:
            raise ValueError(f"no spot available for {symbol} at {ts}")
        return float(df.iloc[end - 1]["close"])


__all__ = [
    "ParquetMarketDataSource",
    "infer_interval_from_path",
    "load_ohlcv_parquet",
    "prepare_ohlcv_frame",
    "resolve_ohlcv_path",
]
