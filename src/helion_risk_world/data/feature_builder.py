"""Shared feature builder + DRY primitives (SPEC.md §10, Day 3).

This module is the SINGLE source of market-feature logic used by training, backtesting AND paper
trading — there is no second, slightly-different implementation anywhere (DRY). It produces
MARKET-plane tensors only; portfolio fields can never enter (enforced via
``assert_no_portfolio_in_market``).

Layout:
  * pure primitives (``log_returns``, ``realized_vol``, ``atr``, ...) — stateless, unit-tested
  * ``MarketDataSource`` protocol + ``InMemoryMarketDataSource`` — point-in-time data access
  * ``MarketBatch`` — the encoder-facing container ([A, L, F] candle tensor + optional futures)
  * ``FeatureBuilder`` — orchestrates window + futures assembly for a decision time ``ts``
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from helion_risk_world.config.data_config import DataConfig
from helion_risk_world.data.alpha_breadth_context import AlphaDataBreadthLoader
from helion_risk_world.data.alpha_features import AlphaDataMarketWindowBuilder
from helion_risk_world.data.alpha_futures_features import AlphaDataFuturesWindowBuilder
from helion_risk_world.data.leakage_checks import assert_no_portfolio_in_market
from helion_risk_world.data.market_window_builder import CANDLE_FEATURE_NAMES
from helion_risk_world.data.option_surface_builder import (
    SURFACE_CONTEXT_FEATURES,
    SURFACE_STRIKE_CHANNELS,
    OptionSurfaceBuilder,
    featurize_surface,
)

# Re-export the shared primitives so callers can do `from ...feature_builder import realized_vol`
# (SPEC.md §10 places the primitives at the feature-builder layer). Single definition lives in
# `data/primitives.py` to keep the dependency graph acyclic.
from helion_risk_world.data.primitives import (  # noqa: F401  (re-exported for the public API)
    atr,
    day_of_week,
    log_returns,
    oi_change,
    realized_vol,
    simple_returns,
    time_of_day,
    volume_zscore,
)
from helion_risk_world.schemas.market_schema import MarketCandle
from helion_risk_world.schemas.option_chain_schema import (
    OptionContractSnapshot,
    OptionSurfaceSnapshot,
)

# --------------------------------------------------------------------------------------
# Point-in-time data access. ``FeatureBuilder`` depends on this Protocol, not a concrete
# store (Dependency Inversion) — training, backtest and paper each supply their own source.
# --------------------------------------------------------------------------------------
@runtime_checkable
class MarketDataSource(Protocol):
    """Point-in-time market data access. All returns must satisfy ``available_at <= end_ts``."""

    def candle_window(self, symbol: str, end_ts: datetime, lookback: int) -> list[MarketCandle]:
        """Return the most recent ``lookback`` candles with ``available_at <= end_ts``."""
        ...

    def option_chain(self, underlying: str, ts: datetime) -> list[OptionContractSnapshot] | None:
        """Return the latest option-chain snapshot available at ``ts`` (or None)."""
        ...

    def spot(self, symbol: str, ts: datetime) -> float:
        """Return the latest spot/close available at ``ts``."""
        ...


@dataclass(frozen=True)
class InMemoryMarketDataSource:
    """Simple in-memory ``MarketDataSource`` for dev/tests (and small cached slices).

    Holds already point-in-time candle lists keyed by symbol. ``candle_window`` filters by
    ``available_at <= end_ts`` then returns the trailing ``lookback`` rows.
    """

    candles: dict[str, list[MarketCandle]]
    chains: dict[str, list[OptionContractSnapshot]] = field(default_factory=dict)

    def candle_window(self, symbol: str, end_ts: datetime, lookback: int) -> list[MarketCandle]:
        rows = [c for c in self.candles.get(symbol, []) if c.available_at <= end_ts]
        rows.sort(key=lambda c: c.ts)
        return rows[-lookback:]

    def option_chain(self, underlying: str, ts: datetime) -> list[OptionContractSnapshot] | None:
        chain = [c for c in self.chains.get(underlying, []) if c.available_at <= ts]
        return chain or None

    def spot(self, symbol: str, ts: datetime) -> float:
        window = self.candle_window(symbol, ts, 1)
        if not window:
            raise ValueError(f"no spot available for {symbol} at {ts}")
        return window[-1].close

    def aligned_frames(self) -> dict[str, pd.DataFrame]:
        """Aligned OHLCV frames keyed by symbol on the shared in-memory timestamp grid."""
        import pandas as pd

        frames: dict[str, pd.DataFrame] = {}
        common: pd.DatetimeIndex | None = None
        for symbol, candles in self.candles.items():
            frame = pd.DataFrame.from_records(
                {
                    "ts": candle.ts,
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                    "oi": candle.oi,
                }
                for candle in sorted(candles, key=lambda row: row.ts)
            )
            if frame.empty:
                continue
            frame = frame.set_index("ts")
            frame.index = pd.to_datetime(frame.index)
            frame = frame.sort_index()
            frames[symbol] = frame
            common = frame.index if common is None else common.intersection(frame.index)
        if common is None or len(common) == 0:
            raise ValueError("no common timestamps across the in-memory universe")
        return {symbol: frame.loc[common] for symbol, frame in frames.items()}

    def timestamp_index(self) -> pd.DatetimeIndex:
        """The aligned common timestamp index across the in-memory universe."""
        frames = self.aligned_frames()
        first = next(iter(frames.values()), None)
        if first is None:
            raise ValueError("no in-memory frames available")
        return first.index

    def candle_frame(self, symbol: str, end_ts: datetime, lookback: int) -> pd.DataFrame:
        """Return the trailing aligned OHLCV frame for ``symbol`` up to ``end_ts``."""
        frames = self.aligned_frames()
        df = frames[symbol]
        end = df.index.searchsorted(end_ts, side="right")
        start = max(0, end - lookback)
        return df.iloc[start:end]


# --------------------------------------------------------------------------------------
# Encoder-facing batch.
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MarketBatch:
    """Market-plane features for one decision step. The ONLY thing encoders consume.

    candle_features: [A, L, F]      (assets, lookback, features)
    feature_names:   length F       (all must be market-plane names)
    futures:         [L, 12] or None — futures microstructure window for FuturesEncoder
    surface:         retained for callers that still use OptionSurfaceSnapshot; model ignores it
    """

    ts: datetime
    symbols: tuple[str, ...]
    candle_features: np.ndarray
    feature_names: tuple[str, ...]
    futures: np.ndarray | None = None        # [L, FUTURES_FEATURE_DIM=14]
    surface: OptionSurfaceSnapshot | None = None

    def __post_init__(self) -> None:
        if self.candle_features.ndim != 3:
            raise ValueError(
                f"candle_features must be [A, L, F]; got ndim={self.candle_features.ndim}"
            )
        a, _, f = self.candle_features.shape
        if a != len(self.symbols):
            raise ValueError(f"asset axis {a} != n_symbols {len(self.symbols)}")
        if f != len(self.feature_names):
            raise ValueError(f"feature axis {f} != n_feature_names {len(self.feature_names)}")
        if self.futures is not None and self.futures.ndim != 2:
            raise ValueError(
                f"futures must be [L, FUTURES_FEATURE_DIM]; got ndim={self.futures.ndim}"
            )


@dataclass(frozen=True)
class MarketFeatureHistory:
    """Precomputed per-bar market features on a shared aligned timestamp grid."""

    index: pd.DatetimeIndex
    symbols: tuple[str, ...]
    feature_names: tuple[str, ...]
    candle_features: np.ndarray  # [T, A, F]

    def __post_init__(self) -> None:
        if self.candle_features.ndim != 3:
            raise ValueError(
                f"candle_features must be [T, A, F]; got ndim={self.candle_features.ndim}"
            )
        t, a, f = self.candle_features.shape
        if t != len(self.index):
            raise ValueError(f"time axis {t} != n_timestamps {len(self.index)}")
        if a != len(self.symbols):
            raise ValueError(f"asset axis {a} != n_symbols {len(self.symbols)}")
        if f != len(self.feature_names):
            raise ValueError(f"feature axis {f} != n_feature_names {len(self.feature_names)}")

    def window_view(self, lookback: int) -> np.ndarray:
        """Return a zero-copy-ish view of trailing windows as [T-L+1, A, L, F]."""
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        if len(self.index) < lookback:
            return np.empty(
                (0, len(self.symbols), lookback, len(self.feature_names)),
                dtype=self.candle_features.dtype,
            )
        view = np.lib.stride_tricks.sliding_window_view(
            self.candle_features,
            window_shape=lookback,
            axis=0,
        )  # [T-L+1, A, F, L]
        view = np.moveaxis(view, -1, 1)   # [T-L+1, L, A, F]
        return np.transpose(view, (0, 2, 1, 3))  # [T-L+1, A, L, F]


class FeatureBuilder:
    """THE shared feature builder for training, backtesting AND paper trading (DRY; SPEC.md §10).

    Produces MARKET-plane tensors only (no portfolio fields). SRP: feature construction / assembly;
    point-in-time access is delegated to a ``MarketDataSource`` (DIP).

    V1 futures path: pass ``futures_builder`` to attach the [L, 12] futures microstructure window
    to each ``MarketBatch``. ``AlphaDataFuturesWindowBuilder`` reads alpha_data's precomputed
    futures-microstructure parquet directly (Phase 2 migration).
    """

    def __init__(
        self,
        cfg: DataConfig,
        source: MarketDataSource,
        *,
        window_builder: AlphaDataMarketWindowBuilder | None = None,
        surface_builder: OptionSurfaceBuilder | None = None,
        futures_builder: AlphaDataFuturesWindowBuilder | None = None,
        breadth_loader: AlphaDataBreadthLoader | None = None,
    ) -> None:
        self._cfg = cfg
        self._source = source
        self._window = window_builder or AlphaDataMarketWindowBuilder(interval=cfg.base_interval)
        self._surface = surface_builder or OptionSurfaceBuilder(n_strikes=cfg.n_strikes)
        self._futures = futures_builder   # None → futures not available; MarketBatch.futures = None
        self._breadth = breadth_loader or AlphaDataBreadthLoader(
            cfg.universe[0], interval=cfg.base_interval
        )

    def build_window(self, ts: datetime) -> MarketBatch:
        """Build the point-in-time market batch at decision time ``ts``.

        Returns a [A, L, F] candle tensor across the configured universe, an optional [L, 12]
        futures microstructure window (when ``futures_builder`` is provided), and the primary
        underlying's ATM-relative option surface (when a chain is available).
        All features are market-plane (no leakage), enforced before returning.
        """
        feats: list[np.ndarray] = []
        names: tuple[str, ...] = CANDLE_FEATURE_NAMES
        frame_window = getattr(self._source, "candle_frame", None)
        timestamp_index = getattr(self._source, "timestamp_index", None)
        full_history_lookback = (
            len(timestamp_index()) if callable(timestamp_index) else self._cfg.lookback_bars
        )
        primary_index: pd.DatetimeIndex | None = None
        for i, symbol in enumerate(self._cfg.universe):
            if callable(frame_window):
                candles = frame_window(symbol, ts, full_history_lookback)
                if len(candles) < 2:
                    raise ValueError(
                        f"insufficient candles for {symbol} at {ts}: got {len(candles)}, need >= 2"
                    )
                build_frame_for_symbol = getattr(self._window, "build_frame_for_symbol", None)
                if callable(build_frame_for_symbol):
                    full_window, names = build_frame_for_symbol(symbol, candles)
                else:
                    full_window, names = self._window.build_frame(candles)  # type: ignore[arg-type]
                window = full_window[-self._cfg.lookback_bars :]
                if i == 0:
                    primary_index = pd.DatetimeIndex(candles.index[-self._cfg.lookback_bars :])
            else:
                candles = self._source.candle_window(symbol, ts, self._cfg.lookback_bars)
                if len(candles) < 2:
                    raise ValueError(
                        f"insufficient candles for {symbol} at {ts}: got {len(candles)}, need >= 2"
                    )
                # Point-in-time safety (defence in depth; the source should already guarantee this).
                for c in candles:
                    if c.available_at > ts:
                        raise ValueError(
                            f"point-in-time violation for {symbol}: {c.available_at} > {ts}"
                        )
                build_for_symbol = getattr(self._window, "build_for_symbol", None)
                if callable(build_for_symbol):
                    window, names = build_for_symbol(symbol, candles, ts)
                else:
                    window, names = self._window.build(candles)  # [L, F]
                if i == 0:
                    primary_index = pd.DatetimeIndex([c.ts for c in candles])
            feats.append(window)

        lengths = {w.shape[0] for w in feats}
        if len(lengths) != 1:
            raise ValueError(f"all symbol windows must share lookback length; got {lengths}")
        candle_features = np.stack(feats, axis=0).copy()  # [A, L, F]  writable copy

        # Fill rel_log_return (col 18): asset log_return minus primary-asset log_return.
        # Primary is index 0 (BANKNIFTY). Primary gets 0 (BNK - BNK = 0).
        if "rel_log_return" in names and "log_return" in names:
            lr_idx  = names.index("log_return")
            rel_idx = names.index("rel_log_return")
            primary_lr = candle_features[0, :, lr_idx]
            for a in range(candle_features.shape[0]):
                candle_features[a, :, rel_idx] = candle_features[a, :, lr_idx] - primary_lr

        # Fill breadth/dispersion (cols 25-26) from alpha_data's real constituent breadth
        # pipeline (feature-onboarding pass) instead of recomputing an equivalent statistic
        # locally from the candle tensor's own trailing N-bar returns — market-wide scalars
        # broadcast IDENTICALLY into every asset's row (unlike rel_log_return, which is
        # per-asset-differentiated), same broadcast shape tod_sin/dow_sin already use.
        if "breadth" in names and "dispersion" in names and primary_index is not None:
            breadth_idx = names.index("breadth")
            dispersion_idx = names.index("dispersion")
            breadth_vals, dispersion_vals = self._breadth.aligned(primary_index)
            candle_features[:, :, breadth_idx] = breadth_vals[None, :]
            candle_features[:, :, dispersion_idx] = dispersion_vals[None, :]

        # Leakage guard: no portfolio field may appear among encoder feature names (SPEC.md §5).
        assert_no_portfolio_in_market(names)

        # Futures microstructure window [L, 12] — only when a futures source was provided.
        futures: np.ndarray | None = None
        if self._futures is not None:
            futures = self._futures.build_window(ts, self._cfg.lookback_bars)

        surface: OptionSurfaceSnapshot | None = None
        primary = self._cfg.universe[0]
        chain = self._source.option_chain(primary, ts)
        if chain:
            surface = self._surface.align_to_atm(chain, self._source.spot(primary, ts), ts)

        return MarketBatch(
            ts=ts,
            symbols=tuple(self._cfg.universe),
            candle_features=candle_features,
            feature_names=names,
            futures=futures,
            surface=surface,
        )

    def build_history(self) -> MarketFeatureHistory:
        """Precompute full-history per-bar features on the source's aligned grid."""
        frames_provider = getattr(self._source, "aligned_frames", None)
        if not callable(frames_provider):
            raise TypeError("build_history requires a source with aligned_frames() support")

        frames = frames_provider()
        primary = frames[self._cfg.universe[0]]
        index = primary.index
        feats: list[np.ndarray] = []
        names: tuple[str, ...] = CANDLE_FEATURE_NAMES
        build_frame_for_symbol = getattr(self._window, "build_frame_for_symbol", None)
        for symbol in self._cfg.universe:
            frame = frames[symbol]
            if len(frame) != len(index) or not frame.index.equals(index):
                raise ValueError("all aligned source frames must share the same timestamp index")
            if callable(build_frame_for_symbol):
                window, names = build_frame_for_symbol(symbol, frame)
            else:
                window, names = self._window.build_frame(frame)
            feats.append(window.astype(np.float32, copy=False))

        candle_features = np.stack(feats, axis=1).astype(np.float32, copy=False)  # [T, A, F]
        if "rel_log_return" in names and "log_return" in names:
            lr_idx = names.index("log_return")
            rel_idx = names.index("rel_log_return")
            primary_lr = candle_features[:, 0, lr_idx]
            candle_features[:, :, rel_idx] = candle_features[:, :, lr_idx] - primary_lr[:, None]

        # Fill breadth/dispersion (cols 25-26) from alpha_data's real constituent breadth
        # pipeline — see build_window()'s identical-logic comment above; here the stack is
        # [T, A, F] (axis=1 is the asset axis) instead of build_window()'s [A, L, F].
        if "breadth" in names and "dispersion" in names:
            breadth_idx = names.index("breadth")
            dispersion_idx = names.index("dispersion")
            breadth_vals, dispersion_vals = self._breadth.aligned(index)
            candle_features[:, :, breadth_idx] = breadth_vals[:, None]
            candle_features[:, :, dispersion_idx] = dispersion_vals[:, None]

        assert_no_portfolio_in_market(names)
        return MarketFeatureHistory(
            index=index,
            symbols=tuple(self._cfg.universe),
            feature_names=names,
            candle_features=candle_features,
        )

    def build_surface_history(
        self, timestamps: Sequence[datetime]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Per-timestamp option-surface tensors for training (feature-onboarding pass).

        Unlike ``build_history()`` (a contiguous per-bar candle history), option chains are
        looked up one arbitrary (typically sparse, label-row) timestamp at a time via
        ``self._source.option_chain(...)`` — there is no vectorized alpha_data chain-history
        equivalent, so this loops per row exactly like ``scripts/train.py``'s existing
        per-row ``regime_builder.build(ts)`` loop for ``regime_context_arr``.

        Returns ``(grid[N,S,C], mask[N,S], context[N,K], eligible[N])``. A row is
        ``eligible=False`` (grid/mask/context left zero) when no chain is available for
        that underlying at that timestamp — the caller should exclude ineligible rows from
        the trainable batch, matching ``AlphaDataFuturesWindowBuilder``'s per-row
        eligibility convention.
        """
        primary = self._cfg.universe[0]
        n = len(timestamps)
        n_tokens = 2 * self._cfg.n_strikes + 1
        grid = np.zeros((n, n_tokens, len(SURFACE_STRIKE_CHANNELS)), dtype=np.float32)
        mask = np.zeros((n, n_tokens), dtype=np.float32)
        context = np.zeros((n, len(SURFACE_CONTEXT_FEATURES)), dtype=np.float32)
        eligible = np.zeros(n, dtype=bool)
        for i, ts in enumerate(timestamps):
            chain = self._source.option_chain(primary, ts)
            if not chain:
                continue
            spot = self._source.spot(primary, ts)
            snapshot = self._surface.align_to_atm(chain, spot, ts)
            g, m, c = featurize_surface(snapshot)
            if g.shape[0] != n_tokens:
                continue  # defensive: unexpected token count for this cycle/day, skip row
            grid[i], mask[i], context[i] = g, m, c
            eligible[i] = True
        return grid, mask, context, eligible

    @property
    def futures_builder(self) -> AlphaDataFuturesWindowBuilder | None:
        return self._futures

    @property
    def surface_builder(self) -> OptionSurfaceBuilder:
        return self._surface
