"""Shared runtime/training input contract for forecaster inference paths."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from helion_risk_world.config.data_config import DataConfig
from helion_risk_world.barrier_context import BarrierContext
from helion_risk_world.data.alpha_features import AlphaDataMarketWindowBuilder
from helion_risk_world.data.alpha_futures_features import AlphaDataFuturesWindowBuilder
from helion_risk_world.data.alpha_option_chain import (
    AlphaDataAtmGreeksLoader,
    AlphaDataOptionChainSource,
    AlphaDataSurfaceStatsLoader,
    CompositeMarketDataSource,
)
from helion_risk_world.data.alpha_regime_context import AlphaDataMacroContextLoader
from helion_risk_world.data.feature_builder import FeatureBuilder, MarketBatch, MarketDataSource
from helion_risk_world.data.market_window_builder import CANDLE_FEATURE_NAMES
from helion_risk_world.data.option_surface_builder import OptionSurfaceBuilder
from helion_risk_world.data.regime_context_builder import RegimeContextBuilder, _VixLoader
from helion_risk_world.schemas.market_schema import EventContext, RegimeContext

RegimeInput = tuple[RegimeContext, EventContext]


def _window_view_2d(values: np.ndarray, lookback: int) -> np.ndarray:
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    if values.shape[0] < lookback:
        return np.empty((0, lookback, values.shape[1]), dtype=values.dtype)
    view = np.lib.stride_tricks.sliding_window_view(values, window_shape=lookback, axis=0)
    return np.moveaxis(view, -1, 1)


@dataclass(frozen=True)
class ModelInputContract:
    """Persisted contract for the model's expected runtime inputs."""

    universe: tuple[str, ...]
    base_interval: str
    lookback_bars: int
    feature_names: tuple[str, ...]
    uses_futures: bool = False
    uses_regime_context: bool = False
    uses_option_surface: bool = False
    require_vix: bool = False
    require_daily_context: bool = False
    barrier_stop_mult: float = 2.0
    barrier_target_mult: float = 2.0
    barrier_vol_span: int = 50
    barrier_horizon_bars: int = 1
    barrier_cost_floor_frac: float = 0.0

    @classmethod
    def from_data_config(
        cls,
        cfg: DataConfig,
        *,
        feature_names: tuple[str, ...] = CANDLE_FEATURE_NAMES,
        uses_futures: bool = False,
        uses_regime_context: bool = False,
        uses_option_surface: bool = False,
        require_vix: bool = False,
        require_daily_context: bool = False,
        barrier_stop_mult: float = 2.0,
        barrier_target_mult: float = 2.0,
        barrier_vol_span: int = 50,
        barrier_horizon_bars: int = 1,
        barrier_cost_floor_frac: float = 0.0,
    ) -> ModelInputContract:
        return cls(
            universe=tuple(cfg.universe),
            base_interval=str(cfg.base_interval),
            lookback_bars=int(cfg.lookback_bars),
            feature_names=tuple(feature_names),
            uses_futures=bool(uses_futures),
            uses_regime_context=bool(uses_regime_context),
            uses_option_surface=bool(uses_option_surface),
            require_vix=bool(require_vix),
            require_daily_context=bool(require_daily_context),
            barrier_stop_mult=float(barrier_stop_mult),
            barrier_target_mult=float(barrier_target_mult),
            barrier_vol_span=int(barrier_vol_span),
            barrier_horizon_bars=int(barrier_horizon_bars),
            barrier_cost_floor_frac=float(barrier_cost_floor_frac),
        )

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> ModelInputContract | None:
        payload = metadata.get("input_contract")
        if payload is None:
            return None
        return cls(
            universe=tuple(payload["universe"]),
            base_interval=str(payload["base_interval"]),
            lookback_bars=int(payload["lookback_bars"]),
            feature_names=tuple(payload["feature_names"]),
            uses_futures=bool(payload.get("uses_futures", False)),
            uses_regime_context=bool(payload.get("uses_regime_context", False)),
            uses_option_surface=bool(payload.get("uses_option_surface", False)),
            require_vix=bool(payload.get("require_vix", False)),
            require_daily_context=bool(payload.get("require_daily_context", False)),
            barrier_stop_mult=float(payload.get("barrier_stop_mult", 2.0)),
            barrier_target_mult=float(payload.get("barrier_target_mult", 2.0)),
            barrier_vol_span=int(payload.get("barrier_vol_span", 50)),
            barrier_horizon_bars=int(payload.get("barrier_horizon_bars", 1)),
            barrier_cost_floor_frac=float(payload.get("barrier_cost_floor_frac", 0.0)),
        )

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)

    def assert_compatible(self, cfg: DataConfig) -> None:
        if tuple(cfg.universe) != self.universe:
            raise ValueError(
                f"config universe {tuple(cfg.universe)} does not match artifact universe {self.universe}"
            )
        if str(cfg.base_interval) != self.base_interval:
            raise ValueError(
                f"config base_interval {cfg.base_interval!r} does not match "
                f"artifact {self.base_interval!r}"
            )
        if int(cfg.lookback_bars) != self.lookback_bars:
            raise ValueError(
                f"config lookback_bars {cfg.lookback_bars} does not match "
                f"artifact {self.lookback_bars}"
            )
        if tuple(CANDLE_FEATURE_NAMES) != self.feature_names:
            raise ValueError(
                "runtime candle feature layout does not match the artifact contract; retrain "
                "or run against matching code."
            )


@dataclass(frozen=True)
class ModelInputSnapshot:
    """Point-in-time market input bundle for prediction."""

    market: MarketBatch
    regime: RegimeInput | None = None
    barrier_context: BarrierContext | None = None


@dataclass
class ModelInputBuilder:
    """Build market + optional futures/regime inputs under one explicit contract."""

    contract: ModelInputContract
    feature_builder: FeatureBuilder
    regime_builder: RegimeContextBuilder | None = None

    @classmethod
    def from_data_dir(
        cls,
        cfg: DataConfig,
        source: MarketDataSource,
        *,
        data_dir: str | Path | None,
        contract: ModelInputContract,
    ) -> ModelInputBuilder:
        """Build from alpha_data's precomputed features (Phase 2 migration -- see
        alpha_data/docs/DATA_CATALOG.md). ``data_dir``/local assembled-parquet paths
        are no longer read; every builder here sources from the shared alpha_data lake
        instead of this repo's own local computation.
        """
        contract.assert_compatible(cfg)

        futures_builder = None
        if contract.uses_futures:
            futures_builder = AlphaDataFuturesWindowBuilder(cfg.universe[0], interval=cfg.base_interval)

        regime_builder = None
        if contract.uses_regime_context:
            if contract.require_daily_context:
                raise ValueError(
                    "artifact requires non-Upstox daily_context regime data; retrain with "
                    "Upstox-only regime inputs"
                )
            root = Path(data_dir) if data_dir is not None else None
            vix_path = root / "ohlcv" / f"INDIAVIX_{cfg.base_interval}.parquet" if root is not None else None
            if contract.require_vix and (vix_path is None or not vix_path.exists()):
                raise FileNotFoundError("artifact expects INDIAVIX regime data, but it is missing")
            vix_loader = (
                _VixLoader.from_parquet(vix_path) if vix_path is not None and vix_path.exists() else None
            )
            regime_builder = RegimeContextBuilder(
                vix_loader=vix_loader,
                daily_ctx=AlphaDataMacroContextLoader(cfg.universe[0], interval=cfg.base_interval),
                symbol=cfg.universe[0],
                allow_non_upstox_context=True,
            )

        surface_builder = None
        if contract.uses_option_surface:
            source = CompositeMarketDataSource(
                source, AlphaDataOptionChainSource(interval=cfg.base_interval),
            )
            surface_builder = OptionSurfaceBuilder(
                n_strikes=cfg.n_strikes,
                stats_source=AlphaDataSurfaceStatsLoader(interval=cfg.base_interval),
                atm_greeks_source=AlphaDataAtmGreeksLoader(interval=cfg.base_interval),
            )

        feature_builder = FeatureBuilder(
            cfg, source,
            window_builder=AlphaDataMarketWindowBuilder(interval=cfg.base_interval),
            surface_builder=surface_builder,
            futures_builder=futures_builder,
        )
        return cls(
            contract=contract,
            feature_builder=feature_builder,
            regime_builder=regime_builder,
        )

    def build(self, ts: datetime) -> ModelInputSnapshot:
        market = self.feature_builder.build_window(ts)
        if tuple(market.feature_names) != self.contract.feature_names:
            raise ValueError(
                "feature builder output does not match the artifact feature contract"
            )
        regime = self.regime_builder.build(ts) if self.regime_builder is not None else None
        barrier_context = None
        futures_builder = self.feature_builder.futures_builder
        if futures_builder is not None:
            futures_builder.validate_window(ts, self.contract.lookback_bars)
            barrier_context = futures_builder.build_barrier_context(
                ts,
                stop_mult=self.contract.barrier_stop_mult,
                target_mult=self.contract.barrier_target_mult,
                vol_span=self.contract.barrier_vol_span,
                horizon_bars=self.contract.barrier_horizon_bars,
                cost_floor_frac=self.contract.barrier_cost_floor_frac,
            )
        return ModelInputSnapshot(
            market=market,
            regime=regime,
            barrier_context=barrier_context,
        )

    def build_many(
        self,
        timestamps: Sequence[datetime],
    ) -> dict[datetime, ModelInputSnapshot]:
        if not timestamps:
            return {}

        history = self.feature_builder.build_history()
        if tuple(history.feature_names) != self.contract.feature_names:
            raise ValueError(
                "feature builder output does not match the artifact feature contract"
            )
        lookback = self.contract.lookback_bars
        market_windows = history.window_view(lookback)
        market_ts = pd.DatetimeIndex(timestamps)
        positions = history.index.get_indexer(market_ts)

        futures_windows = None
        futures_positions = None
        barrier_history = None
        futures_builder = self.feature_builder.futures_builder
        if futures_builder is not None:
            futures_index, futures_history = futures_builder.build_history()
            futures_positions = futures_index.get_indexer(market_ts)
            futures_windows = _window_view_2d(
                futures_history.astype(np.float32, copy=False),
                lookback,
            )
            futures_eligible = futures_builder.eligible_positions(lookback)
            _, barrier_history = futures_builder.build_barrier_context_history(
                stop_mult=self.contract.barrier_stop_mult,
                target_mult=self.contract.barrier_target_mult,
                vol_span=self.contract.barrier_vol_span,
                horizon_bars=self.contract.barrier_horizon_bars,
                cost_floor_frac=self.contract.barrier_cost_floor_frac,
            )
        else:
            futures_eligible = None

        snapshots: dict[datetime, ModelInputSnapshot] = {}
        for idx, ts in enumerate(timestamps):
            pos = int(positions[idx])
            if pos < lookback - 1:
                continue
            futures = None
            barrier_context = None
            if futures_windows is not None and futures_positions is not None:
                fut_pos = int(futures_positions[idx])
                if fut_pos < lookback - 1:
                    continue
                if futures_eligible is not None and not bool(futures_eligible[fut_pos]):
                    continue
                futures = np.ascontiguousarray(
                    futures_windows[fut_pos - lookback + 1],
                    dtype=np.float32,
                )
                if barrier_history is not None and fut_pos >= 0:
                    row = barrier_history[fut_pos]
                    barrier_context = BarrierContext(
                        sigma=float(row[0]),
                        stop_return=float(row[1]),
                        target_return=float(row[2]),
                    )

            regime = self.regime_builder.build(ts) if self.regime_builder is not None else None
            snapshots[ts] = ModelInputSnapshot(
                market=MarketBatch(
                    ts=ts,
                    symbols=history.symbols,
                    candle_features=np.ascontiguousarray(
                        market_windows[pos - lookback + 1],
                        dtype=np.float32,
                    ),
                    feature_names=history.feature_names,
                    futures=futures,
                ),
                regime=regime,
                barrier_context=barrier_context,
            )
        return snapshots


__all__ = [
    "ModelInputContract",
    "ModelInputSnapshot",
    "ModelInputBuilder",
    "RegimeInput",
]
