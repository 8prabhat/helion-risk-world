"""Data layer: builders, primitives, datasets, quality + leakage checks."""

from helion_risk_world.data.dataset import HRWWindowDataset
from helion_risk_world.data.feature_builder import (
    FeatureBuilder,
    InMemoryMarketDataSource,
    MarketBatch,
    MarketDataSource,
)
from helion_risk_world.data.leakage_checks import (
    MARKET_FEATURE_NAMES,
    PORTFOLIO_FEATURE_NAMES,
    LeakageError,
    assert_label_in_future,
    assert_no_portfolio_in_market,
    assert_point_in_time,
)
from helion_risk_world.data.market_window_builder import (
    CANDLE_FEATURE_NAMES,
    MarketWindowBuilder,
)
from helion_risk_world.data.option_surface_builder import (
    OptionSurfaceBuilder,
    infer_strike_step,
)
from helion_risk_world.data.parquet_source import ParquetMarketDataSource
from helion_risk_world.data.portfolio_state_builder import NamedProfile, PortfolioStateBuilder

__all__ = [
    "HRWWindowDataset",
    "FeatureBuilder",
    "MarketBatch",
    "MarketDataSource",
    "InMemoryMarketDataSource",
    "MarketWindowBuilder",
    "CANDLE_FEATURE_NAMES",
    "OptionSurfaceBuilder",
    "infer_strike_step",
    "ParquetMarketDataSource",
    "PortfolioStateBuilder",
    "NamedProfile",
    "MARKET_FEATURE_NAMES",
    "PORTFOLIO_FEATURE_NAMES",
    "LeakageError",
    "assert_label_in_future",
    "assert_no_portfolio_in_market",
    "assert_point_in_time",
]
