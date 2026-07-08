"""Shared feature builder + primitives (SPEC.md §10, §27, Day 3).

Proves: primitive correctness; window shape/feature names; build_window assembly; the SAME builder
feeds train and backtest (parity); and no portfolio field reaches the market features (leakage).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from helion_risk_world.config.data_config import DataConfig
from helion_risk_world.data import primitives as P
from helion_risk_world.data.feature_builder import FeatureBuilder, InMemoryMarketDataSource
from helion_risk_world.data.leakage_checks import MARKET_FEATURE_NAMES, PORTFOLIO_FEATURE_NAMES
from helion_risk_world.data.market_window_builder import CANDLE_FEATURE_NAMES, MarketWindowBuilder
from helion_risk_world.schemas.market_schema import MarketCandle
from helion_risk_world.schemas.option_chain_schema import OptionContractSnapshot, OptionType

START = datetime(2026, 6, 25, 9, 20)


# ---------------- primitives ----------------
def test_log_and_simple_returns() -> None:
    close = np.array([100.0, 110.0, 99.0])
    lr = P.log_returns(close)
    sr = P.simple_returns(close)
    assert lr[0] == 0.0 and sr[0] == 0.0
    assert sr[1] == pytest.approx(0.1)
    assert lr[1] == pytest.approx(np.log(1.1))


def test_log_returns_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        P.log_returns(np.array([100.0, 0.0, 90.0]))


def test_oi_change_and_realized_vol_shapes() -> None:
    oi = np.array([10.0, 12.0, 9.0, 9.0])
    d = P.oi_change(oi)
    assert d[0] == 0.0 and d[1] == 2.0 and d[2] == -3.0
    rv = P.realized_vol(np.array([100.0, 101.0, 102.0, 103.0, 104.0]), window=3)
    assert rv.shape == (5,)
    assert np.isnan(rv[:2]).all() and np.isfinite(rv[2:]).all()


def test_atr_positive_and_time_of_day_bounds() -> None:
    high = np.array([10.0, 11.0, 12.0])
    low = np.array([9.0, 9.5, 10.0])
    close = np.array([9.5, 10.5, 11.0])
    assert np.isfinite(P.atr(high, low, close, window=2)[1:]).all()
    assert 0.0 <= P.time_of_day(datetime(2026, 6, 25, 9, 15)) <= 1.0
    assert P.time_of_day(datetime(2026, 6, 25, 9, 15)) == 0.0
    assert P.time_of_day(datetime(2026, 6, 25, 15, 30)) == 1.0


# ---------------- window builder ----------------
def _candles(symbol: str, n: int = 30, base: float = 100.0) -> list[MarketCandle]:
    rows = []
    for i in range(n):
        ts = START + timedelta(minutes=5 * i)
        close = base + np.sin(i / 3.0) * 2.0 + i * 0.1  # deterministic, strictly positive
        rows.append(
            MarketCandle(
                symbol=symbol, ts=ts, available_at=ts,
                open=close, high=close * 1.01, low=close * 0.99, close=close,
                volume=1000 + i, oi=5000 + i * 10,
            )
        )
    return rows


def test_window_builder_shape_and_names() -> None:
    feats, names = MarketWindowBuilder().build(_candles("BANKNIFTY"))
    assert names == CANDLE_FEATURE_NAMES
    assert feats.shape == (30, len(CANDLE_FEATURE_NAMES))
    assert np.isfinite(feats).all()  # warm-up NaNs filled with 0.0
    assert set(names).issubset(MARKET_FEATURE_NAMES)  # all market-plane
    assert set(names).isdisjoint(PORTFOLIO_FEATURE_NAMES)


def test_window_builder_rejects_unsorted() -> None:
    rows = _candles("X", 5)
    rows[2], rows[3] = rows[3], rows[2]
    with pytest.raises(ValueError):
        MarketWindowBuilder().build(rows)


# ---------------- FeatureBuilder.build_window ----------------
def _cfg() -> DataConfig:
    return DataConfig(universe=("BANKNIFTY", "NIFTY"), lookback_bars=30, n_strikes=2)


def _chain(spot: float = 100.0) -> list[OptionContractSnapshot]:
    ts = START + timedelta(minutes=5 * 29)
    chain = []
    for i in range(-3, 4):
        k = round(spot) + i  # 1-wide grid for the toy underlying
        for opt in (OptionType.CALL, OptionType.PUT):
            chain.append(
                OptionContractSnapshot(
                    underlying="BANKNIFTY", strike=float(k), opt_type=opt, ts=ts, available_at=ts,
                    open=1, high=2, low=0.5, close=1.5, volume=10, oi=100 + i, iv=0.2, dte=2.0,
                )
            )
    return chain


def test_build_window_assembles_batch_with_surface() -> None:
    cfg = _cfg()
    candles = {s: _candles(s) for s in cfg.universe}
    ts = candles["BANKNIFTY"][-1].ts
    spot = candles["BANKNIFTY"][-1].close
    source = InMemoryMarketDataSource(candles=candles, chains={"BANKNIFTY": _chain(spot)})
    batch = FeatureBuilder(cfg, source).build_window(ts)

    assert batch.candle_features.shape == (2, 30, len(CANDLE_FEATURE_NAMES))  # [A, L, F]
    assert batch.symbols == ("BANKNIFTY", "NIFTY")
    assert batch.surface is not None
    assert len(batch.surface.strikes) == 5  # 2N+1
    # Leakage: feature names carry no portfolio field.
    assert set(batch.feature_names).isdisjoint(PORTFOLIO_FEATURE_NAMES)


def test_build_window_train_backtest_parity() -> None:
    """The SAME builder/source produces identical features on repeated calls (offline==online)."""
    cfg = _cfg()
    candles = {s: _candles(s) for s in cfg.universe}
    ts = candles["BANKNIFTY"][-1].ts
    source = InMemoryMarketDataSource(candles=candles)
    builder = FeatureBuilder(cfg, source)
    a = builder.build_window(ts)
    b = builder.build_window(ts)
    assert np.array_equal(a.candle_features, b.candle_features)
    assert a.feature_names == b.feature_names

    # And the builder column matches a direct primitive computation (single source of truth).
    close = np.array([c.close for c in candles["BANKNIFTY"]])
    col = a.feature_names.index("log_return")
    assert np.allclose(a.candle_features[0, :, col], np.nan_to_num(P.log_returns(close)))


def test_build_window_excludes_future_candles() -> None:
    cfg = _cfg()
    candles = {s: _candles(s) for s in cfg.universe}
    ts = candles["BANKNIFTY"][-5].ts  # decide 5 bars before the end
    source = InMemoryMarketDataSource(candles=candles)
    batch = FeatureBuilder(cfg, source).build_window(ts)
    # Only candles with available_at <= ts are visible -> at most 26 of the 30 bars.
    assert batch.candle_features.shape[1] <= 26


def test_build_window_raises_on_insufficient_history() -> None:
    cfg = _cfg()
    candles = {s: _candles(s, n=1) for s in cfg.universe}
    ts = candles["BANKNIFTY"][-1].ts
    source = InMemoryMarketDataSource(candles=candles)
    with pytest.raises(ValueError):
        FeatureBuilder(cfg, source).build_window(ts)


# ---------------- breadth/dispersion (feature/label overhaul Phase 2) ----------------

def _directional_candles(symbol: str, direction: float, n: int = 30) -> list[MarketCandle]:
    """A steadily trending series (up if direction>0, down if direction<0)."""
    rows = []
    for i in range(n):
        ts = START + timedelta(minutes=5 * i)
        close = 100.0 + direction * i * 0.5
        rows.append(
            MarketCandle(
                symbol=symbol, ts=ts, available_at=ts,
                open=close, high=close + 0.1, low=close - 0.1, close=close,
                volume=1000.0, oi=0.0,
            )
        )
    return rows


def test_breadth_dispersion_reflect_universe_direction() -> None:
    """3 non-primary symbols: 2 trending up, 1 trending down -> breadth ~ 2/3, and
    dispersion > 0 since they don't all move together."""
    cfg = DataConfig(universe=("BANKNIFTY", "NIFTY", "FINNIFTY", "HDFCBANK"), lookback_bars=30)
    candles = {
        "BANKNIFTY": _directional_candles("BANKNIFTY", direction=1.0),  # primary, excluded
        "NIFTY": _directional_candles("NIFTY", direction=1.0),
        "FINNIFTY": _directional_candles("FINNIFTY", direction=1.0),
        "HDFCBANK": _directional_candles("HDFCBANK", direction=-1.0),
    }
    ts = candles["BANKNIFTY"][-1].ts
    source = InMemoryMarketDataSource(candles=candles)
    batch = FeatureBuilder(cfg, source).build_window(ts)

    breadth_idx = batch.feature_names.index("breadth")
    dispersion_idx = batch.feature_names.index("dispersion")
    last_breadth = batch.candle_features[:, -1, breadth_idx]
    last_dispersion = batch.candle_features[:, -1, dispersion_idx]

    # 2 of 3 non-primary symbols trending up -> breadth ~ 0.667.
    assert last_breadth[0] == pytest.approx(2.0 / 3.0, abs=1e-6)
    assert last_dispersion[0] > 0.0

    # Market-wide scalar: identical across every asset's row at this timestamp.
    assert np.allclose(last_breadth, last_breadth[0])
    assert np.allclose(last_dispersion, last_dispersion[0])


def test_breadth_uniform_when_all_non_primary_agree() -> None:
    cfg = DataConfig(universe=("BANKNIFTY", "NIFTY", "FINNIFTY"), lookback_bars=30)
    candles = {
        "BANKNIFTY": _directional_candles("BANKNIFTY", direction=-1.0),
        "NIFTY": _directional_candles("NIFTY", direction=1.0),
        "FINNIFTY": _directional_candles("FINNIFTY", direction=1.0),
    }
    ts = candles["BANKNIFTY"][-1].ts
    source = InMemoryMarketDataSource(candles=candles)
    batch = FeatureBuilder(cfg, source).build_window(ts)
    breadth_idx = batch.feature_names.index("breadth")
    assert batch.candle_features[0, -1, breadth_idx] == pytest.approx(1.0, abs=1e-6)


def test_build_window_and_build_history_breadth_consistent() -> None:
    """build_window (per-timestamp) and build_history (batch) must agree on breadth."""
    cfg = DataConfig(universe=("BANKNIFTY", "NIFTY", "FINNIFTY"), lookback_bars=30)
    candles = {
        "BANKNIFTY": _directional_candles("BANKNIFTY", direction=1.0),
        "NIFTY": _directional_candles("NIFTY", direction=1.0),
        "FINNIFTY": _directional_candles("FINNIFTY", direction=-1.0),
    }
    source = InMemoryMarketDataSource(candles=candles)
    builder = FeatureBuilder(cfg, source)
    ts = candles["BANKNIFTY"][-1].ts
    window_batch = builder.build_window(ts)
    history = builder.build_history()
    breadth_idx = history.feature_names.index("breadth")
    window_breadth_idx = window_batch.feature_names.index("breadth")

    history_last_breadth = history.candle_features[-1, 0, breadth_idx]
    window_last_breadth = window_batch.candle_features[0, -1, window_breadth_idx]
    assert history_last_breadth == pytest.approx(window_last_breadth, abs=1e-4)
