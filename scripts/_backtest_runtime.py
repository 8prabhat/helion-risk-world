"""Shared backtest/paper runtime helpers."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from _bootstrap import ensure_src_path

ensure_src_path()

from helion_risk_world.backtesting.backtest_engine import BacktestStep
from helion_risk_world.config.execution_config import CostModelConfig
from helion_risk_world.config.loaders import (
    loss_weights_from_mapping as loss_weights_from_cfg,
    model_config_from_mapping as model_config_from_cfg,
    training_config_from_mapping as training_config_from_cfg,
    walk_forward_from_mapping as walk_forward_from_cfg,
)
from helion_risk_world.data.feature_builder import (
    FeatureBuilder,
    InMemoryMarketDataSource,
)
from helion_risk_world.data.market_window_builder import CANDLE_FEATURE_NAMES
from helion_risk_world.data.model_input_builder import ModelInputSnapshot
from helion_risk_world.data.parquet_source import (
    ParquetMarketDataSource,
    infer_interval_from_path,
    load_ohlcv_parquet,
    prepare_ohlcv_frame,
)
from helion_risk_world.data.primitives import regime_label, simple_returns
from helion_risk_world.execution.instrument_specs import resolve_instrument_spec
from helion_risk_world.heads.regime_head import REGIME_CLASSES
from helion_risk_world.inference import ForecasterPredictor
from helion_risk_world.losses.composite_loss import ForecasterLoss
from helion_risk_world.model import HRWForecaster
from helion_risk_world.runtime import (
    ModelRuntime,
    build_runtime_inputs,
    load_model_runtime,
    predict_snapshot,
)
from helion_risk_world.schemas.execution_schema import ExecutionState
from helion_risk_world.schemas.market_schema import MarketCandle
from helion_risk_world.schemas.prediction_schema import (
    BarrierProbabilities,
    HorizonPrediction,
    ModelPrediction,
)
from helion_risk_world.training.split_manifest import ChronoSplitManifest
from helion_risk_world.training.trainer import ForecastBatch, HRWTrainer

REAL_UNIVERSE = (
    "BANKNIFTY",
    "NIFTY",
    "FINNIFTY",
    "HDFCBANK",
    "ICICIBANK",
    "SBIN",
    "AXISBANK",
    "KOTAKBANK",
)
TRADE_SYMBOL = "BANKNIFTY_FUT_continuous"
_EXECUTION_CFG = CostModelConfig()

def _candles(universe: tuple[str, ...], n: int = 260) -> dict[str, list[MarketCandle]]:
    rng = np.random.default_rng(11)
    start = datetime(2026, 6, 25, 9, 20)
    out: dict[str, list[MarketCandle]] = {}
    for symbol, base in zip(universe, 100 + rng.uniform(0, 50, len(universe)), strict=False):
        rows: list[MarketCandle] = []
        price = float(base)
        for i in range(n):
            ts = start + timedelta(minutes=5 * i)
            drift = 0.009 * float(np.sin(i / 12.0)) + 0.0004 * float(rng.normal())
            price = max(1.0, price * (1.0 + drift))
            rows.append(
                MarketCandle(
                    symbol=symbol,
                    ts=ts,
                    available_at=ts,
                    open=price,
                    high=price * 1.003,
                    low=price * 0.997,
                    close=price,
                    volume=1000 + i,
                    oi=5000 + i * 4,
                )
            )
        out[symbol] = rows
    return out


def _prediction(
    ts: datetime,
    momentum: float,
    sigma: float,
    horizon: int,
    *,
    symbol: str,
) -> ModelPrediction:
    sigma = max(sigma, 1e-4)
    mean = float(np.clip(momentum, -0.05, 0.05))
    quantiles = {
        0.1: mean - 2 * sigma,
        0.25: mean - sigma,
        0.5: mean,
        0.75: mean + sigma,
        0.9: mean + 2 * sigma,
    }
    stop = float(np.clip(0.35 - momentum / max(4 * sigma, 1e-4), 0.05, 0.8))
    target = float(np.clip(0.35 + momentum / max(4 * sigma, 1e-4), 0.05, 0.8))
    timeout = max(0.0, 1.0 - stop - target)
    total = stop + target + timeout
    barrier = BarrierProbabilities(
        stop=stop / total,
        target=target / total,
        timeout=timeout / total,
    )
    horizon_prediction = HorizonPrediction(
        horizon_bars=horizon,
        return_quantiles=quantiles,
        volatility=sigma,
    )
    return ModelPrediction(
        symbol=symbol,
        ts=ts,
        horizon_preds=[horizon_prediction],
        barrier=barrier,
        mae=2 * sigma,
        sigma_H=sigma,
        stop_return=-2.0 * sigma,
        target_return=2.0 * sigma,
        epistemic=0.0,
        aleatoric=sigma,
        ood_score=0.0,
    )


def _market(symbol: str, ts: datetime, price: float) -> ExecutionState:
    spec = resolve_instrument_spec(symbol, _EXECUTION_CFG)
    spread = (
        _EXECUTION_CFG.default_spread_ticks * spec.tick_size
        if spec is not None and spec.tick_size is not None
        else None
    )
    if spread is None:
        half_frac = _EXECUTION_CFG.half_spread_bps
        bid = price * (1.0 - half_frac)
        ask = price * (1.0 + half_frac)
    else:
        half_spread = spread / 2.0
        bid = price - half_spread
        ask = price + half_spread
    return ExecutionState(
        symbol=symbol,
        ts=ts,
        available_at=ts,
        bid=bid,
        ask=ask,
        spread=spread,
    )


def _load_tradeable_frame(data_dir: str | Path, base_interval: str) -> pd.DataFrame:
    path = Path(data_dir) / "ohlcv" / f"{TRADE_SYMBOL}_{base_interval}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing tradeable futures parquet: {path}")
    source_interval = infer_interval_from_path(path, fallback=base_interval)
    return prepare_ohlcv_frame(
        load_ohlcv_parquet(path),
        source_interval=source_interval,
        target_interval=base_interval,
    )


def _aligned_trade_grid(
    dc,
    source,
    *,
    data_dir: str | Path | None = None,
) -> tuple[str, list[datetime], np.ndarray, np.ndarray]:
    fallback_symbol = dc.universe[0]
    if data_dir is not None:
        trade_symbol = TRADE_SYMBOL
        trade_frame = _load_tradeable_frame(data_dir, dc.base_interval)
    else:
        frames = source.aligned_frames()
        trade_symbol = TRADE_SYMBOL if TRADE_SYMBOL in frames else fallback_symbol
        trade_frame = frames[trade_symbol]

    common_index = source.timestamp_index().intersection(trade_frame.index)
    if len(common_index) == 0:
        raise ValueError("no common timestamps between context universe and tradeable execution series")
    aligned = trade_frame.loc[common_index]
    opens = aligned["open"].to_numpy(dtype=float)
    closes = aligned["close"].to_numpy(dtype=float)
    timestamps = [ts.to_pydatetime() for ts in common_index]
    return trade_symbol, timestamps, opens, closes


def _build_steps_from_source(
    dc,
    source,
    horizon: int,
    *,
    data_dir: str | Path | None = None,
) -> list[BacktestStep]:
    feature_builder = FeatureBuilder(dc, source)
    ret_col = CANDLE_FEATURE_NAMES.index("log_return")
    trade_symbol, timestamps, opens, closes = _aligned_trade_grid(dc, source, data_dir=data_dir)
    steps: list[BacktestStep] = []
    for i in range(dc.lookback_bars - 1, len(timestamps) - horizon):
        ts = timestamps[i]
        batch = feature_builder.build_window(ts)
        recent = batch.candle_features[0, -6:, ret_col]
        momentum = float(recent.mean())
        sigma = float(max(recent.std(), 1e-4))
        fill_ts = timestamps[i + 1]
        realized = closes[i + horizon] / opens[i + 1] - 1.0
        steps.append(
            BacktestStep(
                prediction=_prediction(ts, momentum, sigma, horizon, symbol=trade_symbol),
                market=_market(trade_symbol, ts, float(closes[i])),
                execution_market=_market(trade_symbol, fill_ts, float(opens[i + 1])),
                realized_return=realized,
                label_realized_at=timestamps[i + horizon],
                # Per-bar settlement legs (settlement-bug fix, 2026-07-18 — see
                # BacktestStep.fill_to_mark_return): carried positions mark
                # close[i] → open[i+1], post-fill positions mark open[i+1] → close[i+1].
                carry_return=float(opens[i + 1] / closes[i] - 1.0),
                fill_to_mark_return=float(closes[i + 1] / opens[i + 1] - 1.0),
            )
        )
    return steps


def _fit_demo_predictor(
    dc,
    horizon: int,
    cfg: dict,
) -> tuple[ForecasterPredictor, InMemoryMarketDataSource]:
    candles = _candles(dc.universe)
    source = InMemoryMarketDataSource(candles=candles)
    feature_builder = FeatureBuilder(dc, source)
    primary = candles[dc.universe[0]]
    closes = np.array([row.close for row in primary], dtype=float)
    decision_idx = list(range(dc.lookback_bars - 1, len(primary) - horizon))
    split = int(0.6 * len(decision_idx))
    embargo = horizon
    train_idx = decision_idx[:split]

    features, returns, directions, regimes, vols, barriers = [], [], [], [], [], []
    for i in train_idx:
        ts = primary[i].ts
        features.append(feature_builder.build_window(ts).candle_features)
        future = closes[i + 1 : i + horizon + 1]
        forward_return = float(future[-1] / future[0] - 1.0)
        future_rets = simple_returns(future)[1:]
        sigma = float(np.std(future_rets)) if future_rets.size else abs(forward_return)
        returns.append(forward_return)
        directions.append(np.digitize([forward_return], [-0.001, 0.001])[0])
        vols.append(max(sigma, 1e-6))
        regimes.append(REGIME_CLASSES.index(regime_label(forward_return, max(sigma, 1e-6))))
        barriers.append(1 if forward_return > 0.003 else 0 if forward_return < -0.003 else 2)

    batch = ForecastBatch(
        features=torch.tensor(np.stack(features), dtype=torch.float32),
        forward_return=torch.tensor(returns, dtype=torch.float32),
        direction=torch.tensor(directions, dtype=torch.long),
        regime=torch.tensor(regimes, dtype=torch.long),
        realized_vol=torch.tensor(vols, dtype=torch.float32),
        barrier=torch.tensor(barriers, dtype=torch.long),
    )
    model = HRWForecaster(n_features=len(CANDLE_FEATURE_NAMES), cfg=model_config_from_cfg(cfg))
    train_cfg = training_config_from_cfg(cfg)
    train_cfg = replace(train_cfg, embargo_bars=max(train_cfg.embargo_bars, embargo))
    HRWTrainer(model, ForecasterLoss(weights=loss_weights_from_cfg(cfg)), train_cfg).fit([batch])
    model.fit_ood(batch.features)
    return ForecasterPredictor(model, horizon_bars=horizon), source


def _build_model_steps_from_source(
    dc,
    source,
    horizon: int,
    *,
    predictor: ForecasterPredictor | None = None,
    runtime: ModelRuntime | None = None,
    data_dir: str | Path | None = None,
) -> list[BacktestStep]:
    inputs = None
    if runtime is not None:
        inputs = build_runtime_inputs(
            dc,
            source,
            data_dir=data_dir,
            runtime=runtime,
        )
    elif predictor is None:
        raise ValueError("predictor or runtime is required to build model-backed steps")
    feature_builder = FeatureBuilder(dc, source) if inputs is None else None
    trade_symbol, timestamps, opens, closes = _aligned_trade_grid(dc, source, data_dir=data_dir)
    active_predictor = runtime.predictor if runtime is not None else predictor
    assert active_predictor is not None
    precomputed_snapshots = (
        _precompute_runtime_snapshots(inputs, timestamps)
        if inputs is not None
        else None
    )
    steps: list[BacktestStep] = []
    for i in range(dc.lookback_bars - 1, len(timestamps) - horizon):
        ts = timestamps[i]
        if inputs is None:
            market = feature_builder.build_window(ts)  # type: ignore[union-attr]
            pred = active_predictor.predict_one(
                torch.tensor(market.candle_features, dtype=torch.float32),
                trade_symbol,
                ts,
            )
        else:
            snapshot = precomputed_snapshots.get(ts) if precomputed_snapshots is not None else None
            if snapshot is None:
                try:
                    snapshot = inputs.build(ts)
                except ValueError:
                    continue
            pred = predict_snapshot(runtime, snapshot, symbol=trade_symbol, ts=ts)  # type: ignore[arg-type]
        fill_ts = timestamps[i + 1]
        realized = closes[i + horizon] / opens[i + 1] - 1.0
        steps.append(
            BacktestStep(
                prediction=pred,
                market=_market(trade_symbol, ts, float(closes[i])),
                execution_market=_market(trade_symbol, fill_ts, float(opens[i + 1])),
                realized_return=realized,
                label_realized_at=timestamps[i + horizon],
                # Per-bar settlement legs (settlement-bug fix, 2026-07-18 — see
                # BacktestStep.fill_to_mark_return): carried positions mark
                # close[i] → open[i+1], post-fill positions mark open[i+1] → close[i+1].
                carry_return=float(opens[i + 1] / closes[i] - 1.0),
                fill_to_mark_return=float(closes[i + 1] / opens[i + 1] - 1.0),
            )
        )
    return steps


def _precompute_runtime_snapshots(
    inputs,
    timestamps: list[datetime],
) -> dict[datetime, ModelInputSnapshot]:
    return inputs.build_many(timestamps)


def _filter_steps_to_runtime_split(
    steps: list[BacktestStep],
    runtime: ModelRuntime,
    *,
    split: str = "test",
) -> list[BacktestStep]:
    payload = runtime.split_manifest
    if not isinstance(payload, dict):
        raise ValueError(
            "model artifact is missing split_manifest metadata; retrain with the updated train.py "
            "before running real model backtests"
        )
    manifest = ChronoSplitManifest.from_metadata(payload)
    filtered = [
        step for step in steps if bool(manifest.contains(pd.Timestamp(step.prediction.ts), split))
    ]
    if not filtered:
        raise ValueError(
            f"artifact split {split!r} contains no backtest steps after timestamp filtering"
        )
    return filtered


def _build_steps(
    dc, horizon: int, cfg: dict | None = None, *, walk_forward: bool = False
) -> list[BacktestStep]:
    # Walk-forward's split_indices needs first_test_start = min_train + val + 2*embargo
    # to stay below the total step count (see WalkForward.split_indices). Only size the
    # extra buffer off the CONFIGURED embargo (walk_forward_from_cfg) when walk-forward
    # mode is actually requested -- embargo_bars now defaults to the management horizon
    # (feature/label overhaul Phase 4a: 192 bars) regardless of whether `--walk-forward`
    # is passed, so applying this buffer unconditionally made every `--all-strategies`
    # demo run (walk-forward or not) generate ~4x more candles than needed. `horizon`
    # here is one strategy's own decision_horizon_bars (often small, e.g. 3-12) and is
    # NOT the same thing as embargo_bars. 4*embargo+50 comfortably clears
    # WalkForward.split_indices' minimum for the default n_folds=5 (verified: solving
    # first_test_start < n_samples for n_samples in terms of embargo).
    buffer = 80
    if walk_forward and cfg is not None:
        embargo_bars = walk_forward_from_cfg(cfg).embargo_bars
        buffer = max(80, 4 * embargo_bars + 50)
    n = max(260, dc.lookback_bars + horizon + buffer)
    source = InMemoryMarketDataSource(candles=_candles(dc.universe, n=n))
    return _build_steps_from_source(dc, source, horizon)


def _load_runtime_for_horizon(
    model_path: str | Path, horizon: int, *, persist_state: bool = True
) -> ModelRuntime:
    runtime = load_model_runtime(model_path, persist_state=persist_state)
    if horizon not in runtime.available_horizons:
        raise ValueError(
            "artifact horizons "
            f"{runtime.available_horizons} do not include requested strategy horizon {horizon}"
        )
    if int(horizon) != int(runtime.horizon_bars):
        raise ValueError(
            "strategy horizon must match artifact management/barrier horizon: "
            f"strategy={horizon} artifact_management={runtime.horizon_bars}. "
            "Train a matching barrier model before using shorter horizons."
        )
    contract_horizon = int(getattr(runtime.contract, "barrier_horizon_bars", runtime.horizon_bars))
    if contract_horizon != int(runtime.horizon_bars):
        raise ValueError(
            "artifact input contract barrier horizon does not match artifact management "
            f"horizon: contract={contract_horizon} artifact={runtime.horizon_bars}"
        )
    return runtime


def _build_real_steps(
    dc,
    data_dir: str,
    horizon: int,
    model_path: str | Path | None = None,
    *,
    persist_state: bool = True,
    eval_split: str = "test",
) -> list[BacktestStep]:
    source = ParquetMarketDataSource(
        data_dir=data_dir,
        universe=dc.universe,
        base_interval=dc.base_interval,
    )
    if model_path:
        runtime = _load_runtime_for_horizon(model_path, horizon, persist_state=persist_state)
        steps = _build_model_steps_from_source(
            dc,
            source,
            horizon,
            runtime=runtime,
            data_dir=data_dir,
        )
        return _filter_steps_to_runtime_split(steps, runtime, split=eval_split)
    return _build_steps_from_source(dc, source, horizon, data_dir=data_dir)


def predictor_kind_for_run(args: object) -> str:
    if getattr(args, "real"):
        if getattr(args, "model"):
            if not getattr(args, "model_path", None):
                raise ValueError("--model with --real requires --model-path <artifact>")
            return "model_artifact"
        return "heuristic_momentum_REAL"
    if getattr(args, "model"):
        return (
            "model_artifact_demo"
            if getattr(args, "model_path", None)
            else "forecaster_demo_split"
        )
    return "heuristic"


def build_steps_for_run(dc, cfg: dict, args: object, horizon: int) -> tuple[list[BacktestStep], str]:
    predictor_kind = predictor_kind_for_run(args)
    persist_state = bool(getattr(args, "persist_state", True))
    if getattr(args, "real"):
        data_dir = getattr(args, "data_dir", None)
        if not data_dir:
            raise ValueError("--real requires --data-dir <ohlcv parent>")
        steps = _build_real_steps(
            dc,
            data_dir,
            horizon,
            getattr(args, "model_path", None) if getattr(args, "model") else None,
            persist_state=persist_state,
            eval_split=getattr(args, "eval_split", None) or "test",
        )
        return steps, predictor_kind

    if getattr(args, "model"):
        model_path = getattr(args, "model_path", None)
        if model_path:
            runtime = _load_runtime_for_horizon(model_path, horizon, persist_state=persist_state)
            source = InMemoryMarketDataSource(candles=_candles(dc.universe))
            return (
                _build_model_steps_from_source(
                    dc,
                    source,
                    horizon,
                    runtime=runtime,
                ),
                predictor_kind,
            )

        predictor, source = _fit_demo_predictor(dc, horizon, cfg)
        return (
            _build_model_steps_from_source(
                dc,
                source,
                horizon,
                predictor=predictor,
            ),
            predictor_kind,
        )

    return (
        _build_steps(dc, horizon, cfg, walk_forward=bool(getattr(args, "walk_forward", False))),
        predictor_kind,
    )


__all__ = [
    "REAL_UNIVERSE",
    "TRADE_SYMBOL",
    "build_steps_for_run",
    "predictor_kind_for_run",
]
