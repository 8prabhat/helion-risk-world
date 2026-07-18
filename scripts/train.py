"""Train and save the forecaster workflow artifact.

Demo mode is fully self-contained. Real-data mode expects:
  1. OHLCV parquets under ``<data-dir>/ohlcv/<SYMBOL>_<interval>.parquet``
     (native ``base_interval`` files or ``1min`` files that can be resampled)
  2. a labels parquet produced by ``scripts/label.py``

Usage:
    python scripts/train.py --config configs/v1.yaml --demo
    python scripts/train.py --config configs/v1.yaml --data-dir data --labels-path data/processed/labels.parquet
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from _common import log, setup
from helion_risk_world.barrier_context import BarrierSpec, barrier_context_series
from helion_risk_world.config.loaders import (
    data_config_from_mapping as data_config_from_cfg,
    execution_config_from_mapping as execution_config_from_cfg,
    loss_weights_from_mapping as loss_weights_from_cfg,
    management_horizon_from_mapping as management_horizon_from_cfg,
    model_config_from_mapping as model_config_from_cfg,
    training_config_from_mapping as training_config_from_cfg,
)

from helion_risk_world.backtesting.walk_forward import WalkForward
from helion_risk_world.data.capability_profile import DataCapabilityProfile
from helion_risk_world.data.feature_builder import FeatureBuilder, InMemoryMarketDataSource
from helion_risk_world.data.parquet_source import ParquetMarketDataSource
from helion_risk_world.data.model_input_builder import ModelInputBuilder, ModelInputContract
from helion_risk_world.data.provenance import validate_upstox_only_sources
from helion_risk_world.data.primitives import simple_returns, state_regime_label
from helion_risk_world.data.market_window_builder import CANDLE_FEATURE_NAMES
from helion_risk_world.data.regime_builder import REGIME_CONTEXT_FEATURES, featurize_regime
from helion_risk_world.heads.regime_head import REGIME_CLASSES
from helion_risk_world.heads.return_head import DEFAULT_QUANTILES
from helion_risk_world.losses.composite_loss import ForecasterLoss
from helion_risk_world.losses.world_model_loss import WorldModelLoss
from helion_risk_world.encoders.option_surface_encoder import SurfaceTensors
from helion_risk_world.model import HRWForecaster, HRWWorldModel
from helion_risk_world.prediction_calibration import fit_prediction_calibration
from helion_risk_world.schemas.label_schema import (
    BARRIER_COST_FLOOR_COLUMN,
    BARRIER_SIGMA_COLUMN,
    BARRIER_STOP_MULT_COLUMN,
    BARRIER_STOP_RETURN_COLUMN,
    BARRIER_TARGET_MULT_COLUMN,
    BARRIER_TARGET_RETURN_COLUMN,
    BARRIER_VOL_SPAN_COLUMN,
    Barrier,
    horizon_mae_column,
    horizon_mfe_column,
    horizon_return_column,
    horizon_volatility_column,
)
from helion_risk_world.schemas.market_schema import EventType, MarketCandle, Regime
from helion_risk_world.training.artifacts import (
    DEFAULT_TARGET_SYMBOL,
    save_forecaster_artifact,
    save_world_model_artifact,
)
from helion_risk_world.training.checkpoint_metrics import trading_utility_loss
from helion_risk_world.training.opportunity_weighting import (
    OpportunityWeightAudit,
    compute_management_opportunity_weights,
)
from helion_risk_world.training.pretrain_market_state import LatentPair, MarketStatePretrainer
from helion_risk_world.training.split_manifest import ChronoSplitManifest
from helion_risk_world.training.train_heads import HeadTrainer
from helion_risk_world.training.train_world_model import WorldModelTrainer
from helion_risk_world.training.trainer import ForecastBatch, HRWTrainer

from alpha_data.io.paths import DataPaths as AlphaDataPaths

def _futures_data_available(universe: tuple[str, ...], base_interval: str) -> bool:
    """Whether alpha_data's real futures-microstructure parquet exists for this
    universe's primary underlying -- what actually gates whether
    ``AlphaDataFuturesWindowBuilder`` can be constructed (Phase 2 migration; this
    replaced the old ``data/processed/banknifty_5min.parquet``-existence check, which
    stopped being produced once ``scripts/assemble_data.py`` was deleted)."""
    path = AlphaDataPaths().features / f"{universe[0]}_FUT_futures_microstructure_{base_interval}.parquet"
    return path.exists()


def _option_surface_data_available(universe: tuple[str, ...], base_interval: str) -> bool:
    """Whether alpha_data has ingested at least one real option-surface cycle for this
    universe's primary underlying (feature-onboarding pass) -- gates whether
    ``CompositeMarketDataSource``/``AlphaDataOptionChainSource`` can supply real chains.
    Coverage may still be partial across the full label history (options data starts
    later than equity/futures OHLCV); rows outside covered cycles get a zero-filled,
    all-masked surface rather than being excluded from training (see
    ``FeatureBuilder.build_surface_history``)."""
    paths = AlphaDataPaths()
    return any(paths.options.glob(f"{universe[0]}_SURFACE_CE_*_{base_interval}_greeks.parquet"))


_BARRIER_IDX = {Barrier.STOP: 0, Barrier.TARGET: 1, Barrier.TIMEOUT: 2}
_MIN_LABEL_SCHEMA_VERSION = 5
_REGIME_LOOKBACK = 12
_LOG_RETURN_IDX = CANDLE_FEATURE_NAMES.index("log_return")
_DEFAULT_WORLD_MODEL_SEQ_BATCHES = 16
_SUPPORTED_BARRIER_MODES = {"legacy", "derived", "decomposed"}
_SUPPORTED_RETURN_TARGET_MODES = {"exit", "horizon", "timeout"}


@dataclass(frozen=True)
class _ModelSelectionFold:
    fold_id: int
    train_rows: int
    val_rows: int
    best_epoch: int
    best_val_loss: float | None


@dataclass
class _TrainingOutcome:
    model: HRWForecaster | HRWWorldModel
    trainer: HRWTrainer
    pretrainer: MarketStatePretrainer | None = None
    rssm_trainer: WorldModelTrainer | None = None
    head_trainer: HeadTrainer | None = None


@dataclass
class _PredictionCalibrationInputs:
    quantile_levels: tuple[float, ...] = DEFAULT_QUANTILES
    horizon_payloads: dict[int, dict[str, list[float] | list[list[float]]]] = field(
        default_factory=dict
    )
    barrier_probs: list[list[float]] = field(default_factory=list)
    barrier_labels: list[int] = field(default_factory=list)
    regime_probs: list[list[float]] = field(default_factory=list)
    regime_labels: list[int] = field(default_factory=list)

    def append_horizon(
        self,
        *,
        horizon: int,
        pred_quantiles: np.ndarray,
        realized: np.ndarray,
        predicted_volatility: np.ndarray,
        realized_volatility: np.ndarray,
    ) -> None:
        bundle = self.horizon_payloads.setdefault(
            int(horizon),
            {
                "pred_quantiles": [],
                "realized": [],
                "predicted_volatility": [],
                "realized_volatility": [],
            },
        )
        cast_pred_quantiles = bundle["pred_quantiles"]
        cast_realized = bundle["realized"]
        cast_pred_vol = bundle["predicted_volatility"]
        cast_real_vol = bundle["realized_volatility"]
        assert isinstance(cast_pred_quantiles, list)
        assert isinstance(cast_realized, list)
        assert isinstance(cast_pred_vol, list)
        assert isinstance(cast_real_vol, list)
        cast_pred_quantiles.extend(pred_quantiles.tolist())
        cast_realized.extend(realized.tolist())
        cast_pred_vol.extend(predicted_volatility.tolist())
        cast_real_vol.extend(realized_volatility.tolist())

    def extend(self, other: "_PredictionCalibrationInputs") -> None:
        for horizon, bundle in other.horizon_payloads.items():
            own = self.horizon_payloads.setdefault(
                int(horizon),
                {
                    "pred_quantiles": [],
                    "realized": [],
                    "predicted_volatility": [],
                    "realized_volatility": [],
                },
            )
            for key in ("pred_quantiles", "realized", "predicted_volatility", "realized_volatility"):
                cast_own = own[key]
                cast_other = bundle[key]
                assert isinstance(cast_own, list)
                assert isinstance(cast_other, list)
                cast_own.extend(cast_other)
        self.barrier_probs.extend(other.barrier_probs)
        self.barrier_labels.extend(other.barrier_labels)
        self.regime_probs.extend(other.regime_probs)
        self.regime_labels.extend(other.regime_labels)


@dataclass(frozen=True)
class _ReturnTargets:
    forward_return: np.ndarray
    realized_vol: np.ndarray
    mae: np.ndarray
    mfe: np.ndarray
    return_weight: np.ndarray | None = None


def _demo_candles(universe: tuple[str, ...], n: int = 220) -> dict[str, list[MarketCandle]]:
    rng = np.random.default_rng(7)
    start = datetime(2026, 6, 25, 9, 20)
    out: dict[str, list[MarketCandle]] = {}
    for s, base in zip(universe, 100 + rng.uniform(0, 50, len(universe)), strict=False):
        rows, price = [], float(base)
        for i in range(n):
            ts = start + timedelta(minutes=5 * i)
            drift = 0.0025 * float(np.sin(i / 10.0)) + float(rng.normal(0, 0.002))
            price = max(1.0, price * (1.0 + drift))
            rows.append(
                MarketCandle(
                    symbol=s,
                    ts=ts,
                    available_at=ts,
                    open=price,
                    high=price * 1.004,
                    low=price * 0.996,
                    close=price,
                    volume=1000 + i,
                    oi=5000 + i * 5,
                )
            )
        out[s] = rows
    return out


def _direction_idx(values: np.ndarray) -> np.ndarray:
    return np.digitize(values, [-0.001, 0.001]).astype(np.int64)


def _barrier_idx(value: object, forward_return: float) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 1 if forward_return > 0.003 else 0 if forward_return < -0.003 else 2
    if isinstance(value, Barrier):
        if value is Barrier.AMBIGUOUS:
            return 2
        return _BARRIER_IDX[value]
    if isinstance(value, str):
        barrier = Barrier(value)
        if barrier is Barrier.AMBIGUOUS:
            return 2
        return _BARRIER_IDX[barrier]
    return int(value)


def _regime_idx(value: object) -> int:
    if isinstance(value, Regime):
        return REGIME_CLASSES.index(value)
    if isinstance(value, str):
        return REGIME_CLASSES.index(Regime(value))
    if isinstance(value, (int, np.integer)):
        return int(value)
    raise ValueError(f"unsupported regime label: {value!r}")


def _state_regime_idx(
    market_features: np.ndarray,
    *,
    event_type: EventType = EventType.NONE,
) -> int:
    recent = market_features[0, -_REGIME_LOOKBACK:, _LOG_RETURN_IDX]
    trailing_return = float(np.nansum(recent))
    trailing_vol = float(max(np.nanstd(recent), 1e-6))
    return REGIME_CLASSES.index(
        state_regime_label(
            trailing_return,
            trailing_vol,
            event=event_type is not EventType.NONE,
        )
    )


def _sample_weight(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 1.0
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "sample_weight must be numeric; rerun scripts/label.py to regenerate labels"
        ) from exc
    if out < 0:
        raise ValueError("sample_weight must be non-negative")
    return out


def _resolve_barrier_spec(labels: pd.DataFrame | None) -> BarrierSpec:
    if labels is None or labels.empty:
        return BarrierSpec()

    def _constant(name: str, default: float) -> float:
        if name not in labels.columns:
            return float(default)
        values = pd.Series(labels[name]).dropna().unique().tolist()
        if not values:
            return float(default)
        if len(values) > 1:
            raise ValueError(f"labels contain multiple values for {name}: {values[:5]}")
        return float(values[0])

    def _inferred_mult(column: str, default: float) -> float:
        if column not in labels.columns or BARRIER_SIGMA_COLUMN not in labels.columns:
            return float(default)
        sigma = labels[BARRIER_SIGMA_COLUMN].to_numpy(dtype=float)
        values = np.abs(labels[column].to_numpy(dtype=float))
        valid = np.isfinite(sigma) & np.isfinite(values) & (sigma > 0.0)
        if not bool(valid.any()):
            return float(default)
        return float(np.nanmedian(values[valid] / sigma[valid]))

    stop_mult = _constant(
        BARRIER_STOP_MULT_COLUMN,
        _inferred_mult(BARRIER_STOP_RETURN_COLUMN, 2.0),
    )
    target_mult = _constant(
        BARRIER_TARGET_MULT_COLUMN,
        _inferred_mult(BARRIER_TARGET_RETURN_COLUMN, 2.0),
    )
    vol_span = int(round(_constant(BARRIER_VOL_SPAN_COLUMN, 50.0)))
    cost_floor_frac = _constant(BARRIER_COST_FLOOR_COLUMN, 0.0)
    # horizon_bars (feature/label overhaul Phase 4a): must match the H the labels were
    # actually built with, so the reconstructed BarrierSpec's sqrt(horizon_bars) barrier
    # scaling exactly matches labeling-time behavior -- read from the labels' own
    # "horizon_bars" column (always persisted; see LabelRecord.horizon_bars) rather than
    # re-deriving it, the same "reconstruct exact config, don't guess" pattern the other
    # _constant()/_inferred_mult() calls in this function already follow.
    horizon_bars = int(round(_constant("horizon_bars", 1.0)))
    return BarrierSpec(
        stop_mult=stop_mult,
        target_mult=target_mult,
        vol_span=vol_span,
        cost_floor_frac=cost_floor_frac,
        horizon_bars=horizon_bars,
    )


def _barrier_class_weights_from_labels(
    labels: pd.DataFrame, train_end: pd.Timestamp | None
) -> tuple[float, float, float] | None:
    """Inverse-frequency barrier class weights [stop, target, timeout] from the TRAIN split only.

    Weighted by each class's own frequency, these average to 1.0, so the overall barrier-loss
    scale is unchanged (unweighted CE otherwise collapses to always predicting "timeout" — the
    ~80% majority class — with zero recall on stop/target; confirmed on both a smoke run and a
    full 5-fold retrain on 2026-07-06). Returns None if any class is entirely absent from the
    training split (weighting an unseen class is meaningless).
    """
    if "barrier" not in labels.columns:
        return None
    train_labels = labels if train_end is None else labels.loc[labels.index <= train_end]
    counts = train_labels["barrier"].value_counts()
    order = ("stop", "target", "timeout")
    class_counts = [int(counts.get(name, 0)) for name in order]
    if any(c == 0 for c in class_counts):
        return None
    total = sum(class_counts)
    n_classes = len(order)
    return tuple(total / (n_classes * c) for c in class_counts)


def _effective_sample_weights(
    labels: pd.DataFrame,
    *,
    management_horizon: int,
    execution_cfg,
) -> tuple[np.ndarray, OpportunityWeightAudit]:
    base = np.array(
        [
            _sample_weight(value)
            for value in labels.get("sample_weight", pd.Series(1.0, index=labels.index))
        ],
        dtype=np.float32,
    )
    opportunity, audit = compute_management_opportunity_weights(
        labels,
        management_horizon=management_horizon,
        execution_cfg=execution_cfg,
    )
    return np.ascontiguousarray(base * opportunity, dtype=np.float32), audit


def _window_view_2d(values: np.ndarray, lookback: int) -> np.ndarray:
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    if values.shape[0] < lookback:
        return np.empty((0, lookback, values.shape[1]), dtype=values.dtype)
    view = np.lib.stride_tricks.sliding_window_view(values, window_shape=lookback, axis=0)
    return np.moveaxis(view, -1, 1)  # [T-L+1, L, F]


def _return_target_mode_from_cfg(cfg: dict[str, object]) -> str:
    model_cfg = cfg.get("model", {})
    raw = model_cfg.get("return_target_mode", "horizon") if isinstance(model_cfg, dict) else "horizon"
    mode = str(raw).strip().lower()
    if mode not in _SUPPORTED_RETURN_TARGET_MODES:
        raise ValueError(
            "unsupported model.return_target_mode "
            f"{raw!r}; expected one of {sorted(_SUPPORTED_RETURN_TARGET_MODES)}"
        )
    return mode


def _resolve_return_targets(
    labels: pd.DataFrame,
    *,
    management_horizon: int,
    return_target_mode: str,
) -> _ReturnTargets:
    if return_target_mode not in _SUPPORTED_RETURN_TARGET_MODES:
        raise ValueError(f"unsupported return target mode: {return_target_mode!r}")

    management_return_col = horizon_return_column(management_horizon)
    management_vol_col = horizon_volatility_column(management_horizon)
    management_mae_col = horizon_mae_column(management_horizon)
    management_mfe_col = horizon_mfe_column(management_horizon)

    exit_returns_arr = labels["exit_return"].to_numpy(dtype=np.float32)
    exit_vol_arr = np.maximum(
        labels.get(
            "realized_vol",
            pd.Series(np.abs(exit_returns_arr), index=labels.index),
        ).to_numpy(dtype=np.float32),
        1e-6,
    )
    exit_mae_arr = (
        np.maximum(labels["mae"].to_numpy(dtype=np.float32), 0.0)
        if "mae" in labels.columns
        else np.zeros(len(labels), dtype=np.float32)
    )
    exit_mfe_arr = (
        np.maximum(labels["mfe"].to_numpy(dtype=np.float32), 0.0)
        if "mfe" in labels.columns
        else np.zeros(len(labels), dtype=np.float32)
    )

    if (
        management_return_col in labels.columns
        and management_vol_col in labels.columns
        and management_mae_col in labels.columns
        and management_mfe_col in labels.columns
    ):
        horizon_returns_arr = labels[management_return_col].to_numpy(dtype=np.float32)
        horizon_vol_arr = np.maximum(labels[management_vol_col].to_numpy(dtype=np.float32), 1e-6)
        horizon_mae_arr = np.maximum(labels[management_mae_col].to_numpy(dtype=np.float32), 0.0)
        horizon_mfe_arr = np.maximum(labels[management_mfe_col].to_numpy(dtype=np.float32), 0.0)
    else:
        log.warning(
            "train.management_horizon_targets_missing horizon_bars=%s fallback=%s",
            management_horizon,
            "exit_consequence_labels",
        )
        horizon_returns_arr = exit_returns_arr
        horizon_vol_arr = exit_vol_arr
        horizon_mae_arr = exit_mae_arr
        horizon_mfe_arr = exit_mfe_arr

    if return_target_mode == "exit":
        return _ReturnTargets(
            forward_return=exit_returns_arr,
            realized_vol=exit_vol_arr,
            mae=exit_mae_arr,
            mfe=exit_mfe_arr,
            return_weight=None,
        )
    if return_target_mode == "horizon":
        return _ReturnTargets(
            forward_return=horizon_returns_arr,
            realized_vol=horizon_vol_arr,
            mae=horizon_mae_arr,
            mfe=horizon_mfe_arr,
            return_weight=None,
        )

    barrier_values = labels["barrier"] if "barrier" in labels.columns else pd.Series([None] * len(labels), index=labels.index)
    barrier_valid_values = (
        labels["barrier_valid"].astype(bool)
        if "barrier_valid" in labels.columns
        else barrier_values.astype(str).ne(Barrier.AMBIGUOUS.value)
    )
    timeout_values = barrier_values.astype(str).eq(Barrier.TIMEOUT.value)
    return_weight_arr = (
        barrier_valid_values.to_numpy(dtype=np.float32, copy=False)
        * timeout_values.to_numpy(dtype=np.float32, copy=False)
    )
    return _ReturnTargets(
        forward_return=horizon_returns_arr,
        realized_vol=horizon_vol_arr,
        mae=horizon_mae_arr,
        mfe=horizon_mfe_arr,
        return_weight=return_weight_arr,
    )


def _load_labels_frame(labels_path: str | Path) -> pd.DataFrame:
    labels = pd.read_parquet(labels_path)
    if "ts" in labels.columns:
        labels = labels.set_index("ts")
    labels.index = pd.to_datetime(labels.index)
    if (
        "label_schema_version" not in labels.columns
        or int(labels["label_schema_version"].max()) < _MIN_LABEL_SCHEMA_VERSION
    ):
        raise ValueError(
            "labels.parquet predates the point-in-time regime/weight schema; rerun scripts/label.py"
        )
    return labels.sort_index()


def _timestamp_split_mask(
    index: pd.DatetimeIndex,
    split: str,
    manifest: ChronoSplitManifest | None,
) -> np.ndarray:
    if manifest is None or split == "all":
        return np.ones(len(index), dtype=bool)
    return manifest.mask(pd.DatetimeIndex(index), split).to_numpy(dtype=bool, copy=False)


def _label_split(
    labels: pd.DataFrame,
    split: str,
    manifest: ChronoSplitManifest | None,
) -> pd.DataFrame:
    if manifest is None or split == "all":
        return labels
    return manifest.filter_labels(labels, split)


def _chunk_latent_pairs(
    *,
    context: np.ndarray,
    future: np.ndarray,
    context_futures: np.ndarray | None = None,
    future_futures: np.ndarray | None = None,
    context_regime: np.ndarray | None = None,
    future_regime: np.ndarray | None = None,
    batch_size: int,
) -> list[LatentPair]:
    pairs: list[LatentPair] = []
    for start in range(0, len(context), batch_size):
        end = start + batch_size
        pairs.append(
            LatentPair(
                context=torch.tensor(context[start:end], dtype=torch.float32),
                future=torch.tensor(future[start:end], dtype=torch.float32),
                context_futures=(
                    torch.tensor(context_futures[start:end], dtype=torch.float32)
                    if context_futures is not None
                    else None
                ),
                future_futures=(
                    torch.tensor(future_futures[start:end], dtype=torch.float32)
                    if future_futures is not None
                    else None
                ),
                context_regime=(
                    torch.tensor(context_regime[start:end], dtype=torch.float32)
                    if context_regime is not None
                    else None
                ),
                future_regime=(
                    torch.tensor(future_regime[start:end], dtype=torch.float32)
                    if future_regime is not None
                    else None
                ),
            )
        )
    return pairs


def _build_pretrain_pairs_from_history(
    history,
    lookback_bars: int,
    *,
    batch_size: int,
    gap_bars: int,
    split: str = "train",
    split_manifest: ChronoSplitManifest | None = None,
    futures_index: pd.DatetimeIndex | None = None,
    futures_history: np.ndarray | None = None,
    regime_context_rows: np.ndarray | None = None,
    start_ts: pd.Timestamp | None = None,
    end_ts: pd.Timestamp | None = None,
) -> list[LatentPair]:
    if gap_bars > lookback_bars // 2:
        log.warning(
            "train.pretrain_gap_bars_large gap_bars=%s lookback_bars=%s note=%s",
            gap_bars,
            lookback_bars,
            "context/future windows barely overlap; Stage-2 pretraining task may be "
            "nearly disjoint rather than genuinely predictive (review finding H6)",
        )
    windows = history.window_view(lookback_bars)
    usable = len(windows) - gap_bars
    if usable <= 0:
        raise ValueError("insufficient aligned history for Stage-2 pretraining")
    context = np.ascontiguousarray(windows[:usable], dtype=np.float32)
    future = np.ascontiguousarray(windows[gap_bars: gap_bars + usable], dtype=np.float32)
    context_ts = history.index[lookback_bars - 1 : lookback_bars - 1 + usable]
    future_ts = history.index[lookback_bars - 1 + gap_bars : lookback_bars - 1 + gap_bars + usable]
    mask = _timestamp_split_mask(context_ts, split, split_manifest) & _timestamp_split_mask(
        future_ts, split, split_manifest
    )
    if start_ts is not None:
        mask &= pd.DatetimeIndex(context_ts) >= pd.Timestamp(start_ts)
        mask &= pd.DatetimeIndex(future_ts) >= pd.Timestamp(start_ts)
    if end_ts is not None:
        mask &= pd.DatetimeIndex(context_ts) <= pd.Timestamp(end_ts)
        mask &= pd.DatetimeIndex(future_ts) <= pd.Timestamp(end_ts)

    context_futures = None
    future_futures = None
    if futures_index is not None and futures_history is not None:
        futures_windows = _window_view_2d(
            futures_history.astype(np.float32, copy=False),
            lookback_bars,
        )
        context_pos = futures_index.get_indexer(pd.DatetimeIndex(context_ts))
        future_pos = futures_index.get_indexer(pd.DatetimeIndex(future_ts))
        mask &= (context_pos >= lookback_bars - 1) & (future_pos >= lookback_bars - 1)
        context_futures = np.ascontiguousarray(
            futures_windows[context_pos - lookback_bars + 1],
            dtype=np.float32,
        )
        future_futures = np.ascontiguousarray(
            futures_windows[future_pos - lookback_bars + 1],
            dtype=np.float32,
        )

    context_regime = None
    future_regime = None
    if regime_context_rows is not None:
        context_regime = np.ascontiguousarray(
            regime_context_rows[lookback_bars - 1 : lookback_bars - 1 + usable],
            dtype=np.float32,
        )
        future_regime = np.ascontiguousarray(
            regime_context_rows[
                lookback_bars - 1 + gap_bars : lookback_bars - 1 + gap_bars + usable
            ],
            dtype=np.float32,
        )
    if not bool(mask.any()):
        raise ValueError("no Stage-2 pretraining pairs available after time split")
    return _chunk_latent_pairs(
        context=context[mask],
        future=future[mask],
        context_futures=context_futures[mask] if context_futures is not None else None,
        future_futures=future_futures[mask] if future_futures is not None else None,
        context_regime=context_regime[mask] if context_regime is not None else None,
        future_regime=future_regime[mask] if future_regime is not None else None,
        batch_size=batch_size,
    )


def build_demo_pretrain_pairs(
    universe: tuple[str, ...],
    lookback_bars: int,
    *,
    batch_size: int,
    gap_bars: int = 1,
) -> list[LatentPair]:
    candles = _demo_candles(universe, n=max(220, lookback_bars + gap_bars + 80))
    source = InMemoryMarketDataSource(candles=candles)
    dc = data_config_from_cfg({"data": {"universe": universe, "lookback_bars": lookback_bars}})
    history = FeatureBuilder(dc, source).build_history()
    return _build_pretrain_pairs_from_history(
        history,
        lookback_bars,
        batch_size=batch_size,
        gap_bars=gap_bars,
        split="all",
    )


def build_market_pretrain_pairs(
    data_dir: str,
    universe: tuple[str, ...],
    base_interval: str,
    lookback_bars: int,
    *,
    batch_size: int,
    gap_bars: int = 1,
    split: str = "train",
    split_manifest: ChronoSplitManifest | None = None,
    start_ts: pd.Timestamp | None = None,
    end_ts: pd.Timestamp | None = None,
) -> list[LatentPair]:
    dc = data_config_from_cfg(
        {
            "data": {
                "universe": universe,
                "base_interval": base_interval,
                "lookback_bars": lookback_bars,
            }
        }
    )
    source = ParquetMarketDataSource(data_dir=data_dir, universe=universe, base_interval=base_interval)
    vix_path = Path(data_dir) / "ohlcv" / f"INDIAVIX_{base_interval}.parquet"
    ctx_path = Path(data_dir) / "regime" / "daily_context.parquet"
    inputs = ModelInputBuilder.from_data_dir(
        dc,
        source,
        data_dir=data_dir,
        contract=ModelInputContract.from_data_config(
            dc,
            feature_names=CANDLE_FEATURE_NAMES,
            uses_futures=_futures_data_available(universe, base_interval),
            uses_regime_context=True,
            require_vix=vix_path.exists(),
            require_daily_context=False,
        ),
    )
    history = inputs.feature_builder.build_history()
    futures_index = None
    futures_history = None
    if inputs.feature_builder.futures_builder is not None:
        futures_index, futures_history = inputs.feature_builder.futures_builder.build_history()
    regime_context_rows = None
    if inputs.regime_builder is not None:
        regime_context_rows = np.stack(
            [
                featurize_regime(*inputs.regime_builder.build(ts.to_pydatetime()))
                for ts in history.index
            ]
        ).astype(np.float32, copy=False)
    return _build_pretrain_pairs_from_history(
        history,
        lookback_bars,
        batch_size=batch_size,
        gap_bars=gap_bars,
        split=split,
        split_manifest=split_manifest,
        futures_index=futures_index,
        futures_history=futures_history,
        regime_context_rows=regime_context_rows,
        start_ts=start_ts,
        end_ts=end_ts,
    )


def _chunk_batch(
    *,
    features: np.ndarray,
    forward_return: np.ndarray,
    direction: np.ndarray,
    regime: np.ndarray | None = None,
    realized_vol: np.ndarray | None = None,
    vol_baseline: np.ndarray | None = None,
    mae: np.ndarray | None = None,
    mfe: np.ndarray | None = None,
    return_weight: np.ndarray | None = None,      # [N]
    barrier: np.ndarray | None = None,
    barrier_weight: np.ndarray | None = None,     # [N]
    regime_context: np.ndarray | None = None,   # [N, K]
    futures: np.ndarray | None = None,           # [N, L, 12]
    surface_grid: np.ndarray | None = None,      # [N, S, C]
    surface_mask: np.ndarray | None = None,      # [N, S]
    surface_context: np.ndarray | None = None,   # [N, K_surface]
    sample_weight: np.ndarray | None = None,     # [N]
    horizon_returns: np.ndarray | None = None,   # [N, H]
    horizon_volatility: np.ndarray | None = None,   # [N, H]
    horizon_mae: np.ndarray | None = None,       # [N, H]
    horizon_mfe: np.ndarray | None = None,       # [N, H]
    barrier_context: np.ndarray | None = None,   # [N, 3] sigma + explicit stop/target returns
    primary_side: np.ndarray | None = None,      # [N] float in {-1,0,1}
    meta_label: np.ndarray | None = None,        # [N] float in {0,1}, NaN where primary_side==0
    target_horizons: tuple[int, ...] = (),
    batch_size: int,
) -> list[ForecastBatch]:
    batches: list[ForecastBatch] = []
    for start in range(0, len(features), batch_size):
        end = start + batch_size
        batches.append(
            ForecastBatch(
                features=torch.tensor(features[start:end], dtype=torch.float32),
                forward_return=torch.tensor(forward_return[start:end], dtype=torch.float32),
                direction=torch.tensor(direction[start:end], dtype=torch.long),
                regime=(
                    torch.tensor(regime[start:end], dtype=torch.long) if regime is not None else None
                ),
                realized_vol=(
                    torch.tensor(realized_vol[start:end], dtype=torch.float32)
                    if realized_vol is not None
                    else None
                ),
                vol_baseline=(
                    torch.tensor(vol_baseline[start:end], dtype=torch.float32)
                    if vol_baseline is not None
                    else None
                ),
                mae=(
                    torch.tensor(mae[start:end], dtype=torch.float32)
                    if mae is not None
                    else None
                ),
                mfe=(
                    torch.tensor(mfe[start:end], dtype=torch.float32)
                    if mfe is not None
                    else None
                ),
                return_weight=(
                    torch.tensor(return_weight[start:end], dtype=torch.float32)
                    if return_weight is not None
                    else None
                ),
                barrier=(
                    torch.tensor(barrier[start:end], dtype=torch.long) if barrier is not None else None
                ),
                barrier_weight=(
                    torch.tensor(barrier_weight[start:end], dtype=torch.float32)
                    if barrier_weight is not None
                    else None
                ),
                regime_context=(
                    torch.tensor(regime_context[start:end], dtype=torch.float32)
                    if regime_context is not None
                    else None
                ),
                futures=(
                    torch.tensor(futures[start:end], dtype=torch.float32)
                    if futures is not None
                    else None
                ),
                surface_grid=(
                    torch.tensor(surface_grid[start:end], dtype=torch.float32)
                    if surface_grid is not None
                    else None
                ),
                surface_mask=(
                    torch.tensor(surface_mask[start:end], dtype=torch.float32)
                    if surface_mask is not None
                    else None
                ),
                surface_context=(
                    torch.tensor(surface_context[start:end], dtype=torch.float32)
                    if surface_context is not None
                    else None
                ),
                sample_weight=(
                    torch.tensor(sample_weight[start:end], dtype=torch.float32)
                    if sample_weight is not None
                    else None
                ),
                horizon_returns=(
                    torch.tensor(horizon_returns[start:end], dtype=torch.float32)
                    if horizon_returns is not None
                    else None
                ),
                horizon_volatility=(
                    torch.tensor(horizon_volatility[start:end], dtype=torch.float32)
                    if horizon_volatility is not None
                    else None
                ),
                horizon_mae=(
                    torch.tensor(horizon_mae[start:end], dtype=torch.float32)
                    if horizon_mae is not None
                    else None
                ),
                horizon_mfe=(
                    torch.tensor(horizon_mfe[start:end], dtype=torch.float32)
                    if horizon_mfe is not None
                    else None
                ),
                barrier_context=(
                    torch.tensor(barrier_context[start:end], dtype=torch.float32)
                    if barrier_context is not None
                    else None
                ),
                primary_side=(
                    torch.tensor(primary_side[start:end], dtype=torch.float32)
                    if primary_side is not None
                    else None
                ),
                meta_label=(
                    torch.tensor(meta_label[start:end], dtype=torch.float32)
                    if meta_label is not None
                    else None
                ),
                target_horizons=target_horizons,
            )
        )
    return batches


def _fixed_horizon_targets(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    decision_pos: int,
    horizon: int,
) -> tuple[float, float, float, float]:
    entry_pos = decision_pos + 1
    exit_pos = decision_pos + horizon
    if entry_pos >= len(close) or exit_pos >= len(close):
        raise ValueError("insufficient future bars for the requested horizon target")
    path = close[entry_pos : exit_pos + 1]
    high_path = high[entry_pos : exit_pos + 1]
    low_path = low[entry_pos : exit_pos + 1]
    realized_return = float(close[exit_pos] / open_[entry_pos] - 1.0)
    if len(path) > 1:
        realized_vol = float(np.std(np.diff(np.log(path))))
    else:
        realized_vol = 0.0
    entry_px = float(open_[entry_pos])
    mae = float(max((entry_px - low_path.min()) / entry_px, 0.0)) if len(low_path) else 0.0
    mfe = float(max((high_path.max() - entry_px) / entry_px, 0.0)) if len(high_path) else 0.0
    return realized_return, max(realized_vol, 1e-6), mae, mfe


def build_demo_batches(
    universe: tuple[str, ...],
    lookback_bars: int,
    horizon: int,
    *,
    batch_size: int,
    target_horizons: tuple[int, ...] | None = None,
) -> list[ForecastBatch]:
    # World-model sequence training needs seq_len=max(target_horizons) decision rows
    # AFTER the lookback+horizon warmup (see _build_world_model_sequences), not just
    # one -- account for that here so --demo dry-runs don't fail with "insufficient
    # chronological rows" once target_horizons includes a horizon beyond `horizon`
    # itself (feature/label overhaul Phase 4a: target_horizons now goes up to 192).
    max_target_horizon = max(target_horizons) if target_horizons else 0
    candles = _demo_candles(
        universe, n=max(220, lookback_bars + horizon + max_target_horizon + 80)
    )
    source = InMemoryMarketDataSource(candles=candles)
    dc = data_config_from_cfg({"data": {"universe": universe, "lookback_bars": lookback_bars}})
    inputs = ModelInputBuilder.from_data_dir(
        dc,
        source,
        data_dir=None,
        contract=ModelInputContract.from_data_config(
            dc,
            feature_names=CANDLE_FEATURE_NAMES,
            uses_regime_context=True,
        ),
    )
    primary = candles[universe[0]]
    opens = np.array([row.open for row in primary], dtype=float)
    highs = np.array([row.high for row in primary], dtype=float)
    lows = np.array([row.low for row in primary], dtype=float)
    closes = np.array([row.close for row in primary], dtype=float)
    barrier_rows = barrier_context_series(closes, spec=BarrierSpec())
    sorted_horizons = tuple(sorted(set(target_horizons or ())))

    feats, returns, directions, regimes, realized_vols, maes, mfes, barriers, weights = [], [], [], [], [], [], [], [], []
    regime_contexts: list[np.ndarray] = []
    barrier_contexts: list[np.ndarray] = []
    horizon_returns_rows: list[list[float]] = []
    horizon_vol_rows: list[list[float]] = []
    horizon_mae_rows: list[list[float]] = []
    horizon_mfe_rows: list[list[float]] = []
    for i in range(lookback_bars - 1, len(primary) - horizon):
        ts = primary[i].ts
        snapshot = inputs.build(ts)
        feats.append(snapshot.market.candle_features)
        fwd, sigma, mae, mfe = _fixed_horizon_targets(
            opens,
            highs,
            lows,
            closes,
            decision_pos=i,
            horizon=horizon,
        )
        returns.append(fwd)
        directions.append(_direction_idx(np.array([fwd]))[0])
        realized_vols.append(sigma)
        maes.append(mae)
        mfes.append(mfe)
        barriers.append(_barrier_idx(None, fwd))
        barrier_contexts.append(np.asarray(barrier_rows[i], dtype=np.float32))
        event_type = snapshot.regime[1].event_type if snapshot.regime is not None else EventType.NONE
        regimes.append(_state_regime_idx(snapshot.market.candle_features, event_type=event_type))
        regime_contexts.append(
            featurize_regime(*snapshot.regime)
            if snapshot.regime is not None
            else np.zeros(len(REGIME_CONTEXT_FEATURES), dtype=np.float32)
        )
        weights.append(1.0)
        if sorted_horizons:
            returns_row: list[float] = []
            vol_row: list[float] = []
            mae_row: list[float] = []
            mfe_row: list[float] = []
            for target_h in sorted_horizons:
                h_ret, h_vol, h_mae, h_mfe = _fixed_horizon_targets(
                    opens,
                    highs,
                    lows,
                    closes,
                    decision_pos=i,
                    horizon=target_h,
                )
                returns_row.append(h_ret)
                vol_row.append(h_vol)
                mae_row.append(h_mae)
                mfe_row.append(h_mfe)
            horizon_returns_rows.append(returns_row)
            horizon_vol_rows.append(vol_row)
            horizon_mae_rows.append(mae_row)
            horizon_mfe_rows.append(mfe_row)

    return _chunk_batch(
        features=np.stack(feats),
        forward_return=np.array(returns, dtype=np.float32),
        direction=np.array(directions, dtype=np.int64),
        regime=np.array(regimes, dtype=np.int64),
        realized_vol=np.array(realized_vols, dtype=np.float32),
        mae=np.array(maes, dtype=np.float32),
        mfe=np.array(mfes, dtype=np.float32),
        barrier=np.array(barriers, dtype=np.int64),
        regime_context=np.stack(regime_contexts),
        sample_weight=np.array(weights, dtype=np.float32),
        horizon_returns=(
            np.asarray(horizon_returns_rows, dtype=np.float32) if horizon_returns_rows else None
        ),
        horizon_volatility=(
            np.asarray(horizon_vol_rows, dtype=np.float32) if horizon_vol_rows else None
        ),
        horizon_mae=(
            np.asarray(horizon_mae_rows, dtype=np.float32) if horizon_mae_rows else None
        ),
        horizon_mfe=(
            np.asarray(horizon_mfe_rows, dtype=np.float32) if horizon_mfe_rows else None
        ),
        barrier_context=np.asarray(barrier_contexts, dtype=np.float32),
        target_horizons=sorted_horizons,
        batch_size=batch_size,
    )


def build_labeled_batches(
    data_dir: str,
    universe: tuple[str, ...],
    base_interval: str,
    lookback_bars: int,
    labels: pd.DataFrame,
    *,
    management_horizon: int,
    execution_cfg,
    batch_size: int,
    split: str = "train",
    split_manifest: ChronoSplitManifest | None = None,
    target_horizons: tuple[int, ...] | None = None,
    barrier_spec: BarrierSpec | None = None,
    return_target_mode: str = "horizon",
) -> list[ForecastBatch]:
    """Build ForecastBatch list from labeled parquet.

    split boundaries are chronological and derived from the label history via
    ``ChronoSplitManifest`` rather than hard-coded calendar cutoffs.
    """
    dc = data_config_from_cfg(
        {
            "data": {
                "universe": universe,
                "base_interval": base_interval,
                "lookback_bars": lookback_bars,
            }
        }
    )
    source = ParquetMarketDataSource(data_dir=data_dir, universe=universe, base_interval=base_interval)
    labels = _label_split(labels, split, split_manifest).sort_index()
    split_total = len(labels)
    resolved_barrier_spec = barrier_spec or _resolve_barrier_spec(labels)

    vix_path = Path(data_dir) / "ohlcv" / f"INDIAVIX_{base_interval}.parquet"
    ctx_path = Path(data_dir) / "regime" / "daily_context.parquet"
    uses_option_surface = _option_surface_data_available(universe, base_interval)
    inputs = ModelInputBuilder.from_data_dir(
        dc,
        source,
        data_dir=data_dir,
        contract=ModelInputContract.from_data_config(
            dc,
            feature_names=CANDLE_FEATURE_NAMES,
            uses_futures=_futures_data_available(universe, base_interval),
            uses_regime_context=True,
            uses_option_surface=uses_option_surface,
            require_vix=vix_path.exists(),
            require_daily_context=False,
            barrier_stop_mult=resolved_barrier_spec.stop_mult,
            barrier_target_mult=resolved_barrier_spec.target_mult,
            barrier_vol_span=resolved_barrier_spec.vol_span,
            barrier_horizon_bars=resolved_barrier_spec.horizon_bars,
            barrier_cost_floor_frac=resolved_barrier_spec.cost_floor_frac,
        ),
    )
    history = inputs.feature_builder.build_history()
    market_windows = history.window_view(lookback_bars)
    positions = history.index.get_indexer(pd.DatetimeIndex(labels.index))
    valid_mask = positions >= lookback_bars - 1

    futures_windows = None
    futures_positions = None
    futures_quality_reasons: dict[str, int] = {}
    futures_quality_rejected_rows = 0
    futures_builder = inputs.feature_builder.futures_builder
    if futures_builder is not None:
        futures_index, futures_history = futures_builder.build_history()
        futures_positions = futures_index.get_indexer(pd.DatetimeIndex(labels.index))
        futures_eligible = futures_builder.eligible_positions(lookback_bars)
        for pos in futures_positions:
            if pos < 0:
                futures_quality_reasons["missing_futures_timestamp"] = (
                    futures_quality_reasons.get("missing_futures_timestamp", 0) + 1
                )
                continue
            quality = futures_builder.quality_for_window(
                futures_index[int(pos)].to_pydatetime(),
                lookback_bars,
            )
            if quality.eligible:
                continue
            futures_quality_rejected_rows += 1
            for reason in quality.reasons:
                futures_quality_reasons[reason] = futures_quality_reasons.get(reason, 0) + 1
        valid_mask &= futures_positions >= lookback_bars - 1
        valid_mask &= np.array(
            [
                bool(futures_eligible[pos]) if pos >= 0 else False
                for pos in futures_positions
            ],
            dtype=bool,
        )
        futures_windows = _window_view_2d(futures_history.astype(np.float32, copy=False), lookback_bars)

    labels = labels.iloc[valid_mask]
    positions = positions[valid_mask]
    if futures_positions is not None:
        futures_positions = futures_positions[valid_mask]

    if labels.empty:
        raise ValueError("no trainable labeled rows after feature alignment")

    feats = np.ascontiguousarray(
        market_windows[positions - lookback_bars + 1],
        dtype=np.float32,
    )
    sorted_horizons = tuple(sorted(set(target_horizons or ())))
    return_targets = _resolve_return_targets(
        labels,
        management_horizon=management_horizon,
        return_target_mode=return_target_mode,
    )
    barrier_values = labels["barrier"] if "barrier" in labels.columns else pd.Series([None] * len(labels), index=labels.index)
    barrier_valid_values = (
        labels["barrier_valid"].astype(bool)
        if "barrier_valid" in labels.columns
        else barrier_values.astype(str).ne(Barrier.AMBIGUOUS.value)
    )
    barrier_arr = np.array(
        [
            _barrier_idx(barrier, fwd)
            for barrier, fwd in zip(barrier_values, return_targets.forward_return, strict=False)
        ],
        dtype=np.int64,
    )
    barrier_weight_arr = barrier_valid_values.to_numpy(dtype=np.float32, copy=False)
    weight_arr, opportunity_audit = _effective_sample_weights(
        labels,
        management_horizon=management_horizon,
        execution_cfg=execution_cfg,
    )
    barrier_context_arr = None
    if {
        BARRIER_SIGMA_COLUMN,
        BARRIER_STOP_RETURN_COLUMN,
        BARRIER_TARGET_RETURN_COLUMN,
    }.issubset(labels.columns):
        barrier_context_arr = np.ascontiguousarray(
            labels[
                [
                    BARRIER_SIGMA_COLUMN,
                    BARRIER_STOP_RETURN_COLUMN,
                    BARRIER_TARGET_RETURN_COLUMN,
                ]
            ].to_numpy(dtype=np.float32),
            dtype=np.float32,
        )

    regime_context_arr = np.zeros((len(labels), len(REGIME_CONTEXT_FEATURES)), dtype=np.float32)
    regime_values = labels["regime"] if "regime" in labels.columns else pd.Series([np.nan] * len(labels), index=labels.index)
    regimes: list[int] = []
    for idx, (ts, regime_value) in enumerate(zip(labels.index, regime_values, strict=False)):
        regime_input = None
        if inputs.regime_builder is not None:
            regime_input = inputs.regime_builder.build(pd.Timestamp(ts).to_pydatetime())
            regime_context_arr[idx] = featurize_regime(*regime_input)
        if pd.notna(regime_value):
            regimes.append(_regime_idx(regime_value))
        else:
            regimes.append(
                _state_regime_idx(
                    feats[idx],
                    event_type=regime_input[1].event_type if regime_input is not None else EventType.NONE,
                )
            )
    regime_arr = np.array(regimes, dtype=np.int64)

    # Option-surface tensors, one per label row (feature-onboarding pass). Chains are
    # looked up per-timestamp (no vectorized alpha_data chain-history equivalent, unlike
    # candles/futures/regime above) -- see FeatureBuilder.build_surface_history's docstring.
    # Rows outside covered option-surface cycles come back zero-filled/all-masked rather
    # than excluded, so this never shrinks the already-scarce labeled training set.
    surface_grid_arr = surface_mask_arr = surface_context_arr = None
    if uses_option_surface:
        surface_grid_arr, surface_mask_arr, surface_context_arr, surface_eligible = (
            inputs.feature_builder.build_surface_history(
                [pd.Timestamp(ts).to_pydatetime() for ts in labels.index]
            )
        )
        log.info(
            "train.option_surface_coverage",
            split=split,
            eligible_rows=int(surface_eligible.sum()),
            total_rows=len(labels),
        )

    futures_arr = None
    if futures_windows is not None and futures_positions is not None:
        futures_arr = np.ascontiguousarray(
            futures_windows[futures_positions - lookback_bars + 1],
            dtype=np.float32,
        )

    horizon_returns_arr = None
    horizon_vol_arr = None
    horizon_mae_arr = None
    horizon_mfe_arr = None
    if sorted_horizons:
        horizon_returns_arr = np.zeros((len(labels), len(sorted_horizons)), dtype=np.float32)
        horizon_vol_arr = np.zeros((len(labels), len(sorted_horizons)), dtype=np.float32)
        horizon_mae_arr = np.zeros((len(labels), len(sorted_horizons)), dtype=np.float32)
        horizon_mfe_arr = np.zeros((len(labels), len(sorted_horizons)), dtype=np.float32)
        for horizon_idx, target_h in enumerate(sorted_horizons):
            ret_col = horizon_return_column(target_h)
            vol_col = horizon_volatility_column(target_h)
            mae_col = horizon_mae_column(target_h)
            mfe_col = horizon_mfe_column(target_h)
            if ret_col not in labels.columns or vol_col not in labels.columns or mae_col not in labels.columns or mfe_col not in labels.columns:
                raise ValueError(
                    f"labels.parquet is missing fixed-horizon columns for horizon={target_h}; "
                    f"regenerate via scripts/label.py with --target-horizons including {target_h}"
                )
            horizon_returns_arr[:, horizon_idx] = labels[ret_col].to_numpy(dtype=np.float32)
            horizon_vol_arr[:, horizon_idx] = np.maximum(
                labels[vol_col].to_numpy(dtype=np.float32),
                1e-6,
            )
            horizon_mae_arr[:, horizon_idx] = np.maximum(
                labels[mae_col].to_numpy(dtype=np.float32),
                0.0,
            )
            horizon_mfe_arr[:, horizon_idx] = np.maximum(
                labels[mfe_col].to_numpy(dtype=np.float32),
                0.0,
            )

    log.info(
        "train.labels_aligned",
        split=split,
        total_labels=split_total,
        usable=len(feats),
        skipped=int(valid_mask.size - valid_mask.sum()),
        skipped_quality=futures_quality_rejected_rows,
        futures_quality_rejections=futures_quality_reasons,
        return_target_mode=return_target_mode,
        mean_sample_weight=round(float(weight_arr.mean()), 6),
        opportunity_weight_median=round(opportunity_audit.median_weight, 6),
        opportunity_upweighted=round(opportunity_audit.pct_upweighted, 6),
        opportunity_downweighted=round(opportunity_audit.pct_downweighted, 6),
        mean_roundtrip_cost_return=round(opportunity_audit.mean_roundtrip_cost_return, 6),
    )
    vol_baseline_arr = np.maximum(
        labels["barrier_sigma"].to_numpy(dtype=np.float32)
        if "barrier_sigma" in labels.columns
        else return_targets.realized_vol,
        1e-6,
    )
    # Meta-labeling columns (2026-07-18, see labeling/meta_labels.py). Absent entirely for
    # labels.parquet files generated before LABEL_SCHEMA_VERSION 9 -- None here means
    # ForecastBatch.primary_side/meta_label stay None and the loss's meta-label term is
    # simply inert (see ForecasterLoss.forward's `meta_label is not None` guard), so old
    # label files keep training exactly as before, no error.
    primary_side_arr = (
        labels["primary_side"].to_numpy(dtype=np.float32)
        if "primary_side" in labels.columns
        else None
    )
    meta_label_arr = (
        labels["meta_label"].to_numpy(dtype=np.float32)
        if "meta_label" in labels.columns
        else None
    )
    return _chunk_batch(
        features=feats,
        forward_return=return_targets.forward_return,
        direction=_direction_idx(return_targets.forward_return),
        regime=regime_arr,
        realized_vol=return_targets.realized_vol,
        vol_baseline=vol_baseline_arr,
        mae=return_targets.mae,
        mfe=return_targets.mfe,
        return_weight=return_targets.return_weight,
        barrier=barrier_arr,
        barrier_weight=barrier_weight_arr,
        regime_context=regime_context_arr,
        futures=futures_arr,
        surface_grid=surface_grid_arr,
        surface_mask=surface_mask_arr,
        surface_context=surface_context_arr,
        sample_weight=weight_arr,
        horizon_returns=horizon_returns_arr,
        horizon_volatility=horizon_vol_arr,
        horizon_mae=horizon_mae_arr,
        horizon_mfe=horizon_mfe_arr,
        barrier_context=barrier_context_arr,
        primary_side=primary_side_arr,
        meta_label=meta_label_arr,
        target_horizons=sorted_horizons,
        batch_size=batch_size,
    )


def _ood_features(batches: list[ForecastBatch], limit: int = 4096) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    remaining = limit
    for batch in batches:
        if remaining <= 0:
            break
        take = min(remaining, batch.features.shape[0])
        rows.append(batch.features[:take])
        remaining -= take
    return torch.cat(rows, dim=0)


def _ood_optional(batches: list[ForecastBatch], attr: str, limit: int = 4096) -> torch.Tensor | None:
    rows: list[torch.Tensor] = []
    remaining = limit
    for batch in batches:
        tensor = getattr(batch, attr)
        if tensor is None:
            continue
        if remaining <= 0:
            break
        take = min(remaining, tensor.shape[0])
        rows.append(tensor[:take])
        remaining -= take
    return torch.cat(rows, dim=0) if rows else None


def _ood_surface(batches: list[ForecastBatch], limit: int = 4096) -> SurfaceTensors | None:
    grid = _ood_optional(batches, "surface_grid", limit)
    mask = _ood_optional(batches, "surface_mask", limit)
    context = _ood_optional(batches, "surface_context", limit)
    if grid is None or mask is None or context is None:
        return None
    return SurfaceTensors(grid=grid, mask=mask, context=context)


def _input_contract(
    dc,
    data_dir: str | None,
    batches: list[ForecastBatch],
    *,
    barrier_spec: BarrierSpec,
) -> ModelInputContract:
    vix_exists = bool(
        data_dir and (Path(data_dir) / "ohlcv" / f"INDIAVIX_{dc.base_interval}.parquet").exists()
    )
    daily_exists = bool(data_dir and (Path(data_dir) / "regime" / "daily_context.parquet").exists())
    uses_futures = any(batch.futures is not None for batch in batches)
    uses_regime_context = any(batch.regime_context is not None for batch in batches)
    uses_option_surface = any(batch.surface_grid is not None for batch in batches)
    return ModelInputContract.from_data_config(
        dc,
        feature_names=CANDLE_FEATURE_NAMES,
        uses_futures=uses_futures,
        uses_regime_context=uses_regime_context,
        uses_option_surface=uses_option_surface,
        require_vix=vix_exists,
        require_daily_context=False,
        barrier_stop_mult=barrier_spec.stop_mult,
        barrier_target_mult=barrier_spec.target_mult,
        barrier_vol_span=barrier_spec.vol_span,
        barrier_horizon_bars=barrier_spec.horizon_bars,
        barrier_cost_floor_frac=barrier_spec.cost_floor_frac,
    )


def _enabled_encoders(batches: list[ForecastBatch]) -> tuple[str, ...]:
    names = ["temporal", "cross_asset"]
    if any(batch.futures is not None for batch in batches):
        names.append("futures")
    if any(batch.surface_grid is not None for batch in batches):
        names.append("option_surface")
    if any(batch.regime_context is not None for batch in batches):
        names.append("regime")
    return tuple(names)


def _concat_batch_attr(batches: list[ForecastBatch], attr: str) -> torch.Tensor | None:
    rows = [getattr(batch, attr) for batch in batches if getattr(batch, attr) is not None]
    return torch.cat(rows, dim=0) if rows else None


def _build_world_model_sequences(
    batches: list[ForecastBatch],
    *,
    seq_len: int,
    seq_batch_size: int = _DEFAULT_WORLD_MODEL_SEQ_BATCHES,
    seq_stride: int | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, SurfaceTensors | None]]:
    features = _concat_batch_attr(batches, "features")
    if features is None or features.shape[0] < seq_len:
        raise ValueError("insufficient chronological rows for world-model sequence training")
    futures = _concat_batch_attr(batches, "futures")
    regime_context = _concat_batch_attr(batches, "regime_context")
    surface_grid = _concat_batch_attr(batches, "surface_grid")
    surface_mask = _concat_batch_attr(batches, "surface_mask")
    surface_context = _concat_batch_attr(batches, "surface_context")

    stride = seq_stride if seq_stride is not None else max(1, seq_len // 2)
    if stride < 1:
        raise ValueError("seq_stride must be >= 1")
    last_start = int(features.shape[0]) - seq_len
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    if not starts:
        raise ValueError("no sequence start positions available for world-model training")
    sequences: list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, SurfaceTensors | None]] = []
    for chunk_start in range(0, len(starts), seq_batch_size):
        chunk = starts[chunk_start : chunk_start + seq_batch_size]
        raw_seq = torch.stack([features[start : start + seq_len] for start in chunk], dim=1)
        fut_seq = (
            torch.stack([futures[start : start + seq_len] for start in chunk], dim=1)
            if futures is not None
            else None
        )
        reg_seq = (
            torch.stack([regime_context[start : start + seq_len] for start in chunk], dim=1)
            if regime_context is not None
            else None
        )
        surf_seq = (
            SurfaceTensors(
                grid=torch.stack([surface_grid[start : start + seq_len] for start in chunk], dim=1),
                mask=torch.stack([surface_mask[start : start + seq_len] for start in chunk], dim=1),
                context=torch.stack(
                    [surface_context[start : start + seq_len] for start in chunk], dim=1
                ),
            )
            if surface_grid is not None and surface_mask is not None and surface_context is not None
            else None
        )
        sequences.append((raw_seq, fut_seq, reg_seq, surf_seq))
    return sequences


def _target_symbol(*, demo: bool, primary_symbol: str) -> str:
    return DEFAULT_TARGET_SYMBOL if not demo else primary_symbol


def _artifact_path(model_path: str | None, checkpoint_dir: str, *, model_kind: str) -> Path:
    if model_path:
        return Path(model_path)
    filename = "world_model.pt" if model_kind == "world_model" else "forecaster.pt"
    return Path(checkpoint_dir) / filename


def _barrier_mode_from_cfg(cfg: dict[str, object]) -> str:
    model_cfg = cfg.get("model", {})
    raw = model_cfg.get("barrier_mode", "legacy") if isinstance(model_cfg, dict) else "legacy"
    mode = str(raw).strip().lower()
    if mode not in _SUPPORTED_BARRIER_MODES:
        raise ValueError(
            f"unsupported model.barrier_mode {raw!r}; expected one of {sorted(_SUPPORTED_BARRIER_MODES)}"
        )
    return mode


def _interval_to_minutes(interval: str) -> int:
    if interval.endswith("min"):
        return int(interval.removesuffix("min"))
    if interval.endswith("m"):
        return int(interval.removesuffix("m"))
    raise ValueError(f"unsupported interval for walk-forward model selection: {interval!r}")


def _resolved_fold_labels(labels: pd.DataFrame, row_slice: slice) -> pd.DataFrame:
    subset = labels.iloc[row_slice].copy()
    if subset.empty:
        return subset
    if "label_realized_at" in subset.columns:
        cutoff = pd.Timestamp(subset.index.max())
        realized_at = pd.to_datetime(subset["label_realized_at"])
        subset = subset.loc[realized_at <= cutoff]
    return subset.sort_index()


_CHECKPOINT_METRICS: dict[str, object] = {"loss": None, "trading_utility": trading_utility_loss}


def _train_model_once(
    *,
    model_kind: str,
    model_cfg,
    barrier_mode: str,
    loss_weights,
    train_cfg,
    batches: list[ForecastBatch],
    target_horizons: tuple[int, ...],
    pretrain_pairs: list[LatentPair] | None = None,
    val_batches: list[ForecastBatch] | None = None,
    fit_epochs: int | None = None,
    rssm_epochs: int | None = None,
    checkpoint_metric_name: str = "loss",
) -> _TrainingOutcome:
    if model_kind == "world_model":
        model = HRWWorldModel(
            n_features=len(CANDLE_FEATURE_NAMES),
            cfg=model_cfg,
            horizons=target_horizons,
            n_samples=model_cfg.rollout_samples,
        )
    else:
        model = HRWForecaster(n_features=len(CANDLE_FEATURE_NAMES), cfg=model_cfg)
    if hasattr(model, "set_barrier_mode"):
        model.set_barrier_mode(barrier_mode)

    pretrainer: MarketStatePretrainer | None = None
    if pretrain_pairs:
        pretrainer = MarketStatePretrainer(model, train_cfg)
        pretrainer.fit(pretrain_pairs, epochs=train_cfg.pretrain_epochs)

    rssm_trainer: WorldModelTrainer | None = None
    if model_kind == "world_model":
        sequence_inputs = _build_world_model_sequences(
            batches,
            seq_len=max(target_horizons),
        )
        encoded_sequences = [
            WorldModelTrainer.encode_sequence(
                raw_seq, model, futures_seq=fut_seq, regime_seq=reg_seq, surface_seq=surf_seq
            )
            for raw_seq, fut_seq, reg_seq, surf_seq in sequence_inputs
        ]
        rssm_trainer = WorldModelTrainer(model, train_cfg)
        rssm_trainer.fit(encoded_sequences, epochs=rssm_epochs)
        # Review finding M12: actually monitor RSSM posterior collapse instead of
        # leaving world_model_metrics' diagnostics unreachable dead code.
        rssm_diagnostics = rssm_trainer.diagnostics(encoded_sequences)
        log.info("train.rssm_diagnostics", **{k: round(v, 6) for k, v in rssm_diagnostics.items()})
        if rssm_diagnostics.get("kl_collapse_frac", 0.0) > 0.5:
            log.warning(
                "train.rssm_kl_collapse_suspected kl_collapse_frac=%s mean_kl=%s",
                rssm_diagnostics.get("kl_collapse_frac"),
                rssm_diagnostics.get("mean_kl"),
            )
        loss = WorldModelLoss(weights=loss_weights)
    else:
        loss = ForecasterLoss(weights=loss_weights)

    if checkpoint_metric_name not in _CHECKPOINT_METRICS:
        raise ValueError(
            f"unsupported checkpoint_metric_name: {checkpoint_metric_name!r} "
            f"(choices: {sorted(_CHECKPOINT_METRICS)})"
        )
    trainer = HRWTrainer(
        model, loss, train_cfg, checkpoint_metric=_CHECKPOINT_METRICS[checkpoint_metric_name]
    )
    trainer.fit(batches, epochs=fit_epochs, val_batches=val_batches or None)

    # Stage 4 (review finding H7): optional head-only fine-tuning pass on top of
    # the just-completed joint encoder+heads fit. HeadTrainer only applies to the
    # compact forecaster (HRWWorldModel has no equivalent freeze-encoder notion
    # here); off by default (head_finetune_epochs=0) so this changes nothing
    # unless explicitly opted into.
    head_trainer: HeadTrainer | None = None
    if model_kind != "world_model" and train_cfg.head_finetune_epochs > 0:
        head_trainer = HeadTrainer(model, loss, train_cfg, freeze_encoder=True)
        head_trainer.fit(batches, epochs=train_cfg.head_finetune_epochs)

    model.fit_ood(
        _ood_features(batches),
        futures=_ood_optional(batches, "futures"),
        regime=_ood_optional(batches, "regime_context"),
        surface=_ood_surface(batches),
    )
    return _TrainingOutcome(
        model=model,
        trainer=trainer,
        pretrainer=pretrainer,
        rssm_trainer=rssm_trainer,
        head_trainer=head_trainer,
    )


def _collect_prediction_calibration_inputs(
    model: HRWForecaster | HRWWorldModel,
    *,
    model_kind: str,
    batches: list[ForecastBatch],
    target_horizons: tuple[int, ...],
    quantile_levels: tuple[float, ...] = DEFAULT_QUANTILES,
) -> _PredictionCalibrationInputs:
    buffers = _PredictionCalibrationInputs(quantile_levels=quantile_levels)
    if not batches:
        return buffers

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in batches:
            batch_device = batch.to(device)
            surface = None
            if (
                batch_device.surface_grid is not None
                and batch_device.surface_mask is not None
                and batch_device.surface_context is not None
            ):
                surface = SurfaceTensors(
                    grid=batch_device.surface_grid,
                    mask=batch_device.surface_mask,
                    context=batch_device.surface_context,
                )
            forward_kwargs: dict[str, object] = {}
            if surface is not None and "surface" in inspect.signature(model.forward).parameters:
                forward_kwargs["surface"] = surface
            output = model.forward(
                batch_device.features,
                batch_device.futures,
                batch_device.regime_context,
                batch_device.barrier_context,
                **forward_kwargs,
            )
            if model_kind == "world_model":
                pred_quantiles = output["return_quantiles"].detach().cpu().numpy()
                pred_volatility = output["volatility"].detach().cpu().numpy()
                if batch_device.horizon_returns is None:
                    continue
                realized = batch_device.horizon_returns.detach().cpu().numpy()
                realized_vol = (
                    batch_device.horizon_volatility.detach().cpu().numpy()
                    if batch_device.horizon_volatility is not None
                    else np.maximum(np.abs(realized), 1e-6)
                )
                for idx, horizon in enumerate(target_horizons):
                    valid = np.isfinite(realized[:, idx]) & np.isfinite(realized_vol[:, idx])
                    if not bool(valid.any()):
                        continue
                    buffers.append_horizon(
                        horizon=int(horizon),
                        pred_quantiles=pred_quantiles[valid, idx, :],
                        realized=realized[valid, idx],
                        predicted_volatility=pred_volatility[valid, idx],
                        realized_volatility=realized_vol[valid, idx],
                    )
                barrier_probs = output["barrier_probs"].detach().cpu().numpy()
            else:
                pred_quantiles = output["return_quantiles"].detach().cpu().numpy()
                pred_volatility = output["volatility"].detach().reshape(-1).cpu().numpy()
                realized = batch_device.forward_return.detach().cpu().numpy()
                realized_vol = (
                    batch_device.realized_vol.detach().cpu().numpy()
                    if batch_device.realized_vol is not None
                    else np.maximum(np.abs(realized), 1e-6)
                )
                valid = np.isfinite(realized) & np.isfinite(realized_vol)
                if batch_device.return_weight is not None:
                    valid &= batch_device.return_weight.detach().cpu().numpy() > 0.0
                if bool(valid.any()):
                    buffers.append_horizon(
                        horizon=int(target_horizons[-1]),
                        pred_quantiles=pred_quantiles[valid],
                        realized=realized[valid],
                        predicted_volatility=pred_volatility[valid],
                        realized_volatility=realized_vol[valid],
                    )
                barrier_probs = torch.softmax(output["barrier_logits"].detach(), dim=-1).cpu().numpy()

            if batch_device.barrier is not None:
                barrier_labels = batch_device.barrier.detach().cpu().numpy().astype(np.int64, copy=False)
                barrier_mask = None
                if batch_device.barrier_weight is not None:
                    barrier_mask = (
                        batch_device.barrier_weight.detach().cpu().numpy() > 0.0
                    )
                if barrier_mask is None:
                    buffers.barrier_probs.extend(barrier_probs.tolist())
                    buffers.barrier_labels.extend(barrier_labels.tolist())
                elif bool(barrier_mask.any()):
                    buffers.barrier_probs.extend(barrier_probs[barrier_mask].tolist())
                    buffers.barrier_labels.extend(barrier_labels[barrier_mask].tolist())
            if batch_device.regime is not None:
                regime_probs = torch.softmax(output["regime_logits"].detach(), dim=-1).cpu().numpy()
                regime_labels = batch_device.regime.detach().cpu().numpy().astype(np.int64, copy=False)
                buffers.regime_probs.extend(regime_probs.tolist())
                buffers.regime_labels.extend(regime_labels.tolist())
    if was_training:
        model.train()
    return buffers


def _fit_posthoc_prediction_calibration(
    calibration_inputs: _PredictionCalibrationInputs,
    *,
    source: str,
    allow_class_probability_temperatures: bool = True,
) -> dict[str, object] | None:
    calibration = fit_prediction_calibration(
        quantile_levels=calibration_inputs.quantile_levels,
        horizon_payloads=calibration_inputs.horizon_payloads,
        barrier_probs=calibration_inputs.barrier_probs or None,
        barrier_labels=calibration_inputs.barrier_labels or None,
        regime_probs=calibration_inputs.regime_probs or None,
        regime_labels=calibration_inputs.regime_labels or None,
        source=source,
    )
    if calibration is None:
        return None
    metadata = calibration.to_metadata()
    if not allow_class_probability_temperatures:
        metadata["barrier_temperature"] = 1.0
        metadata["regime_temperature"] = 1.0
        # The prior-offset correction (2026-07-18) is a class-probability transform
        # too — strip it under the same gate so "classification transfer disabled"
        # really means identity on the class probabilities.
        metadata.pop("barrier_prior_offsets", None)
        metadata["classification_transfer_disabled"] = True
    return metadata


def _walk_forward_model_selection(
    *,
    data_dir: str,
    dc,
    labels: pd.DataFrame,
    split_manifest: ChronoSplitManifest,
    management_horizon: int,
    execution_cfg,
    batch_size: int,
    target_horizons: tuple[int, ...],
    model_kind: str,
    model_cfg,
    barrier_mode: str,
    return_target_mode: str,
    loss_weights,
    train_cfg,
    rssm_epochs: int,
) -> dict[str, object] | None:
    if train_cfg.n_folds < 2:
        return None
    pretest_labels = split_manifest.filter_labels(labels, "pretest").sort_index()
    if len(pretest_labels) < 3:
        return None
    walk_forward = WalkForward(
        n_folds=train_cfg.n_folds,
        embargo_bars=train_cfg.embargo_bars,
        bar_minutes=_interval_to_minutes(dc.base_interval),
    )
    try:
        folds = walk_forward.split_indices(len(pretest_labels))
    except ValueError as exc:
        log.warning("train.model_selection_unavailable note=%s", str(exc))
        return None

    fold_reports: list[_ModelSelectionFold] = []
    oof_calibration = _PredictionCalibrationInputs()
    epoch_votes: list[int] = []
    for fold in folds:
        train_labels = _resolved_fold_labels(pretest_labels, fold.train_slice)
        val_labels = _resolved_fold_labels(pretest_labels, fold.val_slice)
        if train_labels.empty or val_labels.empty:
            log.warning(
                "train.model_selection_fold_skipped fold_id=%s reason=%s",
                fold.fold_id,
                "empty_train_or_val_after_resolution_filter",
            )
            continue
        try:
            train_batches = build_labeled_batches(
                data_dir,
                dc.universe,
                dc.base_interval,
                dc.lookback_bars,
                train_labels,
                management_horizon=management_horizon,
                execution_cfg=execution_cfg,
                batch_size=batch_size,
                split="all",
                target_horizons=target_horizons if model_kind == "world_model" else None,
                return_target_mode=return_target_mode,
            )
            val_batches = build_labeled_batches(
                data_dir,
                dc.universe,
                dc.base_interval,
                dc.lookback_bars,
                val_labels,
                management_horizon=management_horizon,
                execution_cfg=execution_cfg,
                batch_size=batch_size,
                split="all",
                target_horizons=target_horizons if model_kind == "world_model" else None,
                return_target_mode=return_target_mode,
            )
            fold_pretrain_pairs = (
                build_market_pretrain_pairs(
                    data_dir,
                    dc.universe,
                    dc.base_interval,
                    dc.lookback_bars,
                    batch_size=batch_size,
                    gap_bars=train_cfg.pretrain_gap_bars,
                    split="all",
                    end_ts=pd.Timestamp(train_labels.index.max()),
                )
                if train_cfg.pretrain_epochs > 0
                else []
            )
        except ValueError as exc:
            log.warning(
                "train.model_selection_fold_skipped fold_id=%s reason=%s",
                fold.fold_id,
                str(exc),
            )
            continue
        outcome = _train_model_once(
            model_kind=model_kind,
            model_cfg=model_cfg,
            barrier_mode=barrier_mode,
            loss_weights=loss_weights,
            train_cfg=train_cfg,
            batches=train_batches,
            target_horizons=target_horizons,
            pretrain_pairs=fold_pretrain_pairs,
            val_batches=val_batches,
            rssm_epochs=rssm_epochs if model_kind == "world_model" else None,
        )
        best_epoch = outcome.trainer.best_epoch or len(outcome.trainer.history)
        best_val_loss = (
            float(min(outcome.trainer.val_history))
            if outcome.trainer.val_history
            else None
        )
        fold_reports.append(
            _ModelSelectionFold(
                fold_id=fold.fold_id,
                train_rows=len(train_labels),
                val_rows=len(val_labels),
                best_epoch=int(best_epoch),
                best_val_loss=best_val_loss,
            )
        )
        oof_calibration.extend(
            _collect_prediction_calibration_inputs(
                outcome.model,
                model_kind=model_kind,
                batches=val_batches,
                target_horizons=target_horizons if model_kind == "world_model" else (management_horizon,),
            )
        )
        epoch_votes.append(int(best_epoch))

    if not fold_reports:
        return None

    selected_epochs = int(
        np.clip(
            int(np.rint(np.median(np.asarray(epoch_votes, dtype=float)))),
            1,
            train_cfg.max_epochs,
        )
    )
    summary = {
        "method": "walk_forward_cv",
        "pretest_rows": len(pretest_labels),
        "n_requested_folds": train_cfg.n_folds,
        "n_completed_folds": len(fold_reports),
        "selected_supervised_epochs": selected_epochs,
        "folds": [
            {
                "fold_id": fold.fold_id,
                "train_rows": fold.train_rows,
                "val_rows": fold.val_rows,
                "best_epoch": fold.best_epoch,
                "best_val_loss": fold.best_val_loss,
            }
            for fold in fold_reports
        ],
    }
    prediction_calibration = _fit_posthoc_prediction_calibration(
        oof_calibration,
        source="walk_forward_oof_pretest",
        allow_class_probability_temperatures=False,
    )
    if prediction_calibration is not None:
        summary["prediction_calibration"] = prediction_calibration
    return summary


def main() -> None:
    args, cfg = setup(
        "Train the promoted HRW model workflow and save an artifact.",
        option_groups=(
            "demo",
            "data_dir",
            "labels_path",
            "model_path",
            "model_kind",
            "pretraining",
            "checkpoint_metric",
        ),
    )
    dc = data_config_from_cfg(cfg)
    provenance_report = validate_upstox_only_sources(dc.data_sources_path)
    log.info("train.data_provenance", **provenance_report)
    horizon = management_horizon_from_cfg(cfg)
    horizons_cfg = cfg.get("horizons", {}) if isinstance(cfg, dict) else {}
    target_horizons = tuple(sorted(set(horizons_cfg.get("horizon_steps", [horizon]))))
    model_cfg = model_config_from_cfg(cfg)
    barrier_mode = _barrier_mode_from_cfg(cfg)
    return_target_mode = _return_target_mode_from_cfg(cfg)
    loss_weights = loss_weights_from_cfg(cfg)
    execution_cfg = execution_config_from_cfg(cfg)
    train_cfg = training_config_from_cfg(cfg)
    # Bugfix (2026-07-13): `--seed` was accepted, logged, and applied to Python's/numpy's
    # RNGs inside `setup()` -- but never threaded into `TrainingConfig.seed`, which is what
    # `HRWTrainer.fit()` actually calls `torch.manual_seed()` with. Since every actual
    # weight-init/dropout/batch-order source of randomness in this codebase goes through
    # torch, `--seed` was a complete no-op for training reproducibility/variation --
    # confirmed empirically (two runs differing only in `--seed` produced byte-identical
    # calibration metrics to 7 decimal places). `setup()` doesn't expose its own resolved
    # seed value, so it's recomputed here with the same precedence (CLI overrides config).
    resolved_seed = args.seed if getattr(args, "seed", None) is not None else train_cfg.seed
    train_cfg = replace(train_cfg, seed=resolved_seed)
    train_cfg = replace(train_cfg, embargo_bars=max(train_cfg.embargo_bars, horizon))
    rssm_epochs = (
        int(args.rssm_epochs)
        if getattr(args, "rssm_epochs", None) is not None
        else max(1, min(10, train_cfg.max_epochs // 2))
    )
    if args.pretrain_epochs is not None or args.pretrain_gap_bars is not None:
        train_cfg = replace(
            train_cfg,
            pretrain_epochs=(
                train_cfg.pretrain_epochs
                if args.pretrain_epochs is None
                else int(args.pretrain_epochs)
            ),
            pretrain_gap_bars=(
                train_cfg.pretrain_gap_bars
                if args.pretrain_gap_bars is None
                else int(args.pretrain_gap_bars)
            ),
        )
    if getattr(args, "head_finetune_epochs", None) is not None:
        train_cfg = replace(train_cfg, head_finetune_epochs=int(args.head_finetune_epochs))
    val_batches: list[ForecastBatch] = []
    pretrain_pairs: list[LatentPair] = []
    split_manifest: ChronoSplitManifest | None = None
    capability_profile: DataCapabilityProfile | None = None
    model_selection_summary: dict[str, object] | None = None
    supervised_fit_epochs: int | None = None
    prediction_calibration: dict[str, object] | None = None
    training_split = "train"
    barrier_spec = BarrierSpec()

    if args.demo:
        batches = build_demo_batches(
            dc.universe,
            dc.lookback_bars,
            horizon,
            batch_size=train_cfg.batch_size,
            target_horizons=target_horizons if args.model_kind == "world_model" else None,
        )
        if train_cfg.pretrain_epochs > 0:
            pretrain_pairs = build_demo_pretrain_pairs(
                dc.universe,
                dc.lookback_bars,
                batch_size=train_cfg.batch_size,
                gap_bars=train_cfg.pretrain_gap_bars,
            )
        data_mode = "demo"
    elif args.data_dir and args.labels_path:
        labels = _load_labels_frame(args.labels_path)
        barrier_spec = _resolve_barrier_spec(labels)
        split_manifest = ChronoSplitManifest.from_labels(
            labels,
            train_fraction=train_cfg.train_fraction,
            val_fraction=train_cfg.val_fraction,
            embargo_bars=train_cfg.embargo_bars,
            bar_interval=dc.base_interval,
        )
        if loss_weights.barrier_class_weights is None:
            computed_class_weights = _barrier_class_weights_from_labels(
                labels, pd.Timestamp(split_manifest.train_end)
            )
            if computed_class_weights is not None:
                loss_weights = replace(loss_weights, barrier_class_weights=computed_class_weights)
                log.info(
                    "train.barrier_class_weights",
                    stop=computed_class_weights[0],
                    target=computed_class_weights[1],
                    timeout=computed_class_weights[2],
                )
        capability_profile = DataCapabilityProfile.from_data_dir(args.data_dir, dc)
        model_selection_summary = _walk_forward_model_selection(
            data_dir=args.data_dir,
            dc=dc,
            labels=labels,
            split_manifest=split_manifest,
            management_horizon=horizon,
            execution_cfg=execution_cfg,
            batch_size=train_cfg.batch_size,
            target_horizons=target_horizons,
            model_kind=args.model_kind,
            model_cfg=model_cfg,
            barrier_mode=barrier_mode,
            return_target_mode=return_target_mode,
            loss_weights=loss_weights,
            train_cfg=train_cfg,
            rssm_epochs=rssm_epochs,
        )
        if model_selection_summary is not None:
            supervised_fit_epochs = int(model_selection_summary["selected_supervised_epochs"])
            # Final-refit validation (2026-07-12 fix): this used to train on the FULL "pretest"
            # split (train+val combined) with val_batches=[] -- HRWTrainer.fit() only restores
            # the best checkpoint when val_batches is non-empty, so this stage (unlike each CV
            # fold, which does track best_val_loss and roll back) never benefited from that
            # safeguard. Confirmed empirically: this final refit's own loss curve is
            # non-monotonic (one run's loss rose from 13.37 at epoch 16 back up to 13.98 by
            # epoch 24, with no mechanism to recover the better checkpoint -- the reported
            # "final" model was whatever the fixed epoch count happened to land on, not its own
            # best point). Training on "train"/validating on "val" (the same split_manifest
            # boundary the non-CV code path below already uses correctly) costs some final
            # training data (val is held out, not just an internal CV slice) but makes
            # `selected_supervised_epochs` a genuine ceiling with real early-stopping/
            # best-checkpoint selection underneath it, not a blind fixed count.
            training_split = "train"
            batches = build_labeled_batches(
                args.data_dir,
                dc.universe,
                dc.base_interval,
                dc.lookback_bars,
                labels,
                management_horizon=horizon,
                execution_cfg=execution_cfg,
                batch_size=train_cfg.batch_size,
                split="train",
                split_manifest=split_manifest,
                target_horizons=target_horizons if args.model_kind == "world_model" else None,
                return_target_mode=return_target_mode,
            )
            try:
                val_batches = build_labeled_batches(
                    args.data_dir,
                    dc.universe,
                    dc.base_interval,
                    dc.lookback_bars,
                    labels,
                    management_horizon=horizon,
                    execution_cfg=execution_cfg,
                    batch_size=train_cfg.batch_size,
                    split="val",
                    split_manifest=split_manifest,
                    target_horizons=target_horizons if args.model_kind == "world_model" else None,
                    return_target_mode=return_target_mode,
                )
            except ValueError:
                val_batches = []
            if train_cfg.pretrain_epochs > 0:
                pretrain_pairs = build_market_pretrain_pairs(
                    args.data_dir,
                    dc.universe,
                    dc.base_interval,
                    dc.lookback_bars,
                    batch_size=train_cfg.batch_size,
                    gap_bars=train_cfg.pretrain_gap_bars,
                    split="train",
                    split_manifest=split_manifest,
                )
        else:
            batches = build_labeled_batches(
                args.data_dir,
                dc.universe,
                dc.base_interval,
                dc.lookback_bars,
                labels,
                management_horizon=horizon,
                execution_cfg=execution_cfg,
                batch_size=train_cfg.batch_size,
                split="train",
                split_manifest=split_manifest,
                target_horizons=target_horizons if args.model_kind == "world_model" else None,
                return_target_mode=return_target_mode,
            )
            try:
                val_batches = build_labeled_batches(
                    args.data_dir,
                    dc.universe,
                    dc.base_interval,
                    dc.lookback_bars,
                    labels,
                    management_horizon=horizon,
                    execution_cfg=execution_cfg,
                    batch_size=train_cfg.batch_size,
                    split="val",
                    split_manifest=split_manifest,
                    target_horizons=target_horizons if args.model_kind == "world_model" else None,
                    return_target_mode=return_target_mode,
                )
            except ValueError:
                val_batches = []
            if train_cfg.pretrain_epochs > 0:
                pretrain_pairs = build_market_pretrain_pairs(
                    args.data_dir,
                    dc.universe,
                    dc.base_interval,
                    dc.lookback_bars,
                    batch_size=train_cfg.batch_size,
                    gap_bars=train_cfg.pretrain_gap_bars,
                    split="train",
                    split_manifest=split_manifest,
                )
        data_mode = "real"
    else:
        log.warning(
            "train.no_source note=%s",
            "Use --demo, or provide --data-dir and --labels-path for real training.",
        )
        return

    n_samples = int(sum(batch.features.shape[0] for batch in batches))
    log.info(
        "train.dataset",
        mode=data_mode,
        n_batches=len(batches),
        samples=n_samples,
        batch_size=train_cfg.batch_size,
        horizon_bars=horizon,
        model_kind=args.model_kind,
        training_split=training_split,
        barrier_mode=barrier_mode,
        return_target_mode=return_target_mode,
        target_horizons=target_horizons if args.model_kind == "world_model" else (horizon,),
        val_samples=int(sum(batch.features.shape[0] for batch in val_batches)) if val_batches else 0,
        pretrain_pairs=int(sum(pair.context.shape[0] for pair in pretrain_pairs)),
        pretrain_gap_bars=train_cfg.pretrain_gap_bars,
        pretrain_epochs=train_cfg.pretrain_epochs,
        rssm_epochs=rssm_epochs if args.model_kind == "world_model" else 0,
        selected_supervised_epochs=supervised_fit_epochs,
        model_selection=model_selection_summary,
        split_manifest=split_manifest.to_metadata() if split_manifest is not None else None,
        capability_issues=list(capability_profile.critical_issues()) if capability_profile else [],
    )

    outcome = _train_model_once(
        model_kind=args.model_kind,
        model_cfg=model_cfg,
        barrier_mode=barrier_mode,
        loss_weights=loss_weights,
        train_cfg=train_cfg,
        batches=batches,
        target_horizons=target_horizons,
        pretrain_pairs=pretrain_pairs,
        val_batches=val_batches,
        fit_epochs=supervised_fit_epochs,
        rssm_epochs=rssm_epochs if args.model_kind == "world_model" else None,
        checkpoint_metric_name=getattr(args, "checkpoint_metric", None) or "loss",
    )
    model = outcome.model
    trainer = outcome.trainer
    if outcome.pretrainer is not None:
        log.info(
            "train.pretrain_done",
            epochs=train_cfg.pretrain_epochs,
            first_loss=round(outcome.pretrainer.history[0], 6),
            last_loss=round(outcome.pretrainer.history[-1], 6),
            latent_std=round(outcome.pretrainer.latent_collapse_std(pretrain_pairs), 6),
        )
    if outcome.rssm_trainer is not None:
        sequence_inputs = _build_world_model_sequences(
            batches,
            seq_len=max(target_horizons),
        )
        log.info(
            "train.world_model_dynamics_done",
            sequences=len(sequence_inputs),
            epochs=rssm_epochs,
            first_loss=round(outcome.rssm_trainer.history[0], 6),
            last_loss=round(outcome.rssm_trainer.history[-1], 6),
        )
    # Bugfix (2026-07-13): the walk-forward-CV path used to unconditionally inherit its OOF
    # calibration fit, which forces barrier_temperature/regime_temperature to 1.0 (identity --
    # see `_fit_posthoc_prediction_calibration`'s `allow_class_probability_temperatures=False`)
    # because those OOF probabilities come from the 5 different FOLD models, not the actual
    # final model being evaluated/deployed -- transferring a class-probability temperature
    # across different model instances is unsafe. Confirmed empirically: barrier_temperature
    # was exactly 1.0 in every walk-forward-CV run this session, and barrier_brier/ECE never
    # improved from it. Now that the final refit always has genuine held-out `val_batches`
    # (see the walk-forward-CV branch above, fixed the same day), fit the calibration --
    # INCLUDING class-probability temperatures -- directly from the actual final model's own
    # val-split predictions instead, falling back to the OOF-CV summary only if val_batches
    # is unexpectedly empty.
    if val_batches:
        prediction_calibration = _fit_posthoc_prediction_calibration(
            _collect_prediction_calibration_inputs(
                model,
                model_kind=args.model_kind,
                batches=val_batches,
                target_horizons=target_horizons if args.model_kind == "world_model" else (horizon,),
            ),
            source="val_split",
        )
    elif model_selection_summary is not None:
        candidate = model_selection_summary.get("prediction_calibration")
        if isinstance(candidate, dict):
            prediction_calibration = candidate
    if prediction_calibration is not None:
        log.info(
            "train.prediction_calibration",
            source=prediction_calibration.get("source"),
            sample_count=prediction_calibration.get("sample_count"),
            barrier_temperature=round(
                float(prediction_calibration.get("barrier_temperature", 1.0)),
                6,
            ),
            regime_temperature=round(
                float(prediction_calibration.get("regime_temperature", 1.0)),
                6,
            ),
            calibrated_horizons=sorted(
                int(horizon_key)
                for horizon_key in dict(prediction_calibration.get("horizons", {})).keys()
            ),
        )

    artifact_path = _artifact_path(
        args.model_path,
        train_cfg.checkpoint_dir,
        model_kind=args.model_kind,
    )
    log.info(
        "train.done",
        first_loss=round(trainer.history[0], 6),
        last_loss=round(trainer.history[-1], 6),
        best_epoch=trainer.best_epoch,
        best_val_loss=round(min(trainer.val_history), 6) if trainer.val_history else None,
        selected_supervised_epochs=supervised_fit_epochs,
        artifact=str(artifact_path),
    )
    print("\nTRAINING SUMMARY")
    print("----------------")
    print(f"model_kind: {args.model_kind}")
    print(f"data_mode: {data_mode}")
    print(f"first_loss: {round(trainer.history[0], 6)}")
    print(f"last_loss: {round(trainer.history[-1], 6)}")
    print(f"best_epoch: {trainer.best_epoch}")
    print(f"best_val_loss: {round(min(trainer.val_history), 6) if trainer.val_history else 'NA'}")
    print(f"selected_supervised_epochs: {supervised_fit_epochs}")
    print(f"artifact: {artifact_path}")
    print(f"dry_run: {bool(args.dry_run)}", flush=True)
    if args.dry_run:
        log.info("train.dry_run", note="skipping artifact write")
        return

    input_contract = _input_contract(
        dc,
        None if data_mode == "demo" else args.data_dir,
        batches,
        barrier_spec=barrier_spec,
    )
    common_save_kwargs = dict(
        input_contract=input_contract,
        split_manifest=split_manifest.to_metadata() if split_manifest is not None else None,
        data_capability_profile=(
            capability_profile.to_metadata() if capability_profile is not None else None
        ),
        enabled_encoders=_enabled_encoders(batches),
        disabled_optional_modules=(
            capability_profile.disabled_optional_modules() if capability_profile is not None else ()
        ),
        target_symbol=_target_symbol(demo=(data_mode == "demo"), primary_symbol=dc.universe[0]),
        model_selection=model_selection_summary,
        prediction_calibration=prediction_calibration,
        return_target_mode=return_target_mode,
    )
    if args.model_kind == "world_model":
        saved = save_world_model_artifact(
            artifact_path,
            model,
            n_features=len(CANDLE_FEATURE_NAMES),
            horizons=target_horizons,
            cfg=model_cfg,
            n_samples=model_cfg.rollout_samples,
            **common_save_kwargs,
        )
    else:
        saved = save_forecaster_artifact(
            artifact_path,
            model,
            n_features=len(CANDLE_FEATURE_NAMES),
            horizon_bars=horizon,
            cfg=model_cfg,
            **common_save_kwargs,
        )
    log.info("train.saved", path=str(saved))


if __name__ == "__main__":
    main()
