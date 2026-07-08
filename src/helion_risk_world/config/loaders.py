"""Helpers that map raw config payloads into typed runtime objects."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from helion_risk_world.backtesting import WalkForward
from helion_risk_world.config.data_config import DataConfig
from helion_risk_world.config.execution_config import CostModelConfig, InstrumentSpecConfig
from helion_risk_world.config.model_config import LossWeights, ModelConfig
from helion_risk_world.config.training_config import TrainingConfig
from helion_risk_world.strategy import StrategyProfile, get_strategy_profile

_DATA_DEFAULTS = DataConfig()
_MODEL_DEFAULTS = ModelConfig()
_LOSS_DEFAULTS = LossWeights()
_TRAINING_DEFAULTS = TrainingConfig()
_EXECUTION_DEFAULTS = CostModelConfig()


def data_config_from_mapping(cfg: Mapping[str, Any] | None) -> DataConfig:
    data = _section(cfg, "data")
    return DataConfig(
        universe=tuple(data.get("universe", _DATA_DEFAULTS.universe)),
        base_interval=str(data.get("base_interval", _DATA_DEFAULTS.base_interval)),
        lookback_bars=int(data.get("lookback_bars", _DATA_DEFAULTS.lookback_bars)),
        n_strikes=int(data.get("n_strikes", _DATA_DEFAULTS.n_strikes)),
        feature_cache_dir=str(
            data.get("feature_cache_dir", _DATA_DEFAULTS.feature_cache_dir)
        ),
        data_sources_path=str(
            data.get("data_sources_path", _DATA_DEFAULTS.data_sources_path)
        ),
        use_historical_depth=bool(
            data.get("use_historical_depth", _DATA_DEFAULTS.use_historical_depth)
        ),
    )


def model_config_from_mapping(cfg: Mapping[str, Any] | None) -> ModelConfig:
    model = _section(cfg, "model")
    return ModelConfig(
        size=str(model.get("size", _MODEL_DEFAULTS.size)),
        latent_dim=int(model.get("latent_dim", _MODEL_DEFAULTS.latent_dim)),
        temporal_layers=int(
            model.get("temporal_layers", _MODEL_DEFAULTS.temporal_layers)
        ),
        futures_conv_layers=int(
            model.get("futures_conv_layers", _MODEL_DEFAULTS.futures_conv_layers)
        ),
        cross_asset_heads=int(
            model.get("cross_asset_heads", _MODEL_DEFAULTS.cross_asset_heads)
        ),
        fusion=str(model.get("fusion", _MODEL_DEFAULTS.fusion)),
        rollout_samples=int(
            model.get("rollout_samples", _MODEL_DEFAULTS.rollout_samples)
        ),
        dropout=float(model.get("dropout", _MODEL_DEFAULTS.dropout)),
    )


def loss_weights_from_mapping(cfg: Mapping[str, Any] | None) -> LossWeights:
    loss_cfg = _section(cfg, "loss")
    weights_cfg = loss_cfg.get("weights", {})
    if not isinstance(weights_cfg, Mapping):
        weights_cfg = {}
    return LossWeights(
        return_=float(weights_cfg.get("return", _LOSS_DEFAULTS.return_)),
        direction=float(weights_cfg.get("direction", _LOSS_DEFAULTS.direction)),
        volatility=float(weights_cfg.get("volatility", _LOSS_DEFAULTS.volatility)),
        mae=float(weights_cfg.get("mae", _LOSS_DEFAULTS.mae)),
        mfe=float(weights_cfg.get("mfe", _LOSS_DEFAULTS.mfe)),
        barrier=float(weights_cfg.get("barrier", _LOSS_DEFAULTS.barrier)),
        barrier_intermediate=float(
            weights_cfg.get("barrier_intermediate", _LOSS_DEFAULTS.barrier_intermediate)
        ),
        regime=float(weights_cfg.get("regime", _LOSS_DEFAULTS.regime)),
        calibration=float(weights_cfg.get("calibration", _LOSS_DEFAULTS.calibration)),
        uncertainty=float(weights_cfg.get("uncertainty", _LOSS_DEFAULTS.uncertainty)),
        ood=float(weights_cfg.get("ood", _LOSS_DEFAULTS.ood)),
        barrier_class_weights=_barrier_class_weights_from_mapping(weights_cfg),
    )


def _barrier_class_weights_from_mapping(
    weights_cfg: Mapping[str, Any],
) -> tuple[float, float, float] | None:
    raw = weights_cfg.get("barrier_class_weights")
    if raw is None:
        return None
    values = tuple(float(v) for v in raw)
    if len(values) != 3:
        raise ValueError(
            f"loss.weights.barrier_class_weights must have exactly 3 entries [stop, target, "
            f"timeout]; got {values!r}"
        )
    return values


def training_config_from_mapping(
    cfg: Mapping[str, Any] | None,
    *,
    embargo_bars: int | None = None,
) -> TrainingConfig:
    training = _section(cfg, "training")
    root = cfg if isinstance(cfg, Mapping) else {}
    return TrainingConfig(
        seed=int(root.get("seed", _TRAINING_DEFAULTS.seed)),
        device=str(training.get("device", _TRAINING_DEFAULTS.device)),
        batch_size=int(training.get("batch_size", _TRAINING_DEFAULTS.batch_size)),
        grad_accum_steps=int(
            training.get("grad_accum_steps", _TRAINING_DEFAULTS.grad_accum_steps)
        ),
        max_epochs=int(training.get("max_epochs", _TRAINING_DEFAULTS.max_epochs)),
        lr=float(training.get("lr", _TRAINING_DEFAULTS.lr)),
        weight_decay=float(training.get("weight_decay", _TRAINING_DEFAULTS.weight_decay)),
        grad_clip_norm=float(
            training.get("grad_clip_norm", _TRAINING_DEFAULTS.grad_clip_norm)
        ),
        early_stopping_patience=int(
            training.get(
                "early_stopping_patience",
                _TRAINING_DEFAULTS.early_stopping_patience,
            )
        ),
        checkpoint_dir=str(
            training.get("checkpoint_dir", _TRAINING_DEFAULTS.checkpoint_dir)
        ),
        pretrain_epochs=int(
            training.get("pretrain_epochs", _TRAINING_DEFAULTS.pretrain_epochs)
        ),
        head_finetune_epochs=int(
            training.get("head_finetune_epochs", _TRAINING_DEFAULTS.head_finetune_epochs)
        ),
        pretrain_gap_bars=int(
            training.get("pretrain_gap_bars", _TRAINING_DEFAULTS.pretrain_gap_bars)
        ),
        train_fraction=float(
            training.get("train_fraction", _TRAINING_DEFAULTS.train_fraction)
        ),
        val_fraction=float(
            training.get("val_fraction", _TRAINING_DEFAULTS.val_fraction)
        ),
        n_folds=int(training.get("n_folds", _TRAINING_DEFAULTS.n_folds)),
        embargo_bars=int(
            embargo_bars
            if embargo_bars is not None
            else training.get("embargo_bars", _TRAINING_DEFAULTS.embargo_bars)
        ),
    )


def execution_config_from_mapping(cfg: Mapping[str, Any] | None) -> CostModelConfig:
    execution = _section(cfg, "execution")
    specs_cfg = execution.get("instrument_specs", {})
    if not isinstance(specs_cfg, Mapping):
        specs_cfg = {}
    specs: dict[str, InstrumentSpecConfig] = {}
    for symbol, raw_spec in specs_cfg.items():
        if not isinstance(raw_spec, Mapping):
            continue
        tick_size = raw_spec.get("tick_size")
        specs[str(symbol).upper()] = InstrumentSpecConfig(
            lot_size=float(raw_spec.get("lot_size", 1.0)),
            tick_size=float(tick_size) if tick_size is not None else None,
            margin_fraction=float(raw_spec.get("margin_fraction", 1.0)),
            quantity_step=int(raw_spec.get("quantity_step", 1)),
        )
    return CostModelConfig(
        brokerage_per_order=float(
            execution.get(
                "brokerage_per_order",
                _EXECUTION_DEFAULTS.brokerage_per_order,
            )
        ),
        stt_rate=float(execution.get("stt_rate", _EXECUTION_DEFAULTS.stt_rate)),
        exchange_txn_rate=float(
            execution.get("exchange_txn_rate", _EXECUTION_DEFAULTS.exchange_txn_rate)
        ),
        gst_rate=float(execution.get("gst_rate", _EXECUTION_DEFAULTS.gst_rate)),
        sebi_rate=float(execution.get("sebi_rate", _EXECUTION_DEFAULTS.sebi_rate)),
        stamp_duty_rate=float(
            execution.get("stamp_duty_rate", _EXECUTION_DEFAULTS.stamp_duty_rate)
        ),
        half_spread_bps=float(
            execution.get("half_spread_bps", _EXECUTION_DEFAULTS.half_spread_bps)
        ),
        slippage_bps=float(
            execution.get("slippage_bps", _EXECUTION_DEFAULTS.slippage_bps)
        ),
        default_spread_ticks=float(
            execution.get(
                "default_spread_ticks",
                _EXECUTION_DEFAULTS.default_spread_ticks,
            )
        ),
        tick_size=float(execution.get("tick_size", _EXECUTION_DEFAULTS.tick_size)),
        base_fill_prob=float(
            execution.get("base_fill_prob", _EXECUTION_DEFAULTS.base_fill_prob)
        ),
        realism_high_cost_frac=float(
            execution.get(
                "realism_high_cost_frac",
                _EXECUTION_DEFAULTS.realism_high_cost_frac,
            )
        ),
        realism_low_cost_frac=float(
            execution.get(
                "realism_low_cost_frac",
                _EXECUTION_DEFAULTS.realism_low_cost_frac,
            )
        ),
        instrument_specs=specs or dict(_EXECUTION_DEFAULTS.instrument_specs),
    )


def management_horizon_from_mapping(cfg: Mapping[str, Any] | None) -> int:
    horizons = _section(cfg, "horizons").get("horizon_steps", [12])
    return int(max(horizons))


def strategy_profile_from_mapping(
    cfg: Mapping[str, Any] | None,
    override: str | None = None,
) -> StrategyProfile:
    if override:
        return get_strategy_profile(override)
    strategy_cfg = _section(cfg, "strategy")
    if isinstance(strategy_cfg, str):
        return get_strategy_profile(strategy_cfg)
    if isinstance(strategy_cfg, Mapping):
        name = strategy_cfg.get("name")
        return get_strategy_profile(str(name) if name is not None else None)
    return get_strategy_profile(None)


def risk_profile_name_from_mapping(
    cfg: Mapping[str, Any] | None,
    section: str | None = None,
) -> str:
    root = cfg if isinstance(cfg, Mapping) else {}
    if section:
        section_cfg = _section(cfg, section)
        if section_cfg.get("risk_profile"):
            return str(section_cfg["risk_profile"])
    if root.get("risk_profile"):
        return str(root["risk_profile"])
    return "balanced"


def walk_forward_from_mapping(cfg: Mapping[str, Any] | None) -> WalkForward:
    backtest_cfg = _section(cfg, "backtest")
    training_cfg = _section(cfg, "training")
    data_cfg = _section(cfg, "data")
    interval = str(data_cfg.get("base_interval", _DATA_DEFAULTS.base_interval))
    return WalkForward(
        n_folds=int(backtest_cfg.get("n_folds", training_cfg.get("n_folds", 5))),
        embargo_bars=int(
            backtest_cfg.get(
                "embargo_bars",
                training_cfg.get("embargo_bars", _TRAINING_DEFAULTS.embargo_bars),
            )
        ),
        bar_minutes=_interval_to_minutes(interval),
    )


def _section(cfg: Mapping[str, Any] | None, key: str) -> Mapping[str, Any]:
    if not isinstance(cfg, Mapping):
        return {}
    section = cfg.get(key, {})
    return section if isinstance(section, Mapping) else {}


def _interval_to_minutes(interval: str) -> int:
    if interval.endswith("min"):
        return int(interval.removesuffix("min"))
    if interval.endswith("m"):
        return int(interval.removesuffix("m"))
    return 5


__all__ = [
    "data_config_from_mapping",
    "execution_config_from_mapping",
    "loss_weights_from_mapping",
    "management_horizon_from_mapping",
    "model_config_from_mapping",
    "risk_profile_name_from_mapping",
    "strategy_profile_from_mapping",
    "training_config_from_mapping",
    "walk_forward_from_mapping",
]
