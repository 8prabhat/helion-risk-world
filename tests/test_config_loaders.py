"""Typed config loader tests."""

from __future__ import annotations

from helion_risk_world.config import (
    data_config_from_mapping,
    execution_config_from_mapping,
    loss_weights_from_mapping,
    management_horizon_from_mapping,
    model_config_from_mapping,
    risk_profile_name_from_mapping,
    strategy_profile_from_mapping,
    training_config_from_mapping,
    walk_forward_from_mapping,
)


def test_data_and_model_config_loaders_apply_overrides() -> None:
    cfg = {
        "data": {
            "universe": ["BANKNIFTY", "NIFTY"],
            "base_interval": "15min",
            "lookback_bars": 48,
            "n_strikes": 7,
            "feature_cache_dir": "cache",
        },
        "model": {
            "latent_dim": 256,
            "temporal_layers": 4,
            "dropout": 0.2,
        },
    }

    data_cfg = data_config_from_mapping(cfg)
    model_cfg = model_config_from_mapping(cfg)

    assert data_cfg.universe == ("BANKNIFTY", "NIFTY")
    assert data_cfg.base_interval == "15min"
    assert data_cfg.lookback_bars == 48
    assert data_cfg.n_strikes == 7
    assert data_cfg.feature_cache_dir == "cache"
    assert model_cfg.latent_dim == 256
    assert model_cfg.temporal_layers == 4
    assert model_cfg.dropout == 0.2


def test_training_and_execution_config_loaders_apply_defaults_and_overrides() -> None:
    cfg = {
        "seed": 19,
        "training": {
            "batch_size": 64,
            "pretrain_epochs": 3,
        },
        "execution": {
            "brokerage_per_order": 12.5,
            "base_fill_prob": 0.9,
            "instrument_specs": {
                "banknifty_fut": {
                    "lot_size": 30,
                    "tick_size": 0.2,
                    "margin_fraction": 0.25,
                }
            },
        },
    }

    training_cfg = training_config_from_mapping(cfg, embargo_bars=9)
    execution_cfg = execution_config_from_mapping(cfg)

    assert training_cfg.seed == 19
    assert training_cfg.batch_size == 64
    assert training_cfg.pretrain_epochs == 3
    assert training_cfg.embargo_bars == 9
    assert execution_cfg.brokerage_per_order == 12.5
    assert execution_cfg.base_fill_prob == 0.9
    assert execution_cfg.tick_size == 0.05
    assert execution_cfg.instrument_specs["BANKNIFTY_FUT"].lot_size == 30.0
    assert execution_cfg.instrument_specs["BANKNIFTY_FUT"].tick_size == 0.2
    assert execution_cfg.instrument_specs["BANKNIFTY_FUT"].margin_fraction == 0.25


def test_loss_weights_loader_honors_loss_section() -> None:
    cfg = {
        "loss": {
            "weights": {
                "return": 1.7,
                "volatility": 0.9,
                "uncertainty": 0.05,
            }
        }
    }

    weights = loss_weights_from_mapping(cfg)

    assert weights.return_ == 1.7
    assert weights.volatility == 0.9
    assert weights.uncertainty == 0.05
    assert weights.barrier == 0.5


def test_management_horizon_and_walk_forward_loaders_respect_config() -> None:
    cfg = {
        "data": {"base_interval": "15min"},
        "horizons": {"horizon_steps": [3, 6, 18]},
        "training": {"n_folds": 4, "embargo_bars": 11},
        "backtest": {"n_folds": 2},
    }

    walk_forward = walk_forward_from_mapping(cfg)

    assert management_horizon_from_mapping(cfg) == 18
    assert walk_forward._n_folds == 2
    assert walk_forward._embargo_bars == 11
    assert walk_forward._bar_minutes == 15


def test_strategy_and_risk_profile_loaders_honor_overrides() -> None:
    cfg = {
        "strategy": {"name": "scalping"},
        "risk_profile": "balanced",
        "paper": {"risk_profile": "aggressive"},
    }

    strategy = strategy_profile_from_mapping(cfg)

    assert strategy.name.value == "scalping"
    assert risk_profile_name_from_mapping(cfg) == "balanced"
    assert risk_profile_name_from_mapping(cfg, "paper") == "aggressive"


def test_strategy_package_import_is_not_circular() -> None:
    from helion_risk_world.strategy import StrategyPlanner, get_strategy_profile

    assert StrategyPlanner is not None
    assert get_strategy_profile("low_frequency").decision_horizon_bars == 12
