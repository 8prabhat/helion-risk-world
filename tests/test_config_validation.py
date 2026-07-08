"""Config validation tests (SPEC.md §27)."""
from __future__ import annotations

import pytest

from helion_risk_world.config import DataConfig, ModelConfig, PlannerConfig, TrainingConfig


def test_v1_forbids_historical_depth() -> None:
    with pytest.raises(ValueError):
        DataConfig(use_historical_depth=True)


def test_model_size_presets() -> None:
    m = ModelConfig.from_size("medium")
    assert m.latent_dim == 256 and m.temporal_layers == 4


def test_planner_sizes_must_start_at_zero() -> None:
    with pytest.raises(ValueError):
        PlannerConfig(sizes=(0.1, 0.5, 1.0))


def test_embargo_must_cover_horizon() -> None:
    with pytest.raises(ValueError):
        TrainingConfig(embargo_bars=0)


def test_pretraining_config_requires_non_negative_epochs_and_positive_gap() -> None:
    with pytest.raises(ValueError):
        TrainingConfig(pretrain_epochs=-1)
    with pytest.raises(ValueError):
        TrainingConfig(pretrain_gap_bars=0)


def test_pretrain_gap_bars_default_is_not_near_identity() -> None:
    """Review finding H6: the old default of 1 (vs lookback_bars=96) made Stage-2
    future-latent prediction a near-identity task. Default should be a meaningful
    fraction of the lookback (matching the management horizon, 12)."""
    assert TrainingConfig().pretrain_gap_bars == 12
