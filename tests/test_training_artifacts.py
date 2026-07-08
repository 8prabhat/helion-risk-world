"""Model artifact round-trip tests."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from helion_risk_world.config.model_config import ModelConfig  # noqa: E402
from helion_risk_world.config.data_config import DataConfig  # noqa: E402
from helion_risk_world.data.market_window_builder import CANDLE_FEATURE_NAMES  # noqa: E402
from helion_risk_world.data.model_input_builder import ModelInputContract  # noqa: E402
from helion_risk_world.model import HRWForecaster, HRWWorldModel  # noqa: E402
from helion_risk_world.training.artifacts import (  # noqa: E402
    load_forecaster_artifact,
    load_world_model_artifact,
    save_forecaster_artifact,
    save_world_model_artifact,
)


def test_forecaster_artifact_round_trip(tmp_path) -> None:
    n_features = len(CANDLE_FEATURE_NAMES)
    model = HRWForecaster(n_features=n_features, cfg=ModelConfig(latent_dim=16, temporal_layers=1, dropout=0.0))
    path = tmp_path / "forecaster.pt"
    contract = ModelInputContract.from_data_config(
        DataConfig(universe=("BANKNIFTY",), lookback_bars=12),
        feature_names=CANDLE_FEATURE_NAMES,
        uses_regime_context=True,
    )
    save_forecaster_artifact(
        path,
        model,
        n_features=n_features,
        horizon_bars=12,
        cfg=ModelConfig(latent_dim=16, temporal_layers=1, dropout=0.0),
        input_contract=contract,
    )
    loaded, meta = load_forecaster_artifact(path)
    assert meta["horizon_bars"] == 12
    assert meta["n_features"] == n_features
    assert meta["model_kind"] == "forecaster"
    assert meta["input_contract"]["uses_regime_context"] is True
    assert meta["return_target_mode"] == "horizon"
    out = loaded(torch.randn(2, 1, 12, n_features))
    assert out["return_quantiles"].shape == (2, 5)


def test_world_model_artifact_round_trip(tmp_path) -> None:
    n_features = len(CANDLE_FEATURE_NAMES)
    model = HRWWorldModel(
        n_features=n_features,
        cfg=ModelConfig(latent_dim=16, temporal_layers=1, dropout=0.0),
        horizons=(3, 6, 12),
        n_samples=4,
    )
    path = tmp_path / "world_model.pt"
    contract = ModelInputContract.from_data_config(
        DataConfig(universe=("BANKNIFTY",), lookback_bars=12),
        feature_names=CANDLE_FEATURE_NAMES,
        uses_regime_context=True,
    )
    save_world_model_artifact(
        path,
        model,
        n_features=n_features,
        horizons=(3, 6, 12),
        cfg=ModelConfig(latent_dim=16, temporal_layers=1, dropout=0.0, rollout_samples=4),
        input_contract=contract,
    )
    loaded, meta = load_world_model_artifact(path)
    assert meta["horizon_bars"] == 12
    assert meta["horizons"] == [3, 6, 12]
    assert meta["model_kind"] == "world_model"
    assert meta["target_symbol"] == "BANKNIFTY_FUT_continuous"
    assert meta["return_target_mode"] == "horizon"
    out = loaded(torch.randn(2, 1, 12, n_features))
    assert out["return_quantiles"].shape == (2, 3, 5)


def test_artifact_round_trip_preserves_custom_return_target_mode(tmp_path) -> None:
    n_features = len(CANDLE_FEATURE_NAMES)
    model = HRWForecaster(
        n_features=n_features,
        cfg=ModelConfig(latent_dim=16, temporal_layers=1, dropout=0.0),
    )
    path = tmp_path / "forecaster_timeout.pt"
    save_forecaster_artifact(
        path,
        model,
        n_features=n_features,
        horizon_bars=12,
        cfg=ModelConfig(latent_dim=16, temporal_layers=1, dropout=0.0),
        return_target_mode="timeout",
    )

    _, meta = load_forecaster_artifact(path)
    assert meta["return_target_mode"] == "timeout"
