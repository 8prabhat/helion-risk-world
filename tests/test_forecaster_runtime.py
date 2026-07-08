"""Shared forecaster runtime helper tests."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from helion_risk_world.barrier_context import BarrierContext  # noqa: E402
from helion_risk_world.config.data_config import DataConfig  # noqa: E402
from helion_risk_world.config.model_config import ModelConfig  # noqa: E402
from helion_risk_world.data.feature_builder import MarketBatch  # noqa: E402
from helion_risk_world.data.market_window_builder import CANDLE_FEATURE_NAMES  # noqa: E402
from helion_risk_world.data.model_input_builder import (  # noqa: E402
    ModelInputContract,
    ModelInputSnapshot,
)
from helion_risk_world.model import HRWForecaster, HRWWorldModel  # noqa: E402
from helion_risk_world.prediction_calibration import (  # noqa: E402
    HorizonPredictionCalibration,
    PredictionCalibration,
)
from helion_risk_world.runtime import (  # noqa: E402
    load_forecaster_runtime,
    load_model_runtime,
    predict_snapshot,
)
from helion_risk_world.training.artifacts import (  # noqa: E402
    save_forecaster_artifact,
    save_world_model_artifact,
)

TS = datetime(2026, 6, 29, 10, 0)


def _save_runtime_artifact(tmp_path, *, with_contract: bool) -> tuple[object, int]:
    n_features = len(CANDLE_FEATURE_NAMES)
    model = HRWForecaster(
        n_features=n_features,
        cfg=ModelConfig(latent_dim=16, temporal_layers=1, dropout=0.0),
    )
    path = tmp_path / "forecaster.pt"
    contract = (
        ModelInputContract.from_data_config(
            DataConfig(universe=("BANKNIFTY",), lookback_bars=12),
            feature_names=CANDLE_FEATURE_NAMES,
        )
        if with_contract
        else None
    )
    save_forecaster_artifact(
        path,
        model,
        n_features=n_features,
        horizon_bars=12,
        cfg=ModelConfig(latent_dim=16, temporal_layers=1, dropout=0.0),
        input_contract=contract,
    )
    return path, n_features


def test_load_forecaster_runtime_and_predict_snapshot(tmp_path) -> None:
    path, n_features = _save_runtime_artifact(tmp_path, with_contract=True)

    runtime = load_forecaster_runtime(path)
    snapshot = ModelInputSnapshot(
        market=MarketBatch(
            ts=TS,
            symbols=("BANKNIFTY",),
            candle_features=np.random.randn(1, 12, n_features).astype(np.float32),
            feature_names=CANDLE_FEATURE_NAMES,
        )
    )
    pred = predict_snapshot(runtime, snapshot, symbol="BANKNIFTY", ts=TS)

    assert runtime.horizon_bars == 12
    assert runtime.contract.lookback_bars == 12
    assert runtime.quantiles == (0.1, 0.25, 0.5, 0.75, 0.9)
    assert runtime.model_kind == "forecaster"
    assert pred.symbol == "BANKNIFTY"
    assert pred.horizon_preds[0].horizon_bars == 12


def test_predict_snapshot_preserves_explicit_barrier_context(tmp_path) -> None:
    path, n_features = _save_runtime_artifact(tmp_path, with_contract=True)

    runtime = load_forecaster_runtime(path)
    snapshot = ModelInputSnapshot(
        market=MarketBatch(
            ts=TS,
            symbols=("BANKNIFTY",),
            candle_features=np.random.randn(1, 12, n_features).astype(np.float32),
            feature_names=CANDLE_FEATURE_NAMES,
        ),
        barrier_context=BarrierContext(sigma=0.01, stop_return=-0.02, target_return=0.03),
    )
    pred = predict_snapshot(runtime, snapshot, symbol="BANKNIFTY", ts=TS)

    assert pred.stop_return == pytest.approx(-0.02)
    assert pred.target_return == pytest.approx(0.03)


def test_load_forecaster_runtime_requires_input_contract(tmp_path) -> None:
    path, _ = _save_runtime_artifact(tmp_path, with_contract=False)

    with pytest.raises(ValueError, match="runtime input contract"):
        load_forecaster_runtime(path)


def test_load_model_runtime_world_model_supports_multi_horizon(tmp_path) -> None:
    n_features = len(CANDLE_FEATURE_NAMES)
    model = HRWWorldModel(
        n_features=n_features,
        cfg=ModelConfig(latent_dim=16, temporal_layers=1, dropout=0.0, rollout_samples=4),
        horizons=(3, 6, 12),
        n_samples=4,
    )
    path = tmp_path / "world_model.pt"
    contract = ModelInputContract.from_data_config(
        DataConfig(universe=("BANKNIFTY",), lookback_bars=12),
        feature_names=CANDLE_FEATURE_NAMES,
    )
    save_world_model_artifact(
        path,
        model,
        n_features=n_features,
        horizons=(3, 6, 12),
        cfg=ModelConfig(latent_dim=16, temporal_layers=1, dropout=0.0, rollout_samples=4),
        input_contract=contract,
    )
    runtime = load_model_runtime(path)
    snapshot = ModelInputSnapshot(
        market=MarketBatch(
            ts=TS,
            symbols=("BANKNIFTY",),
            candle_features=np.random.randn(1, 12, n_features).astype(np.float32),
            feature_names=CANDLE_FEATURE_NAMES,
        )
    )
    pred = predict_snapshot(runtime, snapshot, ts=TS)

    assert runtime.model_kind == "world_model"
    assert runtime.available_horizons == (3, 6, 12)
    assert runtime.target_symbol == "BANKNIFTY_FUT_continuous"
    assert [hp.horizon_bars for hp in pred.horizon_preds] == [3, 6, 12]
    assert pred.symbol == "BANKNIFTY_FUT_continuous"


def test_runtime_applies_prediction_calibration_from_artifact(tmp_path) -> None:
    n_features = len(CANDLE_FEATURE_NAMES)
    model = HRWForecaster(
        n_features=n_features,
        cfg=ModelConfig(latent_dim=16, temporal_layers=1, dropout=0.0),
    )
    path = tmp_path / "forecaster_calibrated.pt"
    contract = ModelInputContract.from_data_config(
        DataConfig(universe=("BANKNIFTY",), lookback_bars=12),
        feature_names=CANDLE_FEATURE_NAMES,
    )
    calibration = PredictionCalibration(
        quantile_levels=(0.1, 0.25, 0.5, 0.75, 0.9),
        horizons={
            12: HorizonPredictionCalibration(
                horizon_bars=12,
                quantile_offsets={0.1: -0.1, 0.25: -0.05, 0.5: 0.2, 0.75: 0.25, 0.9: 0.3},
                volatility_scale=2.0,
                volatility_bias=0.01,
                sample_count=16,
            )
        },
        barrier_temperature=2.0,
        source="unit_test",
        sample_count=16,
    )
    save_forecaster_artifact(
        path,
        model,
        n_features=n_features,
        horizon_bars=12,
        cfg=ModelConfig(latent_dim=16, temporal_layers=1, dropout=0.0),
        input_contract=contract,
        prediction_calibration=calibration.to_metadata(),
    )

    runtime = load_model_runtime(path)
    snapshot = ModelInputSnapshot(
        market=MarketBatch(
            ts=TS,
            symbols=("BANKNIFTY",),
            candle_features=np.random.randn(1, 12, n_features).astype(np.float32),
            feature_names=CANDLE_FEATURE_NAMES,
        )
    )
    raw = runtime.predictor.predict_one(
        torch.tensor(snapshot.market.candle_features, dtype=torch.float32),
        "BANKNIFTY",
        TS,
    )
    calibrated = predict_snapshot(runtime, snapshot, symbol="BANKNIFTY", ts=TS)

    assert runtime.prediction_calibration is not None
    assert calibrated.horizon_preds[0].return_quantiles[0.5] > raw.horizon_preds[0].return_quantiles[0.5]
    assert calibrated.sigma_H > raw.sigma_H
    assert max(calibrated.barrier.stop, calibrated.barrier.target, calibrated.barrier.timeout) < max(
        raw.barrier.stop,
        raw.barrier.target,
        raw.barrier.timeout,
    )


def test_load_model_runtime_supports_pre_barrier_context_artifacts(tmp_path) -> None:
    n_features = len(CANDLE_FEATURE_NAMES)
    model = HRWForecaster(
        n_features=n_features,
        cfg=ModelConfig(latent_dim=16, temporal_layers=1, dropout=0.0),
    )
    path = tmp_path / "legacy_barrier_artifact.pt"
    contract = ModelInputContract.from_data_config(
        DataConfig(universe=("BANKNIFTY",), lookback_bars=12),
        feature_names=CANDLE_FEATURE_NAMES,
    )
    save_forecaster_artifact(
        path,
        model,
        n_features=n_features,
        horizon_bars=12,
        cfg=ModelConfig(latent_dim=16, temporal_layers=1, dropout=0.0),
        input_contract=contract,
    )

    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["version"] = 5
    for key in list(payload["state_dict"]):
        if (
            key.startswith("barrier_head.context_gate")
            or key.startswith("barrier_head.input_norm")
            or key.startswith("mae_head.")
            or key.startswith("mfe_head.")
        ):
            del payload["state_dict"][key]
    torch.save(payload, path)

    runtime = load_model_runtime(path)
    snapshot = ModelInputSnapshot(
        market=MarketBatch(
            ts=TS,
            symbols=("BANKNIFTY",),
            candle_features=np.random.randn(1, 12, n_features).astype(np.float32),
            feature_names=CANDLE_FEATURE_NAMES,
        )
    )
    pred = predict_snapshot(runtime, snapshot, symbol="BANKNIFTY", ts=TS)

    assert pred.symbol == "BANKNIFTY"
    assert pred.horizon_preds[0].horizon_bars == 12
    hp = pred.horizon_preds[0]
    derived_mae = abs(hp.return_quantiles[0.5] - hp.return_quantiles[0.1])
    assert pred.mae == pytest.approx(derived_mae, abs=1e-6)
