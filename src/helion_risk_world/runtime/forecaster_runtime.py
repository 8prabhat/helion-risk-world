"""Shared artifact/runtime helpers for inference-facing flows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch

from helion_risk_world.config.data_config import DataConfig
from helion_risk_world.data.feature_builder import MarketDataSource
from helion_risk_world.data.model_input_builder import (
    ModelInputBuilder,
    ModelInputContract,
    ModelInputSnapshot,
)
from helion_risk_world.inference import ForecasterPredictor, WorldModelPredictor
from helion_risk_world.model import HRWForecaster, HRWWorldModel
from helion_risk_world.prediction_calibration import PredictionCalibration
from helion_risk_world.schemas.prediction_schema import ModelPrediction
from helion_risk_world.training.artifacts import load_model_artifact


@dataclass(frozen=True)
class ModelRuntime:
    """Loaded model artifact plus its validated runtime contract."""

    model: HRWForecaster | HRWWorldModel
    predictor: ForecasterPredictor | WorldModelPredictor
    metadata: dict[str, object]
    contract: ModelInputContract
    horizon_bars: int
    available_horizons: tuple[int, ...]
    quantiles: tuple[float, ...]
    model_kind: str
    target_symbol: str
    assets_used: tuple[str, ...]
    enabled_encoders: tuple[str, ...]
    disabled_optional_modules: tuple[str, ...]
    split_manifest: dict[str, object] | None
    data_capability_profile: dict[str, object] | None
    prediction_calibration: PredictionCalibration | None


ForecasterRuntime = ModelRuntime


def load_model_runtime(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    persist_state: bool = True,
) -> ModelRuntime:
    """Load a model artifact into a runnable ``ModelRuntime``.

    ``persist_state`` (review finding H1): for a ``model_kind='world_model'`` artifact,
    controls whether the returned ``WorldModelPredictor`` carries its RSSM belief state
    across successive ``predict_one`` calls (the correct behavior for a live/paper/backtest
    replay loop over ascending timestamps) or resets it every call (the pre-fix behavior,
    useful for A/B comparison). Ignored for ``model_kind='forecaster'`` artifacts, which
    have no RSSM state. Defaults to True.
    """
    model, metadata = load_model_artifact(path, map_location=map_location)
    contract = ModelInputContract.from_metadata(metadata)
    if contract is None:
        raise ValueError("artifact predates the runtime input contract; retrain with scripts/train.py")
    quantiles = tuple(float(q) for q in metadata["quantiles"])
    available_horizons = tuple(
        int(h) for h in metadata.get("horizons", [metadata["horizon_bars"]])
    )
    model_kind = str(metadata.get("model_kind", "forecaster"))
    version = int(metadata.get("version", 0))
    use_predicted_mae = version >= 6
    use_barrier_geometry = version >= 7
    prediction_calibration = PredictionCalibration.from_metadata(
        metadata.get("prediction_calibration")
    )
    predictor: ForecasterPredictor | WorldModelPredictor
    if model_kind == "world_model":
        if not isinstance(model, HRWWorldModel):
            raise ValueError("world-model artifact resolved to a non-world-model instance")
        predictor = WorldModelPredictor(
            model,
            quantiles=quantiles,
            use_predicted_mae=use_predicted_mae,
            use_barrier_geometry=use_barrier_geometry,
            persist_state=persist_state,
        )
    else:
        if not isinstance(model, HRWForecaster):
            raise ValueError("forecaster artifact resolved to a non-forecaster instance")
        predictor = ForecasterPredictor(
            model,
            quantiles=quantiles,
            horizon_bars=int(metadata["horizon_bars"]),
            use_predicted_mae=use_predicted_mae,
            use_barrier_geometry=use_barrier_geometry,
        )
    return ModelRuntime(
        model=model,
        predictor=predictor,
        metadata=metadata,
        contract=contract,
        horizon_bars=int(metadata["horizon_bars"]),
        available_horizons=available_horizons,
        quantiles=quantiles,
        model_kind=model_kind,
        target_symbol=str(metadata.get("target_symbol", "BANKNIFTY_FUT_continuous")),
        assets_used=tuple(metadata.get("assets_used", [])),
        enabled_encoders=tuple(metadata.get("enabled_encoders", [])),
        disabled_optional_modules=tuple(metadata.get("disabled_optional_modules", [])),
        split_manifest=metadata.get("split_manifest"),
        data_capability_profile=metadata.get("data_capability_profile"),
        prediction_calibration=prediction_calibration,
    )


def load_forecaster_runtime(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> ForecasterRuntime:
    runtime = load_model_runtime(path, map_location=map_location)
    if runtime.model_kind != "forecaster":
        raise ValueError(
            f"artifact {path} is {runtime.model_kind!r}; use load_model_runtime for generic loading"
        )
    return runtime


def build_runtime_inputs(
    cfg: DataConfig,
    source: MarketDataSource,
    *,
    data_dir: str | Path | None,
    runtime: ModelRuntime,
) -> ModelInputBuilder:
    return ModelInputBuilder.from_data_dir(
        cfg,
        source,
        data_dir=data_dir,
        contract=runtime.contract,
    )


def predict_snapshot(
    runtime: ModelRuntime,
    snapshot: ModelInputSnapshot,
    *,
    symbol: str | None = None,
    ts: datetime,
) -> ModelPrediction:
    futures = (
        torch.tensor(snapshot.market.futures, dtype=torch.float32)
        if snapshot.market.futures is not None
        else None
    )
    prediction = runtime.predictor.predict_one(
        torch.tensor(snapshot.market.candle_features, dtype=torch.float32),
        symbol or runtime.target_symbol,
        ts,
        futures=futures,
        regime=snapshot.regime,
        barrier_context=snapshot.barrier_context,
        surface=snapshot.market.surface,
    )
    if runtime.prediction_calibration is not None:
        return runtime.prediction_calibration.apply(prediction)
    return prediction


__all__ = [
    "ForecasterRuntime",
    "ModelRuntime",
    "build_runtime_inputs",
    "load_forecaster_runtime",
    "load_model_runtime",
    "predict_snapshot",
]
