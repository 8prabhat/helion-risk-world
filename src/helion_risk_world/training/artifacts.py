"""Persisted training artifacts for promoted HRW model workflows.

The CLI scripts save a small metadata bundle alongside the ``state_dict`` so calibration,
backtesting, and prediction can reconstruct the exact model configuration without
train/serve drift. Both the compact forecaster and the multi-horizon world model use
the same artifact envelope.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from helion_risk_world.config.model_config import ModelConfig
from helion_risk_world.data.model_input_builder import ModelInputContract
from helion_risk_world.losses.quantile_loss import DEFAULT_QUANTILES
from helion_risk_world.model import HRWForecaster, HRWWorldModel
from helion_risk_world.prediction_calibration import PredictionCalibration

FORECASTER_ARTIFACT_KIND = "hrw_forecaster"
WORLD_MODEL_ARTIFACT_KIND = "hrw_world_model"
ARTIFACT_KIND = FORECASTER_ARTIFACT_KIND
ARTIFACT_VERSION = 13  # v13: Upstox-only provenance + persisted barrier horizon/cost contract.
# v12: feature/label overhaul Phases 2-3 dimension changes —
# candle plane 19->30 (trend/vol/session/cross-sectional/Kalman features), futures
# plane 13->14 (oi_basis_interaction), regime plane 20->22 (usdinr_vol/crude_vol).
# v11: ExcursionBarrierHead.linear grew from [3,3] to [3,4] (review M2)
# v10: HRWWorldModel gained the _ood_unfitted_scale buffer (review M1)
DEFAULT_TARGET_SYMBOL = "BANKNIFTY_FUT_continuous"


def save_forecaster_artifact(
    path: str | Path,
    model: HRWForecaster,
    *,
    n_features: int,
    horizon_bars: int,
    cfg: ModelConfig,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    input_contract: ModelInputContract | None = None,
    split_manifest: dict[str, Any] | None = None,
    data_capability_profile: dict[str, Any] | None = None,
    enabled_encoders: tuple[str, ...] = ("temporal", "cross_asset"),
    disabled_optional_modules: tuple[str, ...] = (),
    target_symbol: str = DEFAULT_TARGET_SYMBOL,
    model_selection: dict[str, Any] | None = None,
    prediction_calibration: dict[str, Any] | None = None,
    return_target_mode: str = "horizon",
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        _base_payload(
            kind=FORECASTER_ARTIFACT_KIND,
            model_kind="forecaster",
            state_dict=model.state_dict(),
            n_features=n_features,
            cfg=cfg,
            quantiles=quantiles,
            input_contract=input_contract,
            split_manifest=split_manifest,
            data_capability_profile=data_capability_profile,
            enabled_encoders=enabled_encoders,
            disabled_optional_modules=disabled_optional_modules,
            target_symbol=target_symbol,
            model_selection=model_selection,
            prediction_calibration=prediction_calibration,
            horizon_bars=horizon_bars,
            horizons=(horizon_bars,),
            barrier_mode=getattr(model, "barrier_mode", "derived"),
            return_target_mode=return_target_mode,
        ),
        target,
    )
    return target


def save_world_model_artifact(
    path: str | Path,
    model: HRWWorldModel,
    *,
    n_features: int,
    horizons: tuple[int, ...],
    cfg: ModelConfig,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    input_contract: ModelInputContract | None = None,
    split_manifest: dict[str, Any] | None = None,
    data_capability_profile: dict[str, Any] | None = None,
    enabled_encoders: tuple[str, ...] = ("temporal", "cross_asset"),
    disabled_optional_modules: tuple[str, ...] = (),
    n_samples: int | None = None,
    target_symbol: str = DEFAULT_TARGET_SYMBOL,
    model_selection: dict[str, Any] | None = None,
    prediction_calibration: dict[str, Any] | None = None,
    return_target_mode: str = "horizon",
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sorted_horizons = tuple(sorted(set(int(h) for h in horizons)))
    if not sorted_horizons:
        raise ValueError("world-model artifact requires at least one horizon")
    torch.save(
        _base_payload(
            kind=WORLD_MODEL_ARTIFACT_KIND,
            model_kind="world_model",
            state_dict=model.state_dict(),
            n_features=n_features,
            cfg=cfg,
            quantiles=quantiles,
            input_contract=input_contract,
            split_manifest=split_manifest,
            data_capability_profile=data_capability_profile,
            enabled_encoders=enabled_encoders,
            disabled_optional_modules=disabled_optional_modules,
            target_symbol=target_symbol,
            model_selection=model_selection,
            prediction_calibration=prediction_calibration,
            horizon_bars=max(sorted_horizons),
            horizons=sorted_horizons,
            n_samples=n_samples if n_samples is not None else int(cfg.rollout_samples),
            barrier_mode=getattr(model, "barrier_mode", "derived"),
            return_target_mode=return_target_mode,
        ),
        target,
    )
    return target


def load_forecaster_artifact(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[HRWForecaster, dict[str, Any]]:
    model, metadata = load_model_artifact(path, map_location=map_location)
    if not isinstance(model, HRWForecaster):
        raise ValueError(
            f"artifact {path} is {metadata['model_kind']!r}; expected a forecaster artifact"
        )
    return model, metadata


def load_world_model_artifact(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[HRWWorldModel, dict[str, Any]]:
    model, metadata = load_model_artifact(path, map_location=map_location)
    if not isinstance(model, HRWWorldModel):
        raise ValueError(
            f"artifact {path} is {metadata['model_kind']!r}; expected a world-model artifact"
        )
    return model, metadata


def load_model_artifact(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[HRWForecaster | HRWWorldModel, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    metadata = _coerce_metadata(payload)
    model_kind = str(metadata["model_kind"])
    if model_kind == "forecaster":
        model = HRWForecaster(
            n_features=int(metadata["n_features"]),
            cfg=ModelConfig(**metadata["model_config"]),
            n_quantiles=len(metadata["quantiles"]),
        )
    elif model_kind == "world_model":
        model = HRWWorldModel(
            n_features=int(metadata["n_features"]),
            cfg=ModelConfig(**metadata["model_config"]),
            horizons=tuple(int(h) for h in metadata["horizons"]),
            n_samples=int(
                metadata.get(
                    "n_samples",
                    metadata["model_config"].get("rollout_samples", 16),
                )
            ),
            n_quantiles=len(metadata["quantiles"]),
        )
    else:
        raise ValueError(f"unsupported model_kind: {model_kind!r}")
    strict = int(metadata.get("version", ARTIFACT_VERSION)) >= ARTIFACT_VERSION
    load_result = model.load_state_dict(metadata["state_dict"], strict=strict)
    if not strict and load_result.unexpected_keys:
        unexpected = ", ".join(sorted(load_result.unexpected_keys))
        raise ValueError(f"artifact contains unexpected parameters: {unexpected}")
    if hasattr(model, "set_barrier_mode"):
        model.set_barrier_mode(str(metadata.get("barrier_mode", "derived")))
    model.to(map_location)
    model.eval()
    return model, metadata


def _base_payload(
    *,
    kind: str,
    model_kind: str,
    state_dict: dict[str, Any],
    n_features: int,
    cfg: ModelConfig,
    quantiles: tuple[float, ...],
    input_contract: ModelInputContract | None,
    split_manifest: dict[str, Any] | None,
    data_capability_profile: dict[str, Any] | None,
    enabled_encoders: tuple[str, ...],
    disabled_optional_modules: tuple[str, ...],
    target_symbol: str,
    model_selection: dict[str, Any] | None,
    prediction_calibration: dict[str, Any] | None,
    horizon_bars: int,
    horizons: tuple[int, ...],
    barrier_mode: str,
    return_target_mode: str,
    n_samples: int | None = None,
) -> dict[str, Any]:
    payload = {
        "kind": kind,
        "version": ARTIFACT_VERSION,
        "model_kind": model_kind,
        "state_dict": state_dict,
        "n_features": int(n_features),
        "horizon_bars": int(horizon_bars),
        "horizons": list(int(h) for h in horizons),
        "assets_used": list(input_contract.universe) if input_contract is not None else [],
        "target_symbol": str(target_symbol),
        "model_config": asdict(cfg),
        "quantiles": list(quantiles),
        "input_contract": input_contract.to_metadata() if input_contract is not None else None,
        "enabled_encoders": list(enabled_encoders),
        "disabled_optional_modules": list(disabled_optional_modules),
        "split_manifest": split_manifest,
        "data_capability_profile": data_capability_profile,
        "model_selection": model_selection,
        "prediction_calibration": prediction_calibration,
        "barrier_mode": str(barrier_mode),
        "return_target_mode": str(return_target_mode),
    }
    if n_samples is not None:
        payload["n_samples"] = int(n_samples)
    return payload


def _coerce_metadata(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("model artifact must be a metadata dict")
    if "state_dict" not in payload:
        raise ValueError(
            "legacy raw state_dict artifacts are unsupported; retrain with the updated train.py"
        )
    metadata = dict(payload)
    kind = metadata.get("kind")
    if kind is None:
        kind = FORECASTER_ARTIFACT_KIND
        metadata["kind"] = kind
    if kind not in {FORECASTER_ARTIFACT_KIND, WORLD_MODEL_ARTIFACT_KIND}:
        raise ValueError(f"unsupported artifact kind: {kind!r}")
    metadata.setdefault("version", ARTIFACT_VERSION)
    metadata.setdefault(
        "model_kind",
        "world_model" if kind == WORLD_MODEL_ARTIFACT_KIND else "forecaster",
    )
    metadata.setdefault("quantiles", list(DEFAULT_QUANTILES))
    metadata.setdefault("input_contract", None)
    metadata.setdefault("enabled_encoders", ["temporal", "cross_asset"])
    metadata.setdefault("disabled_optional_modules", [])
    metadata.setdefault("split_manifest", None)
    metadata.setdefault("data_capability_profile", None)
    metadata.setdefault("model_selection", None)
    metadata.setdefault("prediction_calibration", None)
    metadata.setdefault("assets_used", [])
    metadata.setdefault("target_symbol", DEFAULT_TARGET_SYMBOL)
    metadata.setdefault(
        "barrier_mode",
        "legacy",
    )
    metadata.setdefault("return_target_mode", "horizon")
    calibration = PredictionCalibration.from_metadata(metadata.get("prediction_calibration"))
    metadata["prediction_calibration"] = (
        calibration.to_metadata() if calibration is not None else None
    )
    if "model_config" not in metadata:
        raise ValueError("artifact missing model_config metadata")
    if "n_features" not in metadata or "horizon_bars" not in metadata:
        raise ValueError("artifact missing n_features or horizon_bars metadata")
    metadata.setdefault("horizons", [int(metadata["horizon_bars"])])
    metadata["horizons"] = [int(h) for h in metadata["horizons"]]
    if str(metadata["model_kind"]) == "world_model":
        metadata.setdefault(
            "n_samples",
            int(metadata["model_config"].get("rollout_samples", 16)),
        )
    return metadata


__all__ = [
    "ARTIFACT_KIND",
    "ARTIFACT_VERSION",
    "DEFAULT_TARGET_SYMBOL",
    "FORECASTER_ARTIFACT_KIND",
    "WORLD_MODEL_ARTIFACT_KIND",
    "load_forecaster_artifact",
    "load_model_artifact",
    "load_world_model_artifact",
    "save_forecaster_artifact",
    "save_world_model_artifact",
]
