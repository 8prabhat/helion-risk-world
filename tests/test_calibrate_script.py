from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from helion_risk_world.schemas.prediction_schema import (
    BarrierProbabilities,
    HorizonPrediction,
    ModelPrediction,
)
from helion_risk_world.schemas.label_schema import (
    horizon_return_column,
    horizon_volatility_column,
)
from helion_risk_world.training.split_manifest import ChronoSplitManifest


_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
_SPEC = importlib.util.spec_from_file_location("calibrate_script", _ROOT / "scripts" / "calibrate.py")
assert _SPEC is not None and _SPEC.loader is not None
calibrate_script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(calibrate_script)


def test_run_calibration_uses_artifact_split_manifest(tmp_path, monkeypatch) -> None:
    index = pd.date_range("2026-01-01", periods=8, freq="D")
    labels = pd.DataFrame(
        {
            "barrier": ["timeout"] * len(index),
            "exit_return": [0.0] * len(index),
            "realized_vol": [0.0] * len(index),
            "regime": ["range"] * len(index),
            "label_schema_version": [5] * len(index),
        },
        index=index,
    )
    labels_path = tmp_path / "labels.parquet"
    labels.to_parquet(labels_path)
    manifest = ChronoSplitManifest.from_labels(labels, train_fraction=0.5, val_fraction=0.25)
    report_out = tmp_path / "calibration.json"

    runtime = SimpleNamespace(
        horizon_bars=12,
        available_horizons=(12,),
        quantiles=(0.1, 0.25, 0.5, 0.75, 0.9),
        split_manifest=manifest.to_metadata(),
        model_kind="forecaster",
        target_symbol="BANKNIFTY_FUT_continuous",
        enabled_encoders=("temporal",),
        disabled_optional_modules=(),
    )

    class _Source:
        def timestamps(self):
            return [ts.to_pydatetime() for ts in index]

    class _Inputs:
        def build(self, ts):
            return SimpleNamespace(ts=ts)

    def _predict(runtime, snapshot, *, symbol, ts):
        hp = HorizonPrediction(
            horizon_bars=12,
            return_quantiles={0.1: -0.01, 0.25: -0.005, 0.5: 0.0, 0.75: 0.005, 0.9: 0.01},
            volatility=0.01,
        )
        return ModelPrediction(
            symbol=symbol,
            ts=ts,
            horizon_preds=[hp],
            barrier=BarrierProbabilities(stop=0.2, target=0.2, timeout=0.6),
            mae=0.01,
            sigma_H=0.01,
            epistemic=0.0,
            aleatoric=0.01,
            ood_score=0.0,
        )

    monkeypatch.setattr(calibrate_script, "load_model_runtime", lambda path: runtime)
    monkeypatch.setattr(calibrate_script, "ParquetMarketDataSource", lambda **kwargs: _Source())
    monkeypatch.setattr(calibrate_script, "build_runtime_inputs", lambda *args, **kwargs: _Inputs())
    monkeypatch.setattr(calibrate_script, "predict_snapshot", _predict)

    calibrate_script.run_calibration(
        config_path=_ROOT / "configs" / "v1.yaml",
        model_path=tmp_path / "model.pt",
        labels_path=labels_path,
        data_dir=tmp_path,
        calibration_split="test",
        report_out=report_out,
        max_samples=0,
    )

    payload = json.loads(report_out.read_text(encoding="utf-8"))
    assert payload["evaluation_split"] == "test"
    assert payload["sample_counts"]["selected_rows"] == manifest.test_rows
    assert payload["sample_counts"]["usable_rows"] == manifest.test_rows


def test_run_calibration_uses_fixed_horizon_targets_for_multi_horizon_artifacts(
    tmp_path,
    monkeypatch,
) -> None:
    index = pd.date_range("2026-01-01", periods=8, freq="D")
    labels = pd.DataFrame(
        {
            "barrier": ["timeout"] * len(index),
            "barrier_valid": [True] * len(index),
            "exit_return": [0.5] * len(index),
            "realized_vol": [0.5] * len(index),
            "horizon_bars": [12] * len(index),
            horizon_return_column(3): [0.01] * len(index),
            horizon_volatility_column(3): [0.01] * len(index),
            horizon_return_column(12): [0.02] * len(index),
            horizon_volatility_column(12): [0.02] * len(index),
            "regime": ["range"] * len(index),
            "label_schema_version": [5] * len(index),
        },
        index=index,
    )
    labels_path = tmp_path / "labels.parquet"
    labels.to_parquet(labels_path)
    manifest = ChronoSplitManifest.from_labels(labels, train_fraction=0.5, val_fraction=0.25)
    report_out = tmp_path / "calibration_multi.json"

    runtime = SimpleNamespace(
        horizon_bars=12,
        available_horizons=(3, 12),
        quantiles=(0.1, 0.25, 0.5, 0.75, 0.9),
        split_manifest=manifest.to_metadata(),
        model_kind="world_model",
        target_symbol="BANKNIFTY_FUT_continuous",
        enabled_encoders=("temporal", "cross_asset"),
        disabled_optional_modules=(),
    )

    class _Source:
        def timestamps(self):
            return [ts.to_pydatetime() for ts in index]

    class _Inputs:
        def build(self, ts):
            return SimpleNamespace(ts=ts)

    def _predict(runtime, snapshot, *, symbol, ts):
        hp3 = HorizonPrediction(
            horizon_bars=3,
            return_quantiles={0.1: 0.0, 0.25: 0.005, 0.5: 0.01, 0.75: 0.015, 0.9: 0.02},
            volatility=0.01,
        )
        hp12 = HorizonPrediction(
            horizon_bars=12,
            return_quantiles={0.1: 0.01, 0.25: 0.015, 0.5: 0.02, 0.75: 0.025, 0.9: 0.03},
            volatility=0.02,
        )
        return ModelPrediction(
            symbol=symbol,
            ts=ts,
            horizon_preds=[hp3, hp12],
            barrier=BarrierProbabilities(stop=0.0, target=0.0, timeout=1.0),
            mae=0.01,
            sigma_H=0.02,
            epistemic=0.0,
            aleatoric=0.02,
            ood_score=0.0,
        )

    monkeypatch.setattr(calibrate_script, "load_model_runtime", lambda path: runtime)
    monkeypatch.setattr(calibrate_script, "ParquetMarketDataSource", lambda **kwargs: _Source())
    monkeypatch.setattr(calibrate_script, "build_runtime_inputs", lambda *args, **kwargs: _Inputs())
    monkeypatch.setattr(calibrate_script, "predict_snapshot", _predict)

    calibrate_script.run_calibration(
        config_path=_ROOT / "configs" / "v1.yaml",
        model_path=tmp_path / "world_model.pt",
        labels_path=labels_path,
        data_dir=tmp_path,
        calibration_split="test",
        report_out=report_out,
        max_samples=0,
    )

    payload = json.loads(report_out.read_text(encoding="utf-8"))
    assert payload["gate"]["evaluated_horizons"] == [3, 12]
    assert set(payload["per_horizon"]) == {"3", "12"}
    assert payload["sample_counts"]["per_horizon_rows"]["12"] == manifest.test_rows
    assert payload["metrics"]["rollout_mae"] == pytest.approx(0.0, abs=1e-9)


def test_run_calibration_prefers_bulk_input_precompute_when_available(
    tmp_path,
    monkeypatch,
) -> None:
    index = pd.date_range("2026-01-01", periods=4, freq="D")
    labels = pd.DataFrame(
        {
            "barrier": ["timeout"] * len(index),
            "barrier_valid": [True] * len(index),
            "exit_return": [0.0] * len(index),
            "realized_vol": [0.01] * len(index),
            "horizon_bars": [12] * len(index),
            horizon_return_column(12): [0.0] * len(index),
            horizon_volatility_column(12): [0.01] * len(index),
            "regime": ["range"] * len(index),
            "label_schema_version": [5] * len(index),
        },
        index=index,
    )
    labels_path = tmp_path / "labels_bulk.parquet"
    labels.to_parquet(labels_path)
    report_out = tmp_path / "calibration_bulk.json"

    runtime = SimpleNamespace(
        horizon_bars=12,
        available_horizons=(12,),
        quantiles=(0.1, 0.25, 0.5, 0.75, 0.9),
        split_manifest=None,
        model_kind="forecaster",
        target_symbol="BANKNIFTY_FUT_continuous",
        enabled_encoders=("temporal",),
        disabled_optional_modules=(),
    )

    class _Source:
        def timestamps(self):
            return [ts.to_pydatetime() for ts in index]

    class _Inputs:
        def __init__(self) -> None:
            self.build_calls = 0
            self.build_many_calls = 0

        def build_many(self, timestamps):
            self.build_many_calls += 1
            return {ts: SimpleNamespace(ts=ts) for ts in timestamps}

        def build(self, ts):
            self.build_calls += 1
            raise AssertionError("build() should not be called when build_many() covers all rows")

    inputs = _Inputs()

    def _predict(runtime, snapshot, *, symbol, ts):
        hp = HorizonPrediction(
            horizon_bars=12,
            return_quantiles={0.1: -0.01, 0.25: -0.005, 0.5: 0.0, 0.75: 0.005, 0.9: 0.01},
            volatility=0.01,
        )
        return ModelPrediction(
            symbol=symbol,
            ts=ts,
            horizon_preds=[hp],
            barrier=BarrierProbabilities(stop=0.0, target=0.0, timeout=1.0),
            mae=0.01,
            sigma_H=0.01,
            epistemic=0.0,
            aleatoric=0.01,
            ood_score=0.0,
        )

    monkeypatch.setattr(calibrate_script, "load_model_runtime", lambda path: runtime)
    monkeypatch.setattr(calibrate_script, "ParquetMarketDataSource", lambda **kwargs: _Source())
    monkeypatch.setattr(calibrate_script, "build_runtime_inputs", lambda *args, **kwargs: inputs)
    monkeypatch.setattr(calibrate_script, "predict_snapshot", _predict)

    calibrate_script.run_calibration(
        config_path=_ROOT / "configs" / "v1.yaml",
        model_path=tmp_path / "model.pt",
        labels_path=labels_path,
        data_dir=tmp_path,
        calibration_split="all",
        report_out=report_out,
        max_samples=0,
    )

    assert inputs.build_many_calls == 1
    assert inputs.build_calls == 0


def test_run_calibration_skips_ambiguous_barriers_from_barrier_metrics(
    tmp_path,
    monkeypatch,
) -> None:
    index = pd.date_range("2026-01-01", periods=6, freq="D")
    labels = pd.DataFrame(
        {
            "barrier": ["timeout", "ambiguous", "timeout", "ambiguous", "timeout", "timeout"],
            "barrier_valid": [True, False, True, False, True, True],
            "exit_return": [0.0] * len(index),
            "realized_vol": [0.01] * len(index),
            "horizon_bars": [12] * len(index),
            horizon_return_column(12): [0.0] * len(index),
            horizon_volatility_column(12): [0.01] * len(index),
            "regime": ["range"] * len(index),
            "label_schema_version": [5] * len(index),
        },
        index=index,
    )
    labels_path = tmp_path / "labels_ambiguous.parquet"
    labels.to_parquet(labels_path)
    report_out = tmp_path / "calibration_ambiguous.json"

    runtime = SimpleNamespace(
        horizon_bars=12,
        available_horizons=(12,),
        quantiles=(0.1, 0.25, 0.5, 0.75, 0.9),
        split_manifest=None,
        model_kind="forecaster",
        target_symbol="BANKNIFTY_FUT_continuous",
        enabled_encoders=("temporal",),
        disabled_optional_modules=(),
    )

    class _Source:
        def timestamps(self):
            return [ts.to_pydatetime() for ts in index]

    class _Inputs:
        def build(self, ts):
            return SimpleNamespace(ts=ts)

    def _predict(runtime, snapshot, *, symbol, ts):
        hp = HorizonPrediction(
            horizon_bars=12,
            return_quantiles={0.1: -0.01, 0.25: -0.005, 0.5: 0.0, 0.75: 0.005, 0.9: 0.01},
            volatility=0.01,
        )
        return ModelPrediction(
            symbol=symbol,
            ts=ts,
            horizon_preds=[hp],
            barrier=BarrierProbabilities(stop=0.0, target=0.0, timeout=1.0),
            mae=0.01,
            sigma_H=0.01,
            epistemic=0.0,
            aleatoric=0.01,
            ood_score=0.0,
        )

    monkeypatch.setattr(calibrate_script, "load_model_runtime", lambda path: runtime)
    monkeypatch.setattr(calibrate_script, "ParquetMarketDataSource", lambda **kwargs: _Source())
    monkeypatch.setattr(calibrate_script, "build_runtime_inputs", lambda *args, **kwargs: _Inputs())
    monkeypatch.setattr(calibrate_script, "predict_snapshot", _predict)

    calibrate_script.run_calibration(
        config_path=_ROOT / "configs" / "v1.yaml",
        model_path=tmp_path / "model.pt",
        labels_path=labels_path,
        data_dir=tmp_path,
        calibration_split="all",
        report_out=report_out,
        max_samples=0,
    )

    payload = json.loads(report_out.read_text(encoding="utf-8"))
    assert payload["sample_counts"]["usable_rows"] == len(index)
    assert payload["sample_counts"]["barrier_usable_rows"] == 4
    assert payload["metrics"]["barrier_brier"] == pytest.approx(0.0, abs=1e-9)
    assert payload["barrier_evaluation"]["subset_only"] is True
