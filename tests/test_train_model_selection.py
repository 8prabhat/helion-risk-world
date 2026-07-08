from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from helion_risk_world.config.data_config import DataConfig  # noqa: E402
from helion_risk_world.config.model_config import LossWeights, ModelConfig  # noqa: E402
from helion_risk_world.config.training_config import TrainingConfig  # noqa: E402
from helion_risk_world.data.market_window_builder import CANDLE_FEATURE_NAMES  # noqa: E402
from helion_risk_world.training.split_manifest import ChronoSplitManifest  # noqa: E402
from helion_risk_world.training.trainer import ForecastBatch  # noqa: E402


_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
_SPEC = importlib.util.spec_from_file_location("train_script", _ROOT / "scripts" / "train.py")
assert _SPEC is not None and _SPEC.loader is not None
train_script = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = train_script
_SPEC.loader.exec_module(train_script)


def test_walk_forward_model_selection_uses_fold_median_epoch(monkeypatch) -> None:
    index = pd.date_range("2026-01-01 09:15:00", periods=36, freq="5min")
    labels = pd.DataFrame(
        {
            "label_realized_at": index + pd.Timedelta(minutes=5),
            "label_schema_version": [5] * len(index),
        },
        index=index,
    )
    manifest = ChronoSplitManifest.from_labels(
        labels,
        train_fraction=0.6,
        val_fraction=0.2,
        embargo_bars=1,
        bar_interval="5min",
    )

    def _fake_batches(*args, **kwargs):
        n = len(args[4])
        return [
            ForecastBatch(
                features=torch.zeros(max(n, 1), 1, 2, 2),
                forward_return=torch.zeros(max(n, 1)),
                direction=torch.zeros(max(n, 1), dtype=torch.long),
            )
        ]

    epoch_votes = iter((2, 4, 6))

    def _fake_train_once(**kwargs):
        best_epoch = next(epoch_votes)
        return train_script._TrainingOutcome(
            model=SimpleNamespace(),
            trainer=SimpleNamespace(best_epoch=best_epoch, val_history=[1.0 / best_epoch]),
        )

    monkeypatch.setattr(train_script, "build_labeled_batches", _fake_batches)
    monkeypatch.setattr(train_script, "_train_model_once", _fake_train_once)
    monkeypatch.setattr(
        train_script,
        "_collect_prediction_calibration_inputs",
        lambda *args, **kwargs: train_script._PredictionCalibrationInputs(),
    )
    monkeypatch.setattr(
        train_script,
        "_fit_posthoc_prediction_calibration",
        lambda *args, **kwargs: None,
    )

    summary = train_script._walk_forward_model_selection(
        data_dir="data",
        dc=DataConfig(universe=("BANKNIFTY",), base_interval="5min", lookback_bars=12),
        labels=labels,
        split_manifest=manifest,
        management_horizon=12,
        execution_cfg=train_script.execution_config_from_cfg({}),
        batch_size=8,
        target_horizons=(3, 6, 12),
        model_kind="world_model",
        model_cfg=ModelConfig(latent_dim=16, temporal_layers=1, dropout=0.0, rollout_samples=4),
        barrier_mode="legacy",
        return_target_mode="exit",
        loss_weights=LossWeights(),
        train_cfg=TrainingConfig(device="cpu", n_folds=3, max_epochs=10, embargo_bars=1),
        rssm_epochs=1,
    )

    assert summary is not None
    assert summary["method"] == "walk_forward_cv"
    assert summary["selected_supervised_epochs"] == 4
    assert summary["n_completed_folds"] == 3


def test_walk_forward_model_selection_attaches_oof_prediction_calibration(monkeypatch) -> None:
    index = pd.date_range("2026-01-01 09:15:00", periods=36, freq="5min")
    labels = pd.DataFrame(
        {
            "label_realized_at": index + pd.Timedelta(minutes=5),
            "label_schema_version": [5] * len(index),
        },
        index=index,
    )
    manifest = ChronoSplitManifest.from_labels(
        labels,
        train_fraction=0.6,
        val_fraction=0.2,
        embargo_bars=1,
        bar_interval="5min",
    )

    def _fake_batches(*args, **kwargs):
        n = len(args[4])
        return [
            ForecastBatch(
                features=torch.zeros(max(n, 1), 1, 2, 2),
                forward_return=torch.zeros(max(n, 1)),
                direction=torch.zeros(max(n, 1), dtype=torch.long),
            )
        ]

    def _fake_train_once(**kwargs):
        return train_script._TrainingOutcome(
            model=SimpleNamespace(),
            trainer=SimpleNamespace(best_epoch=2, val_history=[0.4]),
        )

    monkeypatch.setattr(train_script, "build_labeled_batches", _fake_batches)
    monkeypatch.setattr(train_script, "_train_model_once", _fake_train_once)
    monkeypatch.setattr(
        train_script,
        "_collect_prediction_calibration_inputs",
        lambda *args, **kwargs: train_script._PredictionCalibrationInputs(),
    )
    monkeypatch.setattr(
        train_script,
        "_fit_posthoc_prediction_calibration",
        lambda *args, **kwargs: {
            "source": "walk_forward_oof_pretest",
            "sample_count": 24,
            "quantile_levels": [0.1, 0.25, 0.5, 0.75, 0.9],
            "barrier_temperature": 1.5,
            "regime_temperature": 1.0,
            "horizons": {},
        },
    )

    summary = train_script._walk_forward_model_selection(
        data_dir="data",
        dc=DataConfig(universe=("BANKNIFTY",), base_interval="5min", lookback_bars=12),
        labels=labels,
        split_manifest=manifest,
        management_horizon=12,
        execution_cfg=train_script.execution_config_from_cfg({}),
        batch_size=8,
        target_horizons=(3, 6, 12),
        model_kind="world_model",
        model_cfg=ModelConfig(latent_dim=16, temporal_layers=1, dropout=0.0, rollout_samples=4),
        barrier_mode="legacy",
        return_target_mode="exit",
        loss_weights=LossWeights(),
        train_cfg=TrainingConfig(device="cpu", n_folds=3, max_epochs=10, embargo_bars=1),
        rssm_epochs=1,
    )

    assert summary is not None
    assert summary["prediction_calibration"]["source"] == "walk_forward_oof_pretest"


def test_fit_posthoc_prediction_calibration_disables_class_temperature_transfer(monkeypatch) -> None:
    class _FakeCalibration:
        def to_metadata(self):
            return {
                "version": 1,
                "source": "walk_forward_oof_pretest",
                "sample_count": 10,
                "quantile_levels": [0.1, 0.25, 0.5, 0.75, 0.9],
                "barrier_temperature": 4.0,
                "regime_temperature": 1.3,
                "horizons": {},
            }

    monkeypatch.setattr(train_script, "fit_prediction_calibration", lambda **kwargs: _FakeCalibration())

    metadata = train_script._fit_posthoc_prediction_calibration(
        train_script._PredictionCalibrationInputs(),
        source="walk_forward_oof_pretest",
        allow_class_probability_temperatures=False,
    )

    assert metadata is not None
    assert metadata["barrier_temperature"] == 1.0
    assert metadata["regime_temperature"] == 1.0
    assert metadata["classification_transfer_disabled"] is True


def test_barrier_mode_from_cfg_defaults_and_validates() -> None:
    assert train_script._barrier_mode_from_cfg({}) == "legacy"
    assert train_script._barrier_mode_from_cfg({"model": {"barrier_mode": "derived"}}) == "derived"
    with pytest.raises(ValueError):
        train_script._barrier_mode_from_cfg({"model": {"barrier_mode": "unsupported"}})


def test_return_target_mode_from_cfg_defaults_and_validates() -> None:
    assert train_script._return_target_mode_from_cfg({}) == "horizon"
    assert train_script._return_target_mode_from_cfg({"model": {"return_target_mode": "horizon"}}) == "horizon"
    assert train_script._return_target_mode_from_cfg({"model": {"return_target_mode": "timeout"}}) == "timeout"
    assert train_script._return_target_mode_from_cfg({"model": {"return_target_mode": "exit"}}) == "exit"
    with pytest.raises(ValueError):
        train_script._return_target_mode_from_cfg({"model": {"return_target_mode": "unsupported"}})


def test_resolve_return_targets_uses_horizon_targets_by_default() -> None:
    labels = pd.DataFrame(
        {
            "exit_return": [0.01, -0.02],
            "realized_vol": [0.02, 0.03],
            "mae": [0.004, 0.005],
            "mfe": [0.006, 0.007],
            "barrier": ["timeout", "stop"],
            "barrier_valid": [True, True],
            "horizon_return_12": [0.03, 0.04],
            "horizon_vol_12": [0.011, 0.012],
            "horizon_mae_12": [0.013, 0.014],
            "horizon_mfe_12": [0.015, 0.016],
        }
    )

    targets = train_script._resolve_return_targets(
        labels,
        management_horizon=12,
        return_target_mode="horizon",
    )

    assert targets.return_weight is None
    assert targets.forward_return.tolist() == pytest.approx([0.03, 0.04])
    assert targets.realized_vol.tolist() == pytest.approx([0.011, 0.012])
    assert targets.mae.tolist() == pytest.approx([0.013, 0.014])
    assert targets.mfe.tolist() == pytest.approx([0.015, 0.016])


def test_resolve_return_targets_exit_mode_preserves_legacy_exit_target() -> None:
    labels = pd.DataFrame(
        {
            "exit_return": [0.01, -0.02],
            "realized_vol": [0.02, 0.03],
            "mae": [0.004, 0.005],
            "mfe": [0.006, 0.007],
            "horizon_return_12": [0.03, 0.04],
            "horizon_vol_12": [0.011, 0.012],
            "horizon_mae_12": [0.013, 0.014],
            "horizon_mfe_12": [0.015, 0.016],
        }
    )

    targets = train_script._resolve_return_targets(
        labels,
        management_horizon=12,
        return_target_mode="exit",
    )

    assert targets.return_weight is None
    assert targets.forward_return.tolist() == pytest.approx([0.01, -0.02])
    assert targets.realized_vol.tolist() == pytest.approx([0.02, 0.03])
    assert targets.mae.tolist() == pytest.approx([0.004, 0.005])
    assert targets.mfe.tolist() == pytest.approx([0.006, 0.007])


def test_resolve_return_targets_timeout_mode_masks_non_timeout_rows() -> None:
    labels = pd.DataFrame(
        {
            "exit_return": [0.01, -0.02, 0.05],
            "realized_vol": [0.02, 0.03, 0.04],
            "mae": [0.004, 0.005, 0.006],
            "mfe": [0.006, 0.007, 0.008],
            "barrier": ["timeout", "stop", "ambiguous"],
            "barrier_valid": [True, True, False],
            "horizon_return_12": [0.03, 0.04, 0.08],
            "horizon_vol_12": [0.011, 0.012, 0.013],
            "horizon_mae_12": [0.013, 0.014, 0.015],
            "horizon_mfe_12": [0.015, 0.016, 0.017],
        }
    )

    targets = train_script._resolve_return_targets(
        labels,
        management_horizon=12,
        return_target_mode="timeout",
    )

    assert targets.return_weight is not None
    assert targets.return_weight.tolist() == pytest.approx([1.0, 0.0, 0.0])
    assert targets.forward_return.tolist() == pytest.approx([0.03, 0.04, 0.08])
    assert targets.realized_vol.tolist() == pytest.approx([0.011, 0.012, 0.013])
    assert targets.mae.tolist() == pytest.approx([0.013, 0.014, 0.015])
    assert targets.mfe.tolist() == pytest.approx([0.015, 0.016, 0.017])


def test_effective_sample_weights_mix_uniqueness_and_opportunity() -> None:
    labels = pd.DataFrame(
        {
            "symbol": ["BANKNIFTY_FUT_continuous", "BANKNIFTY_FUT_continuous"],
            "entry_price": [50000.0, 50000.0],
            "sample_weight": [0.2, 0.2],
            "horizon_return_12": [0.004, 0.0001],
            "horizon_mae_12": [0.0003, 0.0015],
            "horizon_mfe_12": [0.0045, 0.0016],
        }
    )

    effective, audit = train_script._effective_sample_weights(
        labels,
        management_horizon=12,
        execution_cfg=train_script.execution_config_from_cfg({}),
    )

    assert effective.shape == (2,)
    assert effective[0] > effective[1]
    assert effective[1] > 0.0
    assert audit.mean_weight > 0.0


def test_train_model_once_applies_requested_barrier_mode() -> None:
    n_features = len(train_script.CANDLE_FEATURE_NAMES)
    batch = ForecastBatch(
        features=torch.randn(4, 2, 12, n_features),
        forward_return=torch.randn(4),
        direction=torch.zeros(4, dtype=torch.long),
        barrier_context=torch.tensor([[0.01, -0.02, 0.02]] * 4, dtype=torch.float32),
    )
    outcome = train_script._train_model_once(
        model_kind="forecaster",
        model_cfg=ModelConfig(latent_dim=16, temporal_layers=1, dropout=0.0),
        barrier_mode="derived",
        loss_weights=LossWeights(
            return_=1.0,
            direction=0.0,
            volatility=0.0,
            mae=0.0,
            mfe=0.0,
            barrier=0.0,
            regime=0.0,
            calibration=0.0,
            uncertainty=0.0,
            ood=0.0,
        ),
        train_cfg=TrainingConfig(device="cpu", max_epochs=1, lr=1e-3, embargo_bars=12),
        batches=[batch],
        target_horizons=(12,),
        fit_epochs=1,
    )

    assert getattr(outcome.model, "barrier_mode", None) == "derived"


def test_collect_prediction_calibration_inputs_respects_return_weight_mask() -> None:
    class _FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.probe = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, features, futures=None, regime_context=None, barrier_context=None):
            batch = features.shape[0]
            return {
                "return_quantiles": torch.tensor(
                    [[-0.02, -0.01, 0.01, 0.02, 0.03]] * batch,
                    dtype=torch.float32,
                    device=features.device,
                ),
                "volatility": torch.full((batch,), 0.01, dtype=torch.float32, device=features.device),
                "barrier_logits": torch.zeros((batch, 3), dtype=torch.float32, device=features.device),
                "regime_logits": torch.zeros((batch, 6), dtype=torch.float32, device=features.device),
            }

    batch = ForecastBatch(
        features=torch.zeros(2, 1, 2, 2),
        forward_return=torch.tensor([0.01, 0.99], dtype=torch.float32),
        direction=torch.tensor([2, 2], dtype=torch.long),
        realized_vol=torch.tensor([0.01, 0.99], dtype=torch.float32),
        return_weight=torch.tensor([1.0, 0.0], dtype=torch.float32),
    )

    inputs = train_script._collect_prediction_calibration_inputs(
        _FakeModel(),
        model_kind="forecaster",
        batches=[batch],
        target_horizons=(12,),
    )

    horizon_payload = inputs.horizon_payloads[12]
    assert horizon_payload["realized"] == pytest.approx([0.01])
    assert horizon_payload["realized_volatility"] == pytest.approx([0.01])


def test_train_model_once_wires_head_finetune_when_requested() -> None:
    """Review finding H7: _train_model_once (the single shared training routine
    used by both demo and real runs) must actually invoke HeadTrainer when
    head_finetune_epochs > 0, and must not when 0 (the default). The world-model
    path's own guard (`model_kind != "world_model"`) is a simple, directly-visible
    one-line condition in the source and isn't separately exercised here."""
    n = 4
    batches = [
        ForecastBatch(
            features=torch.randn(n, 1, 8, len(CANDLE_FEATURE_NAMES)),
            forward_return=torch.ones(n) * 0.01,
            direction=torch.zeros(n, dtype=torch.long),
            realized_vol=torch.full((n,), 0.01),
            barrier=torch.zeros(n, dtype=torch.long),
            sample_weight=torch.ones(n),
        )
    ]
    model_cfg = ModelConfig(latent_dim=16, temporal_layers=1, dropout=0.0)

    outcome_off = train_script._train_model_once(
        model_kind="forecaster",
        model_cfg=model_cfg,
        barrier_mode="legacy",
        loss_weights=LossWeights(),
        train_cfg=TrainingConfig(device="cpu", embargo_bars=12, head_finetune_epochs=0),
        batches=batches,
        target_horizons=(12,),
        fit_epochs=1,
    )
    assert outcome_off.head_trainer is None

    outcome_on = train_script._train_model_once(
        model_kind="forecaster",
        model_cfg=model_cfg,
        barrier_mode="legacy",
        loss_weights=LossWeights(),
        train_cfg=TrainingConfig(device="cpu", embargo_bars=12, head_finetune_epochs=2),
        batches=batches,
        target_horizons=(12,),
        fit_epochs=1,
    )
    assert outcome_on.head_trainer is not None
    assert len(outcome_on.head_trainer.history) == 2
