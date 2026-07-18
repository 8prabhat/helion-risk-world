"""Day-4 forecaster: head shapes, non-crossing quantiles, trainer overfit (SPEC.md §17, §20, §27).

Direction head removed (SPEC.md §15); side is inferred from return-quantile asymmetry.
"""

from __future__ import annotations

import pytest
from types import SimpleNamespace

torch = pytest.importorskip("torch")

from helion_risk_world.config.model_config import LossWeights, ModelConfig  # noqa: E402
from helion_risk_world.config.training_config import TrainingConfig  # noqa: E402
from helion_risk_world.losses.composite_loss import ForecasterLoss  # noqa: E402
from helion_risk_world.losses.quantile_loss import QuantileLoss  # noqa: E402
from helion_risk_world.model import HRWForecaster  # noqa: E402
from helion_risk_world.training.trainer import ForecastBatch, HRWTrainer  # noqa: E402

B, A, L, F = 8, 2, 12, 9


def _model(latent: int = 32) -> HRWForecaster:
    cfg = ModelConfig(size="small", latent_dim=latent, temporal_layers=1, dropout=0.0)
    return HRWForecaster(n_features=F, cfg=cfg, n_quantiles=5)


def test_forecaster_forward_shapes() -> None:
    out = _model()(torch.randn(B, A, L, F))
    assert out["z"].shape == (B, 32)
    assert out["return_quantiles"].shape == (B, 5)
    # direction_logits removed from spec (§15)
    assert "direction_logits" not in out
    assert out["volatility"].shape == (B,)          # [B] not [B, 1]
    assert out["mae"].shape == (B,)
    assert out["mfe"].shape == (B,)
    assert out["barrier_logits"].shape == (B, 3)
    assert out["uncertainty"].shape == (B,)
    assert (out["uncertainty"] >= 0).all()           # softplus-positive


def test_model_config_wiring_rejects_unimplemented_fusion_modes() -> None:
    with pytest.raises(NotImplementedError):
        HRWForecaster(
            n_features=F,
            cfg=ModelConfig(latent_dim=32, temporal_layers=1, dropout=0.0, fusion="attention"),
        )


def test_futures_conv_layers_are_wired_from_model_config() -> None:
    model = HRWForecaster(
        n_features=F,
        cfg=ModelConfig(latent_dim=32, temporal_layers=1, dropout=0.0, futures_conv_layers=3),
    )
    conv_layers = [layer for layer in model.futures_encoder.conv if isinstance(layer, torch.nn.Conv1d)]
    assert len(conv_layers) == 3


def test_return_quantiles_are_non_decreasing() -> None:
    q = _model()(torch.randn(B, A, L, F))["return_quantiles"]
    diffs = q[:, 1:] - q[:, :-1]
    assert (diffs >= -1e-6).all(), "quantiles must not cross"


def test_untrained_return_quantiles_start_on_target_scale() -> None:
    q = _model()(torch.randn(B, A, L, F))["return_quantiles"].detach()
    means = q.mean(dim=0)
    assert float(means[0]) == pytest.approx(-0.003, abs=0.0015)
    assert float(means[2]) == pytest.approx(0.0, abs=0.0015)
    assert float(means[-1]) == pytest.approx(0.003, abs=0.0015)


def test_quantile_loss_pinball_value() -> None:
    loss = QuantileLoss(quantiles=(0.5,))
    pred = torch.tensor([[1.0]])
    target = torch.tensor([2.0])
    assert float(loss(pred, target)) == pytest.approx(0.5)


def test_forecaster_loss_includes_calibration_component() -> None:
    loss = ForecasterLoss(
        weights=LossWeights(return_=0.0, uncertainty=0.0, calibration=1.0),
    )
    prediction = {
        "return_quantiles": torch.tensor(
            [[-0.02, -0.01, 0.0, 0.01, 0.02], [-0.02, -0.01, 0.0, 0.01, 0.02]],
            dtype=torch.float32,
        ),
        "uncertainty": torch.full((2,), 0.01, dtype=torch.float32),
    }
    target = SimpleNamespace(
        forward_return=torch.tensor([0.03, 0.03], dtype=torch.float32),
        direction=torch.zeros(2, dtype=torch.long),
        realized_vol=torch.full((2,), 0.01, dtype=torch.float32),
    )

    total = loss(prediction, target)

    assert float(total) > 0.0
    assert "calibration" in loss.last_components
    assert loss.last_components["calibration"] > 0.0


def test_forecaster_direction_surrogate_penalizes_wrong_median_sign() -> None:
    loss = ForecasterLoss(
        weights=LossWeights(
            return_=0.0,
            calibration=0.0,
            uncertainty=0.0,
            volatility=0.0,
            mae=0.0,
            mfe=0.0,
            barrier=0.0,
            regime=0.0,
            direction=1.0,
        )
    )
    target = SimpleNamespace(
        forward_return=torch.tensor([0.01], dtype=torch.float32),
        direction=torch.tensor([2], dtype=torch.long),
    )
    aligned = {
        "return_quantiles": torch.tensor([[-0.002, -0.001, 0.004, 0.006, 0.008]], dtype=torch.float32),
        "uncertainty": torch.tensor([0.01], dtype=torch.float32),
    }
    misaligned = {
        "return_quantiles": torch.tensor([[-0.008, -0.006, -0.004, -0.001, 0.002]], dtype=torch.float32),
        "uncertainty": torch.tensor([0.01], dtype=torch.float32),
    }

    aligned_loss = float(loss(aligned, target))
    misaligned_loss = float(loss(misaligned, target))

    assert misaligned_loss > aligned_loss
    assert loss.last_components["direction"] > 0.0


def test_forecaster_loss_uses_barrier_geometry_for_derived_barrier_supervision() -> None:
    loss = ForecasterLoss(
        weights=LossWeights(
            return_=0.0,
            calibration=0.0,
            uncertainty=0.0,
            volatility=0.0,
            mae=0.0,
            mfe=0.0,
            barrier=1.0,
            regime=0.0,
            direction=0.0,
        )
    )
    target = SimpleNamespace(
        forward_return=torch.tensor([0.0], dtype=torch.float32),
        direction=torch.tensor([1], dtype=torch.long),
        mae=torch.tensor([0.006], dtype=torch.float32),
        mfe=torch.tensor([0.001], dtype=torch.float32),
        barrier_context=torch.tensor([[0.001, -0.002, 0.002]], dtype=torch.float32),
    )
    aligned = {
        "return_quantiles": torch.zeros((1, 5), dtype=torch.float32),
        "uncertainty": torch.tensor([0.01], dtype=torch.float32),
        "barrier_logits": torch.tensor([[3.0, -3.0, -3.0]], dtype=torch.float32),
    }
    misaligned = {
        "return_quantiles": torch.zeros((1, 5), dtype=torch.float32),
        "uncertainty": torch.tensor([0.01], dtype=torch.float32),
        "barrier_logits": torch.tensor([[-3.0, 3.0, -3.0]], dtype=torch.float32),
    }

    aligned_loss = float(loss(aligned, target))
    aligned_aux = loss.last_components["excursion_barrier"]
    misaligned_loss = float(loss(misaligned, target))
    misaligned_aux = loss.last_components["excursion_barrier"]

    assert misaligned_loss > aligned_loss
    assert misaligned_aux > aligned_aux


def test_forecaster_return_supervision_can_mask_non_timeout_rows() -> None:
    loss = ForecasterLoss(
        LossWeights(
            return_=1.0,
            calibration=0.0,
            uncertainty=0.0,
            volatility=0.0,
            mae=0.0,
            mfe=0.0,
            barrier=0.0,
            regime=0.0,
            direction=1.0,
        )
    )
    masked_target = SimpleNamespace(
        forward_return=torch.tensor([0.01, -0.03], dtype=torch.float32),
        direction=torch.tensor([2, 0], dtype=torch.long),
        return_weight=torch.tensor([1.0, 0.0], dtype=torch.float32),
    )
    first_only_target = SimpleNamespace(
        forward_return=torch.tensor([0.01], dtype=torch.float32),
        direction=torch.tensor([2], dtype=torch.long),
    )
    masked_prediction = {
        "return_quantiles": torch.tensor(
            [
                [-0.002, -0.001, 0.004, 0.006, 0.008],
                [0.020, 0.030, 0.040, 0.050, 0.060],
            ],
            dtype=torch.float32,
        ),
        "uncertainty": torch.tensor([0.01, 0.01], dtype=torch.float32),
    }
    first_only_prediction = {
        "return_quantiles": masked_prediction["return_quantiles"][:1],
        "uncertainty": masked_prediction["uncertainty"][:1],
    }

    masked_loss = float(loss(masked_prediction, masked_target))
    reference_loss = float(loss(first_only_prediction, first_only_target))

    assert masked_loss == pytest.approx(reference_loss)


def test_forecaster_loss_penalizes_return_excursion_incoherence() -> None:
    loss = ForecasterLoss(
        LossWeights(
            return_=0.0,
            calibration=0.0,
            uncertainty=0.0,
            volatility=0.0,
            barrier=0.0,
            regime=0.0,
            direction=0.0,
            mae=1.0,
            mfe=1.0,
        )
    )
    target = SimpleNamespace(
        forward_return=torch.tensor([0.0], dtype=torch.float32),
        direction=torch.tensor([1], dtype=torch.long),
        mae=torch.tensor([0.005], dtype=torch.float32),
        mfe=torch.tensor([0.005], dtype=torch.float32),
    )
    coherent = {
        "return_quantiles": torch.tensor([[-0.004, -0.002, 0.0, 0.002, 0.004]], dtype=torch.float32),
        "uncertainty": torch.tensor([0.01], dtype=torch.float32),
        "mae": torch.tensor([0.005], dtype=torch.float32),
        "mfe": torch.tensor([0.005], dtype=torch.float32),
    }
    incoherent = {
        "return_quantiles": torch.tensor([[-0.007, -0.006, 0.0, 0.006, 0.007]], dtype=torch.float32),
        "uncertainty": torch.tensor([0.01], dtype=torch.float32),
        "mae": torch.tensor([0.005], dtype=torch.float32),
        "mfe": torch.tensor([0.005], dtype=torch.float32),
    }

    coherent_loss = float(loss(coherent, target))
    coherent_aux = loss.last_components["excursion_coherence"]
    incoherent_loss = float(loss(incoherent, target))
    incoherent_aux = loss.last_components["excursion_coherence"]

    assert incoherent_loss > coherent_loss
    assert incoherent_aux > coherent_aux


def test_trainer_overfits_small_slice() -> None:
    torch.manual_seed(0)
    features = torch.randn(B, A, L, F)
    signal = features.mean(dim=(1, 2, 3))
    forward_return = signal * 0.05
    direction = torch.bucketize(signal, torch.tensor([-0.3, 0.3]))
    batch = ForecastBatch(features=features, forward_return=forward_return, direction=direction)

    model = _model(latent=32)
    loss_fn = ForecasterLoss(
        LossWeights(
            return_=1.0,
            calibration=0.0,
            uncertainty=0.0,
            volatility=0.0,
            mae=0.0,
            mfe=0.0,
            barrier=0.0,
            regime=0.0,
            direction=0.0,
            repr_var=0.0,
            repr_cov=0.0,
        )
    )
    cfg = TrainingConfig(seed=0, device="cpu", lr=1e-2, max_epochs=200, embargo_bars=12)
    trainer = HRWTrainer(model, loss_fn, cfg)
    trainer.fit([batch])

    assert len(trainer.history) == 200
    first, best = trainer.history[0], min(trainer.history)
    assert best < first, f"expected loss to improve, got {first:.4f} -> best {best:.4f}"

    # Return quantiles reflect the learnt signal (positive median for positive signals).
    model.eval()
    with torch.no_grad():
        q_median = model(features)["return_quantiles"][:, 2]  # 0.5 quantile
    corr = torch.corrcoef(torch.stack([forward_return, q_median]))[0, 1]
    assert float(corr) > 0.6
