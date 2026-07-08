"""skip_if_non_finite + its wiring into the training loops (review finding H5).

A single bad batch (e.g. a division-by-zero in a rolling-vol feature, or a roll-gap
bar) can produce a NaN/Inf loss. Without a guard, `.backward()` propagates NaN
gradients and `optim.step()` silently corrupts every model parameter for the rest of
the run. These tests engineer exactly that batch and confirm the trainers skip it
instead of corrupting the model.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from helion_risk_world.config.training_config import TrainingConfig
from helion_risk_world.training.nan_guard import skip_if_non_finite
from helion_risk_world.training.trainer import ForecastBatch, HRWTrainer


def test_skip_if_non_finite_detects_nan_and_inf() -> None:
    assert skip_if_non_finite(torch.tensor(float("nan")), context="t") is True
    assert skip_if_non_finite(torch.tensor(float("inf")), context="t") is True
    assert skip_if_non_finite(torch.tensor(-float("inf")), context="t") is True


def test_skip_if_non_finite_passes_finite_values() -> None:
    assert skip_if_non_finite(torch.tensor(0.0), context="t") is False
    assert skip_if_non_finite(torch.tensor(3.14), context="t") is False


class _BiasModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def forward(self, features, futures=None, regime=None):
        return {"value": self.bias.expand(features.shape[0])}


def _discriminating_loss(prediction, batch):
    # NaN sentinel in forward_return -> NaN loss, mirroring a real bad-batch scenario
    # (e.g. a feature-engineering bug producing NaN inputs for one bar).
    return prediction["value"].mean() + batch.forward_return.mean()


def test_hrw_trainer_skips_non_finite_batch_without_corrupting_model() -> None:
    good = ForecastBatch(
        features=torch.zeros(4, 1, 1, 1),
        forward_return=torch.ones(4),
        direction=torch.zeros(4, dtype=torch.long),
    )
    bad = ForecastBatch(
        features=torch.zeros(4, 1, 1, 1),
        forward_return=torch.full((4,), float("nan")),
        direction=torch.zeros(4, dtype=torch.long),
    )

    trainer = HRWTrainer(
        _BiasModel(),
        _discriminating_loss,
        TrainingConfig(device="cpu", lr=0.1, max_epochs=1, embargo_bars=12, grad_accum_steps=1),
    )
    model = trainer.fit([good, bad])

    assert trainer.n_skipped_batches == 1
    assert bool(torch.isfinite(model.bias))
    assert torch.isfinite(torch.tensor(trainer.history[0]))


def test_hrw_trainer_all_batches_finite_skips_nothing() -> None:
    good = ForecastBatch(
        features=torch.zeros(4, 1, 1, 1),
        forward_return=torch.ones(4),
        direction=torch.zeros(4, dtype=torch.long),
    )
    trainer = HRWTrainer(
        _BiasModel(),
        _discriminating_loss,
        TrainingConfig(device="cpu", lr=0.1, max_epochs=1, embargo_bars=12),
    )
    trainer.fit([good])
    assert trainer.n_skipped_batches == 0
