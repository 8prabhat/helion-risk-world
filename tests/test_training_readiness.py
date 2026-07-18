"""Training-readiness regressions for labeling, weighting, and trainer behavior."""

from __future__ import annotations

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from helion_risk_world.config.model_config import ModelConfig
from helion_risk_world.config.training_config import TrainingConfig
from helion_risk_world.config.execution_config import CostModelConfig
from helion_risk_world.losses.composite_loss import ForecasterLoss
from helion_risk_world.model import HRWForecaster
from helion_risk_world.training.opportunity_weighting import (
    compute_management_opportunity_weights,
)
from helion_risk_world.training.trainer import ForecastBatch, HRWTrainer
from helion_risk_world.training import trainer as trainer_module
from helion_risk_world.training.train_heads import HeadTrainer

# NOTE (Phase 2 migration): the BarrierLabeler/compute_uniqueness regression tests
# formerly here (drops-tail-without-full-horizon, next-bar-entry, ambiguous-same-bar
# dual-touch, uniqueness-position-as-start-index) were removed along with
# labeling/barrier_labeler.py and labeling/uniqueness.py. Their scenarios are now
# covered against the replacement engine directly: entry-offset/next-bar-open and
# ambiguous-tie handling in quanthelion's tests/unit/test_p2_labels_and_executor.py
# (TestTripleBarrier's entry_offset tests), and the full ambiguous/cost-floor/gap/
# session-boundary adapter behavior in this repo's tests/test_alpha_labels.py. AFML
# uniqueness weighting itself was superseded (not ported) by alpha_data's
# combined_sample_weights (uniqueness x return-attribution x time-decay) -- an
# accepted, deliberate upgrade, so there's no equivalent formula to regression-test.


def test_forecaster_loss_respects_sample_weights() -> None:
    loss = ForecasterLoss()
    prediction = {
        "return_quantiles": torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0], [5.0, 5.0, 5.0, 5.0, 5.0]],
            dtype=torch.float32,
        ),
        "uncertainty": torch.ones(2, dtype=torch.float32),
    }
    target_unweighted = ForecastBatch(
        features=torch.zeros(2, 1, 1, 1),
        forward_return=torch.zeros(2),
        direction=torch.zeros(2, dtype=torch.long),
    )
    target_weighted = ForecastBatch(
        features=torch.zeros(2, 1, 1, 1),
        forward_return=torch.zeros(2),
        direction=torch.zeros(2, dtype=torch.long),
        sample_weight=torch.tensor([1.0, 0.0], dtype=torch.float32),
    )

    unweighted = float(loss(prediction, target_unweighted))
    weighted = float(loss(prediction, target_weighted))

    assert weighted < unweighted


def test_opportunity_weights_emphasize_clean_cost_clearing_moves() -> None:
    labels = pd.DataFrame(
        {
            "symbol": [
                "BANKNIFTY_FUT_continuous",
                "BANKNIFTY_FUT_continuous",
                "BANKNIFTY_FUT_continuous",
            ],
            "entry_price": [50000.0, 50000.0, 50000.0],
            "horizon_return_12": [0.0045, 0.0002, -0.0035],
            "horizon_mae_12": [0.0004, 0.0018, 0.0038],
            "horizon_mfe_12": [0.0050, 0.0019, 0.0003],
        }
    )

    weights, audit = compute_management_opportunity_weights(
        labels,
        management_horizon=12,
        execution_cfg=CostModelConfig(),
    )

    assert weights[0] > 1.0
    assert weights[2] > 1.0
    assert weights[1] < 1.0
    assert weights[0] > weights[1]
    assert weights[2] > weights[1]
    assert audit.mean_roundtrip_cost_return > 0.0
    assert audit.pct_upweighted > 0.0


def test_trainer_early_stops_and_restores_best_val_epoch() -> None:
    torch.manual_seed(0)
    features = torch.randn(4, 1, 8, 5)
    batch = ForecastBatch(
        features=features,
        forward_return=torch.zeros(4),
        direction=torch.zeros(4, dtype=torch.long),
        sample_weight=torch.ones(4),
    )
    model = HRWForecaster(
        n_features=5,
        cfg=ModelConfig(latent_dim=16, temporal_layers=1, dropout=0.0),
    )
    trainer = HRWTrainer(
        model,
        ForecasterLoss(),
        TrainingConfig(
            device="cpu",
            lr=0.0,
            max_epochs=10,
            early_stopping_patience=2,
            embargo_bars=12,
        ),
    )

    trainer.fit([batch], val_batches=[batch])

    assert len(trainer.history) < 10
    assert trainer.best_epoch == 1
    assert len(trainer.val_history) == len(trainer.history)


def test_head_trainer_freezes_encoder_and_only_updates_head_params() -> None:
    """Review finding H7: HeadTrainer (Stage 4) is documented but was never wired
    into any CLI script. Verify it does what it claims on its own: encoder
    weights are unchanged after fine-tuning, only head weights move."""
    torch.manual_seed(0)
    features = torch.randn(4, 1, 8, 5)
    batch = ForecastBatch(
        features=features,
        forward_return=torch.ones(4) * 0.01,
        direction=torch.zeros(4, dtype=torch.long),
        realized_vol=torch.full((4,), 0.01),
        barrier=torch.zeros(4, dtype=torch.long),
        sample_weight=torch.ones(4),
    )
    model = HRWForecaster(n_features=5, cfg=ModelConfig(latent_dim=16, temporal_layers=1, dropout=0.0))
    encoder_before = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if name.startswith(("temporal", "cross_asset", "futures_encoder", "regime_encoder", "fusion"))
    }
    head_before = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if not name.startswith(("temporal", "cross_asset", "futures_encoder", "regime_encoder", "fusion"))
    }

    head_trainer = HeadTrainer(
        model, ForecasterLoss(), TrainingConfig(device="cpu", lr=0.1, embargo_bars=12),
        freeze_encoder=True,
    )
    head_trainer.fit([batch], epochs=2)

    assert len(head_trainer.history) == 2
    for name, p in model.named_parameters():
        if name in encoder_before:
            assert torch.equal(p, encoder_before[name]), f"encoder param {name} changed"
    changed = [
        name for name, p in model.named_parameters()
        if name in head_before and not torch.equal(p, head_before[name])
    ]
    assert changed, "expected at least one head parameter to change"
    # fit() re-enables requires_grad on everything afterward (avoid silent freeze
    # leaking into subsequent use of the model).
    assert all(p.requires_grad for p in model.parameters())


def test_trainer_history_is_sample_weighted_across_uneven_batches() -> None:
    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(()))

        def forward(self, features, futures=None, regime=None):
            return {"value": self.bias.expand(features.shape[0])}

    def _loss(prediction, batch):
        return prediction["value"].mean() + batch.forward_return.mean()

    batch_a = ForecastBatch(
        features=torch.zeros(4, 1, 1, 1),
        forward_return=torch.ones(4),
        direction=torch.zeros(4, dtype=torch.long),
    )
    batch_b = ForecastBatch(
        features=torch.zeros(1, 1, 1, 1),
        forward_return=torch.full((1,), 3.0),
        direction=torch.zeros(1, dtype=torch.long),
    )

    trainer = HRWTrainer(
        _Model(),
        _loss,
        TrainingConfig(device="cpu", lr=0.0, max_epochs=1, embargo_bars=12),
    )
    trainer.fit([batch_a, batch_b])

    assert trainer.history[0] == pytest.approx(1.4, abs=1e-6)


def test_epoch_batch_indices_are_deterministic_and_shuffled() -> None:
    order_a = trainer_module._epoch_batch_indices(5, seed=7, epoch=0)
    order_b = trainer_module._epoch_batch_indices(5, seed=7, epoch=0)
    order_c = trainer_module._epoch_batch_indices(5, seed=7, epoch=1)

    assert order_a == order_b
    assert sorted(order_a) == [0, 1, 2, 3, 4]
    assert order_a != [0, 1, 2, 3, 4]
    assert order_c != order_a
