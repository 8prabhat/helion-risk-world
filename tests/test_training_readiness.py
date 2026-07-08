"""Training-readiness regressions for labeling, weighting, and trainer behavior."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from helion_risk_world.config.model_config import ModelConfig
from helion_risk_world.config.training_config import TrainingConfig
from helion_risk_world.config.execution_config import CostModelConfig
from helion_risk_world.labeling.barrier_labeler import BarrierLabeler
from helion_risk_world.labeling.uniqueness import compute_uniqueness
from helion_risk_world.losses.composite_loss import ForecasterLoss
from helion_risk_world.model import HRWForecaster
from helion_risk_world.schemas.label_schema import Barrier, LabelRecord
from helion_risk_world.training.opportunity_weighting import (
    compute_management_opportunity_weights,
)
from helion_risk_world.training.trainer import ForecastBatch, HRWTrainer
from helion_risk_world.training import trainer as trainer_module
from helion_risk_world.training.train_heads import HeadTrainer


def test_barrier_labeler_drops_tail_without_full_horizon() -> None:
    ts0 = datetime(2026, 1, 1, 9, 15)
    timestamps = [ts0 + timedelta(minutes=5 * i) for i in range(8)]
    close = [100.0, 101.0, 100.5, 100.2, 100.4, 100.7, 100.9, 101.0]

    records = BarrierLabeler(H=3, add_uniqueness=False).label(timestamps, close)

    assert len(records) == len(close) - 3
    assert records[-1].ts == timestamps[4]
    assert all(1 <= record.exit_t <= 3 for record in records)


def test_barrier_labeler_uses_next_bar_as_entry() -> None:
    ts0 = datetime(2026, 1, 1, 9, 15)
    timestamps = [ts0 + timedelta(minutes=5 * i) for i in range(5)]
    close = [100.0, 110.0, 111.0, 112.0, 113.0]

    record = BarrierLabeler(H=2, u=100.0, d=100.0, add_uniqueness=False).label(timestamps, close)[0]

    assert record.ts == timestamps[0]
    assert record.exit_return == pytest.approx((111.0 / 110.0) - 1.0, abs=1e-6)


def test_barrier_labeler_marks_same_bar_dual_touch_as_ambiguous() -> None:
    ts0 = datetime(2026, 1, 1, 9, 15)
    timestamps = [ts0 + timedelta(minutes=5 * i) for i in range(5)]
    close = [100.0, 100.0, 100.0, 100.0, 100.0]
    open_prices = [100.0, 100.0, 100.0, 100.0, 100.0]
    high_prices = [100.0, 101.0, 100.5, 100.5, 100.5]
    low_prices = [100.0, 99.0, 99.5, 99.5, 99.5]

    record = BarrierLabeler(H=2, u=0.1, d=0.1, add_uniqueness=False).label(
        timestamps,
        close,
        open_prices=open_prices,
        high_prices=high_prices,
        low_prices=low_prices,
    )[0]

    assert record.barrier is Barrier.AMBIGUOUS
    assert record.barrier_valid is False
    assert record.entry_price == pytest.approx(100.0, abs=1e-6)
    assert record.exit_price == pytest.approx(100.0, abs=1e-6)


def test_uniqueness_uses_record_position_as_start_index() -> None:
    ts0 = datetime(2026, 1, 1, 9, 15)
    records = [
        LabelRecord(
            symbol="X",
            ts=ts0,
            label_realized_at=ts0 + timedelta(minutes=5),
            horizon_bars=3,
            barrier=Barrier.TARGET,
            exit_return=0.01,
            exit_t=1,
            realized_vol=0.01,
            mae=0.0,
            mfe=0.01,
        ),
        LabelRecord(
            symbol="X",
            ts=ts0 + timedelta(minutes=5),
            label_realized_at=ts0 + timedelta(minutes=10),
            horizon_bars=3,
            barrier=Barrier.TIMEOUT,
            exit_return=0.0,
            exit_t=2,
            realized_vol=0.01,
            mae=0.0,
            mfe=0.0,
        ),
    ]

    weights = compute_uniqueness(records)

    assert weights[0] == pytest.approx(0.75, abs=1e-6)
    assert weights[1] == pytest.approx(5.0 / 6.0, abs=1e-6)


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
