"""Trading-utility checkpoint-selection metric (2026-07-18)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from helion_risk_world.config.model_config import ModelConfig  # noqa: E402
from helion_risk_world.config.training_config import TrainingConfig  # noqa: E402
from helion_risk_world.losses.composite_loss import ForecasterLoss  # noqa: E402
from helion_risk_world.model import HRWForecaster  # noqa: E402
from helion_risk_world.training.checkpoint_metrics import (  # noqa: E402
    trading_utility_loss,
    trading_utility_score,
)
from helion_risk_world.training.trainer import ForecastBatch, HRWTrainer  # noqa: E402

B, A, L, F = 8, 2, 12, 9


def _model() -> HRWForecaster:
    cfg = ModelConfig(size="small", latent_dim=16, temporal_layers=1, dropout=0.0)
    return HRWForecaster(n_features=F, cfg=cfg, n_quantiles=5, meta_label_lookback=12)


def _batch(meta_label: list[float], forced_side: torch.Tensor | None = None) -> ForecastBatch:
    n = len(meta_label)
    return ForecastBatch(
        features=torch.randn(n, A, L, F),
        forward_return=torch.zeros(n),
        direction=torch.zeros(n, dtype=torch.long),
        meta_label=torch.tensor(meta_label, dtype=torch.float32),
        primary_side=forced_side if forced_side is not None else torch.ones(n),
    )


def test_score_is_neutral_zero_with_untrained_model_and_no_meta_label() -> None:
    """No meta_label on the batch at all -> nothing to score -> neutral 0.0."""
    model = _model()
    batch = ForecastBatch(
        features=torch.randn(B, A, L, F),
        forward_return=torch.zeros(B),
        direction=torch.zeros(B, dtype=torch.long),
    )
    score = trading_utility_score(model, [batch], torch.device("cpu"))
    assert score == pytest.approx(0.0)


def test_score_and_loss_are_negations_of_each_other() -> None:
    model = _model()
    batch = _batch([1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    score = trading_utility_score(model, [batch], torch.device("cpu"), decision_threshold=-1.0)
    loss = trading_utility_loss(model, [batch], torch.device("cpu"), decision_threshold=-1.0)
    assert loss == pytest.approx(-score)


def test_forcing_all_trades_taken_reflects_true_label_balance() -> None:
    """decision_threshold=-1.0 forces every row with a proposed side to be 'taken'
    (sigmoid output is always > -1), so the score must equal the raw (correct-incorrect)/total
    label balance regardless of what the untrained head predicts."""
    model = _model()
    labels = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]  # 4 correct, 4 incorrect -> 0.0
    batch = _batch(labels)
    score = trading_utility_score(model, [batch], torch.device("cpu"), decision_threshold=-1.0)
    assert score == pytest.approx(0.0)

    labels_positive = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]  # 6 correct, 2 incorrect
    batch_positive = _batch(labels_positive)
    score_positive = trading_utility_score(
        model, [batch_positive], torch.device("cpu"), decision_threshold=-1.0
    )
    assert score_positive == pytest.approx((6 - 2) / 8)


def test_rows_with_primary_side_zero_are_excluded() -> None:
    """A row with primary_side==0 has no proposed bet -- must never count toward the
    score even if meta_label happens to be populated for it."""
    model = _model()
    side = torch.tensor([1.0, 1.0, 0.0, 0.0])
    batch = _batch([1.0, 0.0, 1.0, 0.0], forced_side=side)
    score = trading_utility_score(model, [batch], torch.device("cpu"), decision_threshold=-1.0)
    # Only the first two rows (side != 0) count: 1 correct, 1 incorrect -> 0.0
    assert score == pytest.approx(0.0)


def test_nan_meta_label_rows_are_excluded() -> None:
    model = _model()
    batch = _batch([1.0, float("nan"), 1.0, float("nan")])
    score = trading_utility_score(model, [batch], torch.device("cpu"), decision_threshold=-1.0)
    assert score == pytest.approx(1.0)  # only the two valid rows, both correct


def test_checkpoint_metric_drives_selection_in_trainer_fit() -> None:
    """HRWTrainer.fit() with a custom checkpoint_metric must populate
    val_metric_history and select best_epoch by that metric, not composite val_loss."""
    model = _model()
    cfg = TrainingConfig(max_epochs=2, lr=1e-3, seed=7, early_stopping_patience=0)
    loss_fn = ForecasterLoss()
    train_batch = _batch([1.0] * B)
    val_batch = _batch([1.0] * B)
    trainer = HRWTrainer(model, loss_fn, cfg, checkpoint_metric=trading_utility_loss)
    trainer.fit([train_batch], val_batches=[val_batch])
    assert len(trainer.val_metric_history) == 2
    assert trainer.best_epoch is not None


def test_default_trainer_behavior_unchanged_without_checkpoint_metric() -> None:
    model = _model()
    cfg = TrainingConfig(max_epochs=2, lr=1e-3, seed=7, early_stopping_patience=0)
    loss_fn = ForecasterLoss()
    train_batch = _batch([1.0] * B)
    val_batch = _batch([1.0] * B)
    trainer = HRWTrainer(model, loss_fn, cfg)
    trainer.fit([train_batch], val_batches=[val_batch])
    assert trainer.val_metric_history == []
    assert len(trainer.val_history) == 2
