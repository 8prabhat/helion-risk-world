from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from helion_risk_world.config.training_config import TrainingConfig
from helion_risk_world.integration.quanthelion_adapter import ModelProtocol
from helion_risk_world.losses.composite_loss import ForecasterLoss
from helion_risk_world.training.trainer import ForecastBatch, HRWTrainer


class ForecasterTrainer:
    """Stage 3: multi-horizon distributional heads (SPEC.md §20)."""

    def __init__(
        self,
        model: ModelProtocol,
        cfg: TrainingConfig,
        loss: ForecasterLoss | None = None,
    ) -> None:
        self._trainer = HRWTrainer(model, loss or ForecasterLoss(), cfg)

    def fit(
        self,
        batches: Sequence[ForecastBatch],
        *,
        epochs: int | None = None,
        val_batches: Sequence[ForecastBatch] | None = None,
    ) -> ModelProtocol:
        return self._trainer.fit(batches, epochs=epochs, val_batches=val_batches)

    def run(self, cfg: Any) -> Any:
        batches = cfg["batches"]
        epochs = cfg.get("epochs")
        return self.fit(batches, epochs=epochs, val_batches=cfg.get("val_batches"))
