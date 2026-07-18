"""Stage 4: supervised head fine-tuning on triple-barrier labels (SPEC.md §20, Stage 4).

Fine-tunes the distribution heads (barrier, volatility, regime, return-quantile) of a
pre-trained HRWForecaster using labeled data from ``data/alpha_labels.py``'s
``build_alpha_labels`` (Phase 2 migration).

Two modes:
  freeze_encoder=True  (default) — only head parameters optimised; encoder is frozen.
  freeze_encoder=False           — full model fine-tuning at a lower learning rate.

SRP: head training only — encoder pretraining lives in pretrain_market_state.py (Stage 2),
     RSSM dynamics training lives in train_world_model.py (Stage 3).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from helion_risk_world.config.training_config import TrainingConfig
from helion_risk_world.losses.composite_loss import ForecasterLoss
from helion_risk_world.model import HRWForecaster
from helion_risk_world.training.nan_guard import skip_if_non_finite
from helion_risk_world.training.trainer import ForecastBatch, resolve_device

# Single source of truth for "which modules are the encoder" (review finding M11:
# this used to be duplicated as two independently-maintained literal sets, one in
# _head_parameters() and one inline in fit(), which could silently drift apart).
_ENCODER_MODULE_PREFIXES = ("temporal", "cross_asset", "futures_encoder", "regime_encoder", "fusion")


class HeadTrainer:
    """Fine-tune distribution heads on triple-barrier labeled data.

    Args:
        model:          HRWForecaster with a pre-trained encoder.
        loss:           ForecasterLoss (quantile + volatility + barrier + regime terms).
        cfg:            TrainingConfig (lr, max_epochs, seed, device).
        freeze_encoder: if True, only head parameters receive gradients.
    """

    def __init__(
        self,
        model: HRWForecaster,
        loss: ForecasterLoss,
        cfg: TrainingConfig,
        freeze_encoder: bool = True,
    ) -> None:
        self._model = model
        self._loss = loss
        self._cfg = cfg
        self._freeze_encoder = freeze_encoder
        self.history: list[float] = []
        self.n_skipped_batches: int = 0

    def _head_parameters(self):  # type: ignore[return]
        """Return parameters from head modules only (excludes encoder/fusion)."""
        for name, param in self._model.named_parameters():
            if not any(name.startswith(mod) for mod in _ENCODER_MODULE_PREFIXES):
                yield param

    def fit(self, batches: Sequence[ForecastBatch], *, epochs: int | None = None) -> HRWForecaster:
        """Fine-tune on labeled batches.

        Args:
            batches: sequence of ForecastBatch with barrier, realized_vol, regime labels
                     populated alongside the standard forward_return and features.
            epochs:  override cfg.max_epochs.

        Returns:
            The fine-tuned model (same object, mutated in-place).
        """
        if not batches:
            raise ValueError("fit requires at least one batch")
        torch.manual_seed(self._cfg.seed)
        device = resolve_device(self._cfg.device)
        model = self._model.to(device).train()

        if self._freeze_encoder:
            # Freeze encoder modules to prevent gradient flow from heads backward into encoder
            for name, param in model.named_parameters():
                if any(name.startswith(m) for m in _ENCODER_MODULE_PREFIXES):
                    param.requires_grad_(False)
            params = list(self._head_parameters())
        else:
            # Full fine-tuning at (potentially lower) lr from config
            for param in model.parameters():
                param.requires_grad_(True)
            params = list(model.parameters())

        optim = torch.optim.Adam(params, lr=self._cfg.lr)
        n_epochs = epochs or self._cfg.max_epochs
        self.history = []
        self.n_skipped_batches = 0

        for epoch in range(n_epochs):
            running, n = 0.0, 0
            accum_steps = self._cfg.grad_accum_steps
            optim.zero_grad()
            for idx, batch in enumerate(batches, start=1):
                batch = batch.to(device)
                out = model(batch.features, batch.futures, batch.regime_context)
                loss = self._loss(out, batch)
                if skip_if_non_finite(loss, context=f"HeadTrainer.fit epoch={epoch + 1} idx={idx}"):
                    self.n_skipped_batches += 1
                    optim.zero_grad()
                    continue
                (loss / accum_steps).backward()
                should_step = (idx % accum_steps == 0) or (idx == len(batches))
                if should_step:
                    if self._cfg.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(params, self._cfg.grad_clip_norm)
                    optim.step()
                    optim.zero_grad()
                running += float(loss.detach())
                n += 1
            self.history.append(running / max(n, 1))

        # Re-enable all parameters after training (avoid silent freeze in subsequent use)
        for param in model.parameters():
            param.requires_grad_(True)
        return model

    def run(self, cfg: Any) -> Any:
        return self.fit(cfg["batches"], epochs=cfg.get("epochs"))


__all__ = ["HeadTrainer"]
