"""Stage 2 — self-supervised market representation pretraining (SPEC.md §20.2).

Pretrains the market encoder by **future-latent prediction**: from the latent of a context window it
predicts the latent of a window ``gap`` steps in the future, and the (encoded) future latent is the
self-supervised target. Trained with the VICReg-style ``LatentPredictionLoss`` so the representation
stays informative (anti-collapse) without any reconstruction or labels. This makes the latent state
a *model of the world* rather than a feature bag — and is the prerequisite for the Stage-4 latent
dynamics. Implemented LOCALLY; never imports msh_jepa (SPEC.md §4).

The other Stage-2 tasks named in the spec (masked-window, cross-asset consistency, regime-
contrastive, vol reconstruction) are auxiliary; future-latent prediction is the core
dynamics-relevant objective and the others plug in as additional terms.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from helion_risk_world.config.training_config import TrainingConfig
from helion_risk_world.integration.quanthelion_adapter import get_logger
from helion_risk_world.losses.latent_consistency_loss import LatentPredictionLoss
from helion_risk_world.training.nan_guard import skip_if_non_finite
from helion_risk_world.training.trainer import resolve_device

_log = get_logger("hrw.pretrain")


@dataclass
class LatentPair:
    """A (context, future) window pair for future-latent prediction. Features only in V1."""

    context: Tensor   # [B, A, L, F]  window ending at t
    future: Tensor    # [B, A, L, F]  window ending at t + gap
    context_futures: Tensor | None = None
    future_futures: Tensor | None = None
    context_regime: Tensor | None = None
    future_regime: Tensor | None = None

    def to(self, device: torch.device) -> LatentPair:
        def _mv(t: Tensor | None) -> Tensor | None:
            return t.to(device) if t is not None else None

        return LatentPair(
            self.context.to(device),
            self.future.to(device),
            context_futures=_mv(self.context_futures),
            future_futures=_mv(self.future_futures),
            context_regime=_mv(self.context_regime),
            future_regime=_mv(self.future_regime),
        )


class MarketStatePretrainer:
    """Stage 2: self-supervised encoder pretraining via future-latent prediction (SPEC.md §20.2).

    Trains ``model.encode`` (the temporal/cross-asset/surface/regime fusion) plus a small predictor
    so the latent state predicts its own future — the world-model representation objective.
    Implemented LOCALLY — never imports msh_jepa.
    """

    def __init__(
        self, model: nn.Module, cfg: TrainingConfig, loss: LatentPredictionLoss | None = None
    ) -> None:
        self._model = model
        self._cfg = cfg
        self._loss = loss or LatentPredictionLoss()
        d = int(model.latent_dim)  # type: ignore[attr-defined]
        # Predictor: present latent -> predicted future latent (discarded after pretraining).
        self._predictor = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.history: list[float] = []
        self.n_skipped_batches: int = 0

    def fit(self, pairs: Sequence[LatentPair], *, epochs: int | None = None) -> nn.Module:
        """Pretrain the encoder on future-latent prediction. Returns the (pretrained) model."""
        if not pairs:
            raise ValueError("fit requires a non-empty sequence of LatentPair")
        torch.manual_seed(self._cfg.seed)
        device = resolve_device(self._cfg.device)
        self._model.to(device)
        self._predictor.to(device)
        self._model.train()

        params = list(self._model.parameters()) + list(self._predictor.parameters())
        optim = torch.optim.Adam(params, lr=self._cfg.lr, weight_decay=self._cfg.weight_decay)
        n_epochs = epochs if epochs is not None else self._cfg.max_epochs

        self.history = []
        self.n_skipped_batches = 0
        for epoch in range(n_epochs):
            running, total_mass = 0.0, 0.0
            for idx, pair in enumerate(pairs, start=1):
                pair = pair.to(device)
                optim.zero_grad()
                z_ctx = self._model.encode(
                    pair.context,
                    pair.context_futures,
                    pair.context_regime,
                )        # [B, d]  (online)
                with torch.no_grad():
                    z_future = self._model.encode(
                        pair.future,
                        pair.future_futures,
                        pair.future_regime,
                    )  # [B, d]  (target, stop-grad)
                pred = self._predictor(z_ctx)
                loss = self._loss(pred, z_future)
                if skip_if_non_finite(
                    loss, context=f"MarketStatePretrainer.fit epoch={epoch + 1} idx={idx}"
                ):
                    self.n_skipped_batches += 1
                    optim.zero_grad()
                    continue
                loss.backward()
                if self._cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, self._cfg.grad_clip_norm)
                optim.step()
                batch_mass = float(pair.context.shape[0])
                running += float(loss.detach()) * batch_mass
                total_mass += batch_mass
            mean = running / max(total_mass, 1e-8)
            self.history.append(mean)
            if epoch == 0 or (epoch + 1) % max(1, n_epochs // 5) == 0:
                _log.info("hrw.pretrain.epoch", epoch=epoch + 1, loss=round(mean, 5),
                          **{k: round(v, 4) for k, v in self._loss.last_components.items()})
        return self._model

    @torch.no_grad()
    def latent_collapse_std(self, pairs: Sequence[LatentPair]) -> float:
        """Mean per-dim std of the encoded latents — a collapse check (should stay well above 0)."""
        device = resolve_device(self._cfg.device)
        self._model.eval()
        zs = [
            self._model.encode(
                moved.context,
                moved.context_futures,
                moved.context_regime,
            )
            for moved in (p.to(device) for p in pairs)
        ]
        return float(torch.cat(zs, dim=0).std(dim=0).mean())


__all__ = ["MarketStatePretrainer", "LatentPair"]
