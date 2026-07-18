"""Stage 3: RSSM dynamics training (SPEC.md §14.2, §20, Stage 3).

Trains the RSSM latent world model (prior + posterior + decode head) using ``RSSMLoss``
(L_dyn + L_imag). The encoder E_φ is assumed to be frozen or pre-trained (Stage 2); only
``model.market_world.rssm`` parameters are optimised by default.

Input contract for ``fit()``:
  Each ``seq`` in ``sequences`` must be a pre-encoded tensor ``[T, B, embed_dim]`` —
  T consecutive lookback-window embeddings produced by the model encoder.
  Use ``WorldModelTrainer.encode_sequence(raw_seq, model)`` to produce these from raw
  feature windows ``[T, B, A, L, F]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor

from helion_risk_world.config.training_config import TrainingConfig
from helion_risk_world.encoders.option_surface_encoder import SurfaceTensors
from helion_risk_world.evaluation import world_model_metrics
from helion_risk_world.losses.rssm_loss import RSSMLoss, kl_balanced
from helion_risk_world.model import HRWWorldModel
from helion_risk_world.training.nan_guard import skip_if_non_finite
from helion_risk_world.training.trainer import resolve_device


class WorldModelTrainer:
    """Stage 3: RSSM dynamics training — latent world model z_t → z_{t+H} (SPEC.md §20)."""

    def __init__(self, model: HRWWorldModel, cfg: TrainingConfig, loss: RSSMLoss | None = None) -> None:
        self._model = model
        self._cfg = cfg
        self._loss = loss or RSSMLoss(model.market_world.rssm)
        self.history: list[float] = []
        self.n_skipped_batches: int = 0

    @staticmethod
    def encode_sequence(
        raw_seq: Tensor,
        model: HRWWorldModel,
        *,
        futures_seq: Tensor | None = None,
        regime_seq: Tensor | None = None,
        surface_seq: SurfaceTensors | None = None,
    ) -> Tensor:
        """Encode a raw feature sequence into RSSM-compatible embeddings.

        Args:
            raw_seq: [T, B, A, L, F]  — T consecutive lookback windows, one per bar.
            model:   HRWWorldModel with a trained (or pre-trained) encoder E_φ.
            futures_seq: optional [T, B, L, F_fut] futures microstructure windows.
            regime_seq: optional [T, B, K] regime-context rows.
            surface_seq: optional ``SurfaceTensors`` whose ``grid``/``mask``/``context``
                fields are each ``[T, B, ...]`` option-surface tensors (feature-onboarding
                pass).

        Returns:
            seq_e: [T, B, embed_dim] — pre-encoded embeddings for RSSMLoss.

        Each time-step is encoded independently so the RSSM can learn cross-bar dynamics.
        The TemporalEncoder captures within-window history; the RSSM learns between-window
        dynamics from the sequence.

        Moves inputs to ``model``'s own device before encoding: callers may invoke
        this after Stage-2 pretraining has already moved ``model`` off CPU (e.g.
        ``MarketStatePretrainer.fit()``, which calls ``model.to(device)``
        internally), while ``raw_seq``/``futures_seq``/``regime_seq``/``surface_seq`` —
        built directly from the original CPU-resident batches — would otherwise still be
        on CPU, crashing with a device-mismatch error inside the encoder's Linear
        layers.
        """
        device = next(model.parameters()).device
        raw_seq = raw_seq.to(device)
        if futures_seq is not None:
            futures_seq = futures_seq.to(device)
        if regime_seq is not None:
            regime_seq = regime_seq.to(device)
        if surface_seq is not None:
            surface_seq = SurfaceTensors(*(t.to(device) for t in surface_seq))
        T = raw_seq.shape[0]
        with torch.no_grad():
            return torch.stack(
                [
                    model.encode(
                        raw_seq[t],
                        futures_seq[t] if futures_seq is not None else None,
                        regime_seq[t] if regime_seq is not None else None,
                        SurfaceTensors(surface_seq.grid[t], surface_seq.mask[t], surface_seq.context[t])
                        if surface_seq is not None
                        else None,
                    )
                    for t in range(T)
                ],
                dim=0,
            )

    def fit(self, sequences: Sequence[Tensor], *, epochs: int | None = None) -> HRWWorldModel:
        """Train RSSM on pre-encoded sequences.

        Args:
            sequences: iterable of [T, B, embed_dim] tensors.  Each tensor is a
                       temporally-ordered sequence of encoder embeddings.
                       Use ``encode_sequence()`` to produce these from raw features.
            epochs:    override cfg.max_epochs if provided.
        """
        if not sequences:
            raise ValueError("fit requires a non-empty sequence collection")
        torch.manual_seed(self._cfg.seed)
        device = resolve_device(self._cfg.device)
        self._model.to(device).train()
        optim = torch.optim.Adam(
            self._model.market_world.rssm.parameters(),
            lr=self._cfg.lr,
            weight_decay=self._cfg.weight_decay,
        )
        n_epochs = epochs or self._cfg.max_epochs
        self.history = []
        self.n_skipped_batches = 0
        for epoch in range(n_epochs):
            running, total_mass = 0.0, 0.0
            accum_steps = self._cfg.grad_accum_steps
            optim.zero_grad()
            for idx, seq in enumerate(sequences, start=1):
                seq = seq.to(device)
                loss_dict = self._loss(seq)
                if skip_if_non_finite(
                    loss_dict["loss"], context=f"WorldModelTrainer.fit epoch={epoch + 1} idx={idx}"
                ):
                    self.n_skipped_batches += 1
                    optim.zero_grad()
                    continue
                (loss_dict["loss"] / accum_steps).backward()
                batch_mass = float(seq.shape[1])
                should_step = (idx % accum_steps == 0) or (idx == len(sequences))
                if should_step:
                    if self._cfg.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self._model.market_world.rssm.parameters(), self._cfg.grad_clip_norm
                        )
                    optim.step()
                    optim.zero_grad()
                running += float(loss_dict["loss"].detach()) * batch_mass
                total_mass += batch_mass
            self.history.append(running / max(total_mass, 1e-8))
        return self._model

    @torch.no_grad()
    def diagnostics(
        self, sequences: Sequence[Tensor], *, n_prior_samples: int = 16
    ) -> dict[str, float]:
        """Post-training RSSM health check (review finding M12).

        ``evaluation/world_model_metrics.py``'s KL-collapse and prior-predictive-
        coverage diagnostics were implemented and unit-tested, but no caller in
        the pipeline ever supplied real ``kl_per_step``/``prior_samples``/
        ``posterior_mean`` — so RSSM posterior collapse could occur silently.
        Call this after ``fit()`` to actually monitor it; wired into
        ``scripts/train.py``'s post-training logging.
        """
        device = resolve_device(self._cfg.device)
        rssm = self._model.market_world.rssm
        was_training = self._model.training
        self._model.eval()

        kl_per_step: list[float] = []
        prior_cov_samples: list[np.ndarray] = []
        posterior_means: list[np.ndarray] = []
        for seq in sequences:
            seq = seq.to(device)
            T, B, _ = seq.shape
            state = rssm.initial_state(B, device=seq.device)
            for t in range(T):
                state, post, prior = rssm.step_posterior(state, seq[t])
                kl_per_step.append(float(kl_balanced(post, prior)))
            # Prior-predictive coverage at the end of this sequence: does the
            # trained prior's ensemble span the posterior mean? (mean-reduced
            # across the stochastic dim to match world_model_metrics' scalar
            # per-sample interface.)
            prior_dist = rssm.prior(state.h)
            samples = prior_dist.sample((n_prior_samples,)).mean(dim=-1)  # [S, B]
            prior_cov_samples.append(samples.cpu().numpy())
            posterior_means.append(post.mean.mean(dim=-1).cpu().numpy())

        if was_training:
            self._model.train()

        prior_samples = np.concatenate(prior_cov_samples, axis=1) if prior_cov_samples else None
        posterior_mean = np.concatenate(posterior_means) if posterior_means else None
        return world_model_metrics.compute(
            kl_per_step=kl_per_step,
            prior_samples=prior_samples,
            posterior_mean=posterior_mean,
        )

    def run(self, cfg: Any) -> Any:
        return self.fit(cfg["sequences"], epochs=cfg.get("epochs"))
