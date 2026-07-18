"""Fusion encoder: combine plane embeddings into latent market state z_t (SPEC.md §16, Day 4).

V1 default is gated fusion. Inputs other than ``temporal`` are optional so the model is usable while
some plane encoders are absent for a given run. OCP: MoE / uncertainty-aware fusion can replace this
behind the same signature. SRP: fusion only.

Feature-onboarding pass: widened from 4 to 5 slots to give ``OptionSurfaceEncoder`` its own named
input. Previously the 4th slot was literally named ``surface`` but was fed the FUTURES embedding
(``model.py`` called ``fusion(temporal, cross=cross, surface=futures_emb, regime=regime_emb)``) --
there was no free slot for a genuine option-surface embedding. This bumps ``max_inputs`` to 5 and
resizes ``gate``/``candidate`` accordingly, which changes this module's parameter shapes (breaks
old checkpoint ``state_dict`` compatibility -- see ``training/artifacts.py``'s ``ARTIFACT_VERSION``).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class FusionEncoder(nn.Module):
    """Fuse temporal/cross-asset/futures/option-surface/regime embeddings into z_t.

    Inputs:  embeddings, each [B, d_*] (only ``temporal`` is required in V1)
    Output:  z_t [B, latent_dim]
    """

    def __init__(self, latent_dim: int = 128, method: str = "gated", max_inputs: int = 5) -> None:
        super().__init__()
        if method != "gated":
            raise NotImplementedError(f"fusion method {method!r} not implemented in V1")
        self.latent_dim = latent_dim
        self._method = method
        # Project a concatenation of up to ``max_inputs`` plane embeddings (each latent_dim) into
        # both a gate and a candidate, then down-project to latent_dim.
        self.gate = nn.Linear(latent_dim * max_inputs, latent_dim)
        self.candidate = nn.Linear(latent_dim * max_inputs, latent_dim)
        self.out = nn.Linear(latent_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)
        self._max_inputs = max_inputs

    def forward(
        self,
        temporal: Tensor,
        cross: Tensor | None = None,
        futures: Tensor | None = None,
        option_surface: Tensor | None = None,
        regime: Tensor | None = None,
    ) -> Tensor:
        parts = [p for p in (temporal, cross, futures, option_surface, regime) if p is not None]
        if not parts:
            raise ValueError("FusionEncoder requires at least the temporal embedding")
        b = parts[0].shape[0]
        # Zero-pad missing planes so the gate/candidate input width is fixed (stable parameters).
        zeros = torch.zeros(b, self.latent_dim, device=parts[0].device, dtype=parts[0].dtype)
        slots = [temporal, cross, futures, option_surface, regime]
        filled = [s if s is not None else zeros for s in slots][: self._max_inputs]
        concat = torch.cat(filled, dim=-1)               # [B, d * max_inputs]
        g = torch.sigmoid(self.gate(concat))             # [B, d]
        c = torch.tanh(self.candidate(concat))           # [B, d]
        z = self.out(g * c)                              # [B, d]
        return self.norm(z)                              # [B, latent_dim]


__all__ = ["FusionEncoder"]
