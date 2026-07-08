"""Fusion encoder: combine plane embeddings into latent market state z_t (SPEC.md §16, Day 4).

V1 default is gated fusion. Inputs other than ``temporal`` are optional so the model is usable while
the cross-asset / option-surface / regime encoders are still being wired (Day 4 uses temporal only).
OCP: MoE / uncertainty-aware fusion can replace this behind the same signature. SRP: fusion only.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class FusionEncoder(nn.Module):
    """Fuse temporal/cross-asset/option/regime embeddings into latent market state z_t.

    Inputs:  embeddings, each [B, d_*] (only ``temporal`` is required in V1)
    Output:  z_t [B, latent_dim]
    """

    def __init__(self, latent_dim: int = 128, method: str = "gated", max_inputs: int = 4) -> None:
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
        surface: Tensor | None = None,
        regime: Tensor | None = None,
    ) -> Tensor:
        parts = [p for p in (temporal, cross, surface, regime) if p is not None]
        if not parts:
            raise ValueError("FusionEncoder requires at least the temporal embedding")
        b = parts[0].shape[0]
        # Zero-pad missing planes so the gate/candidate input width is fixed (stable parameters).
        zeros = torch.zeros(b, self.latent_dim, device=parts[0].device, dtype=parts[0].dtype)
        slots = [temporal, cross, surface, regime]
        filled = [s if s is not None else zeros for s in slots][: self._max_inputs]
        concat = torch.cat(filled, dim=-1)               # [B, d * max_inputs]
        g = torch.sigmoid(self.gate(concat))             # [B, d]
        c = torch.tanh(self.candidate(concat))           # [B, d]
        z = self.out(g * c)                              # [B, d]
        return self.norm(z)                              # [B, latent_dim]


__all__ = ["FusionEncoder"]
