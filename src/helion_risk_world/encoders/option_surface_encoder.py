"""Option-surface encoder (SPEC.md §16, derivatives-awareness).

A permutation-invariant DeepSets encoder over the ATM-relative strike tokens — NOT a naive
flatten of hundreds of static columns. Each strike row is embedded by a shared MLP,
masked-mean-pooled across strikes (so missing/illiquid strikes are ignored, not zero-polluting),
then combined with a snapshot-level context vector (PCR, IV skew, walls, max-pain, ...). SRP:
surface encoding only.

Tensorisation lives in ``data.option_surface_builder.featurize_surface`` (torch-free); this module
consumes the resulting tensors so feature definitions stay in one place (DRY).
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn


class SurfaceTensors(NamedTuple):
    """Model-ready option-surface tensors (batched)."""

    grid: Tensor      # [B, S, C]
    mask: Tensor      # [B, S]
    context: Tensor   # [B, K]


class OptionSurfaceEncoder(nn.Module):
    """DeepSets encoder over ATM-relative strikes + snapshot context (SPEC.md §16)."""

    def __init__(self, n_channels: int, n_context: int, latent_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self._n_channels = n_channels
        self._n_context = n_context
        self.phi = nn.Sequential(
            nn.Linear(n_channels, latent_dim), nn.ReLU(), nn.Linear(latent_dim, latent_dim)
        )
        self.context = nn.Sequential(nn.Linear(n_context, latent_dim), nn.ReLU())
        self.out = nn.Linear(latent_dim * 2, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, surface: SurfaceTensors) -> Tensor:
        grid, mask, context = surface
        if grid.ndim != 3:
            raise ValueError(f"grid must be [B, S, C]; got {tuple(grid.shape)}")
        if grid.shape[-1] != self._n_channels:
            raise ValueError(f"grid channels {grid.shape[-1]} != configured {self._n_channels}")
        h = self.phi(grid)                                        # [B, S, d]
        m = mask.unsqueeze(-1)                                    # [B, S, 1]
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)  # masked mean -> [B, d]
        c = self.context(context)                                # [B, d]
        z = self.out(torch.cat([pooled, c], dim=-1))             # [B, d]
        return self.norm(z)                                       # [B, latent_dim]


__all__ = ["OptionSurfaceEncoder", "SurfaceTensors"]
