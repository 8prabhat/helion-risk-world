"""Direction head: stop-first or target-first, conditional on a barrier being touched at all.

Architecture change (2026-07-13), motivated directly by this project's own diagnostics: the
previous single 3-way barrier softmax forced two very different questions through one shared
latent bottleneck --
  (a) "will price move enough to touch either barrier?" (a magnitude/volatility question --
      the model's strongest, most robust known signal), and
  (b) "if it moves, which way?" (a direction question -- the option-surface features `pcr`/
      `max_pain_rel`/`iv_skew` were found to carry real, stable directional IC, up to ~0.43,
      but that signal was repeatedly observed NOT surviving the fusion encoder's shared
      bottleneck: `z` was measured collapsing to ~1 effective dimension under supervised
      training, and the option-surface plane specifically was shown to barely move the
      barrier head's predictions).
Splitting these into `TouchHead` (question a, over the fused `z`) and this `DirectionHead`
(question b) lets each be supervised by loss gradient aligned with the feature that actually
carries its signal. This head additionally takes a SKIP CONNECTION directly from the
option-surface embedding (bypassing the shared fusion bottleneck) rather than trusting that
signal to survive `z` alone -- the same fix pattern as a ResNet/multi-task skip connection:
give the sub-task a direct path to the input it specifically needs.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DirectionHead(nn.Module):
    """Emit a single stop-vs-target logit, conditional on touch. Inputs: z [B, d], surface
    embedding [B, d_surface] (zero-filled by the caller when no surface data is available,
    matching FusionEncoder's own missing-plane convention). Output: [B] (raw logit; positive
    favors target/up, negative favors stop/down)."""

    def __init__(self, latent_dim: int = 128, surface_dim: int | None = None, hidden_dim: int | None = None) -> None:
        super().__init__()
        surface_dim = surface_dim if surface_dim is not None else latent_dim
        hidden_dim = hidden_dim if hidden_dim is not None else latent_dim
        self.latent_dim = latent_dim
        self.surface_dim = surface_dim
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + surface_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: Tensor, surface_emb: Tensor) -> Tensor:
        if surface_emb.shape[0] != z.shape[0]:
            raise ValueError(
                f"surface_emb batch {surface_emb.shape[0]} != z batch {z.shape[0]}"
            )
        x = torch.cat([z, surface_emb], dim=-1)
        return self.mlp(x).squeeze(-1)  # [B]


__all__ = ["DirectionHead"]
