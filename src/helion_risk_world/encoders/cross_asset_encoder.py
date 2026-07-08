"""Cross-asset relation encoder (SPEC.md §16).

Learns relationships among the universe's assets (index ↔ sector ↔ constituent banks) so the model
can tell a *broad-based* move from a *concentrated* one. V1 is asset self-attention: each asset
gets a temporal summary, multi-head attention mixes information across assets, and residual + pool
collapses to a single cross-asset embedding. Order-invariant over assets (the pool is symmetric).
SRP: cross-asset relations only — temporal/option/regime live in their own encoders.
"""

from __future__ import annotations

from torch import Tensor, nn


class CrossAssetEncoder(nn.Module):
    """Asset self-attention over per-asset summaries. Input [B, A, L, F] -> [B, latent_dim]."""

    def __init__(self, n_features: int, latent_dim: int = 128, n_heads: int = 4) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self._n_features = n_features
        self.proj = nn.Linear(n_features, latent_dim)
        self.attn = nn.MultiheadAttention(latent_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"CrossAssetEncoder expects [B, A, L, F]; got {tuple(x.shape)}")
        if x.shape[-1] != self._n_features:
            raise ValueError(f"feature dim {x.shape[-1]} != configured {self._n_features}")
        summary = self.proj(x.mean(dim=2))        # per-asset temporal summary -> [B, A, d]
        mixed, _ = self.attn(summary, summary, summary)  # mix across assets -> [B, A, d]
        pooled = (summary + mixed).mean(dim=1)    # residual + symmetric pool over assets -> [B, d]
        return self.norm(pooled)                  # [B, latent_dim]


__all__ = ["CrossAssetEncoder"]
