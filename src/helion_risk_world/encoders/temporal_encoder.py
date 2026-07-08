"""Temporal candle/OI encoder (SPEC.md §16, Day 4).

Compact by design (Mac Studio 64GB): a per-feature projection + a small GRU over the lookback axis,
mean-pooled across assets. This is the V1 default; a patch/compact-SSM variant can be swapped in
behind the same ``EncoderProtocol`` (OCP) later. SRP: encodes temporal windows only — no
portfolio/broker knowledge.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor, nn


@runtime_checkable
class EncoderProtocol(Protocol):
    """Liskov-substitutable encoder contract (SPEC.md §16, §26)."""

    def forward(self, x: Tensor) -> Tensor:  # [B, ...] -> [B, d]
        ...


class TemporalEncoder(nn.Module):
    """Encode multi-asset candle/volume/return/OI history into a temporal embedding.

    Input:  ``x`` [B, A, L, F]  (batch, assets, lookback, features)
    Output: embedding [B, d_t]  (assets mean-pooled)
    """

    def __init__(self, n_features: int, latent_dim: int = 128, layers: int = 2,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self._n_features = n_features
        self.proj = nn.Linear(n_features, latent_dim)
        self.gru = nn.GRU(
            latent_dim,
            latent_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"TemporalEncoder expects [B, A, L, F]; got shape {tuple(x.shape)}")
        b, a, length, f = x.shape
        if f != self._n_features:
            raise ValueError(f"feature dim {f} != configured n_features {self._n_features}")
        flat = x.reshape(b * a, length, f)              # [B*A, L, F]
        h = self.proj(flat)                              # [B*A, L, d]
        out, _ = self.gru(h)                             # [B*A, L, d]
        last = out[:, -1, :]                             # [B*A, d] — final step
        per_asset = last.reshape(b, a, self.latent_dim)  # [B, A, d]
        pooled = per_asset.mean(dim=1)                    # [B, d] — assets mean-pooled
        return self.norm(pooled)                          # [B, d_t]


__all__ = ["EncoderProtocol", "TemporalEncoder"]
