"""Regime/event context encoder (SPEC.md §16).

Encodes the slow-moving market context vector (India VIX + percentile, IV summary, expiry/event
flags, blackout, FII/DII flow, global cues, one-hot event type) into a regime embedding. V1 is a
compact MLP. The feature vector is produced torch-free by
``data.regime_builder.featurize_regime`` so the layout lives in one place (DRY). SRP: regime/event
encoding only.
"""

from __future__ import annotations

from torch import Tensor, nn


class RegimeEncoder(nn.Module):
    """MLP over the regime/event feature vector. Input [B, K] -> [B, latent_dim]."""

    def __init__(self, n_features: int, latent_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self._n_features = n_features
        self.mlp = nn.Sequential(
            nn.Linear(n_features, latent_dim), nn.ReLU(), nn.Linear(latent_dim, latent_dim)
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError(f"RegimeEncoder expects [B, K]; got {tuple(x.shape)}")
        if x.shape[-1] != self._n_features:
            raise ValueError(f"feature dim {x.shape[-1]} != configured {self._n_features}")
        return self.norm(self.mlp(x))  # [B, latent_dim]


__all__ = ["RegimeEncoder"]
