"""Aleatoric uncertainty head for the compact forecaster path.

The deterministic forecaster can support a learned aleatoric scale, but not a distinct learned
epistemic output honestly. World-model epistemic uncertainty comes from RSSM rollout spread instead.
ISP: knows nothing of brokers.
"""

from __future__ import annotations

import math

import torch.nn.functional as F
from torch import Tensor, nn


def _inverse_softplus(value: float) -> float:
    if value <= 0.0:
        raise ValueError("value must be > 0 for inverse softplus")
    return math.log(math.expm1(value))


class UncertaintyHead(nn.Module):
    """Emit aleatoric scale >= 0. Input z: [B, d]. Output: [B]."""

    def __init__(
        self,
        latent_dim: int = 128,
        min_scale: float = 1e-3,
        *,
        init_value: float = 0.005,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self._min_scale = min_scale
        self.linear = nn.Linear(latent_dim, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, _inverse_softplus(max(init_value - min_scale, 1e-6)))

    def forward(self, z: Tensor) -> Tensor:
        return (F.softplus(self.linear(z)) + self._min_scale).squeeze(-1)  # [B] >= min_scale


__all__ = ["UncertaintyHead"]
