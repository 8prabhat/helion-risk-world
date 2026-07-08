"""Excursion head. Predicts non-negative path excursions such as MAE or MFE."""

from __future__ import annotations

import math

import torch.nn.functional as F
from torch import Tensor, nn


def _inverse_softplus(value: float) -> float:
    if value <= 0.0:
        raise ValueError("value must be > 0 for inverse softplus")
    return math.log(math.expm1(value))


class ExcursionHead(nn.Module):
    """Emit a non-negative excursion scalar from a latent state."""

    def __init__(self, latent_dim: int = 128, *, init_value: float = 0.002) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.linear = nn.Linear(latent_dim, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, _inverse_softplus(init_value))

    def forward(self, z: Tensor) -> Tensor:
        return F.softplus(self.linear(z)).squeeze(-1)


__all__ = ["ExcursionHead"]
