"""Return quantile head (SPEC.md §17, Day 4).

Emits non-decreasing return quantiles by construction: the lowest quantile is unconstrained and each
higher quantile adds a non-negative (softplus) increment, so ``q10 <= q25 <= ... <= q90`` always
holds — no quantile-crossing. ISP: knows nothing of brokers/planner.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Default quantile levels (must match schemas.prediction_schema.QUANTILE_LEVELS).
DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)


def _inverse_softplus(value: float) -> float:
    if value <= 0.0:
        raise ValueError("value must be > 0 for inverse softplus")
    return math.log(math.expm1(value))


class ReturnQuantileHead(nn.Module):
    """Emit monotone return quantiles. Input z: [B, d]. Output: [B, Q]."""

    def __init__(
        self,
        latent_dim: int = 128,
        n_quantiles: int = len(DEFAULT_QUANTILES),
        *,
        init_center: float = 0.0,
        init_half_span: float = 0.003,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.n_quantiles = n_quantiles
        self.base = nn.Linear(latent_dim, 1)               # lowest quantile
        self.increments = nn.Linear(latent_dim, n_quantiles - 1)  # non-neg gaps
        nn.init.zeros_(self.base.weight)
        nn.init.constant_(self.base.bias, float(init_center - init_half_span))
        if n_quantiles > 1:
            nn.init.zeros_(self.increments.weight)
            step = max((2.0 * init_half_span) / (n_quantiles - 1), 1e-6)
            nn.init.constant_(self.increments.bias, _inverse_softplus(step))

    def forward(self, z: Tensor) -> Tensor:
        base = self.base(z)                                # [B, 1]
        gaps = F.softplus(self.increments(z))              # [B, Q-1] >= 0
        higher = base + torch.cumsum(gaps, dim=-1)         # [B, Q-1], each >= base
        return torch.cat([base, higher], dim=-1)           # [B, Q], non-decreasing


__all__ = ["ReturnQuantileHead", "DEFAULT_QUANTILES"]
