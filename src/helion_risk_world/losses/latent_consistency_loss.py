"""Latent-prediction / latent-consistency loss (SPEC.md §13, §20.2, §20.4).

The self-supervised objective that makes HRW an actual world model: predict the FUTURE latent from
the present one and require the prediction to match the (encoded) future state. Used by:

  * Stage 2 (``MarketStatePretrainer``) — future-latent prediction to pretrain the encoder;
  * Stage 4 (``WorldModelTrainer``) — latent-consistency, training the dynamics ``f_theta`` to match
    the next encoded latent.

A bare prediction loss collapses (the encoder can map everything to a constant and predict it
perfectly). We therefore use a VICReg-style objective (Bardes et al. 2022): an **invariance** term
(prediction matches target) plus **variance** (each latent dim keeps spread — anti-collapse) and
**covariance** (decorrelate dims) regularisers on the embeddings. No reconstruction, no EMA needed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_EPS = 1e-4


def _variance_term(z: Tensor, gamma: float = 1.0) -> Tensor:
    """Hinge that keeps per-dimension std near ``gamma`` (prevents collapse to a point)."""
    std = torch.sqrt(z.var(dim=0) + _EPS)          # [D]
    return torch.relu(gamma - std).mean()


def _covariance_term(z: Tensor) -> Tensor:
    """Off-diagonal covariance magnitude (decorrelates latent dimensions)."""
    n, d = z.shape
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / max(n - 1, 1)              # [D, D]
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).sum() / d


class LatentPredictionLoss(nn.Module):
    """VICReg-style future-latent prediction loss.

    forward(prediction, target):
        prediction: [B, D] predicted future latent (from the predictor / dynamics)
        target:     [B, D] encoded future latent (the self-supervised target)
    Per-term values are stashed on ``self.last_components``.
    """

    def __init__(self, sim: float = 25.0, var: float = 25.0, cov: float = 1.0) -> None:
        super().__init__()
        self._sim, self._var, self._cov = sim, var, cov
        self.last_components: dict[str, float] = {}

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        if prediction.shape != target.shape or prediction.ndim != 2:
            raise ValueError(
                f"prediction/target must be matching [B, D]; got {tuple(prediction.shape)} "
                f"and {tuple(target.shape)}"
            )
        invariance = F.smooth_l1_loss(prediction, target)
        variance = _variance_term(prediction) + _variance_term(target)
        covariance = _covariance_term(prediction) + _covariance_term(target)
        total = self._sim * invariance + self._var * variance + self._cov * covariance
        self.last_components = {
            "invariance": float(invariance.detach()),
            "variance": float(variance.detach()),
            "covariance": float(covariance.detach()),
            "total": float(total.detach()),
        }
        return total


__all__ = ["LatentPredictionLoss"]
