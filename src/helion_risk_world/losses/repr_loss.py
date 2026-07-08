"""VICReg representation pretraining loss (SPEC.md §14.4, §22).

L_repr = sim · smooth_l1(P(e_t), sg(e_{t+gap}))         # future-latent prediction (JEPA)
       + var · [var_hinge(P(e_t)) + var_hinge(e_{t+gap})]  # anti-collapse: std → 1 per dim
       + cov · [offdiag_cov(P(e_t)) + offdiag_cov(e_{t+gap})]  # decorrelate dimensions

Stage-2 pretraining on the full 2022→2026 spot/constituent history.
SRP: representation loss only (no RSSM dynamics).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _var_hinge(z: Tensor, gamma: float = 1.0, eps: float = 1e-4) -> Tensor:
    """Hinge loss encouraging std ≥ gamma per feature dimension.  [B, d] → scalar."""
    std = torch.sqrt(z.var(dim=0) + eps)           # [d]
    return torch.relu(gamma - std).mean()


def _offdiag_cov(z: Tensor, eps: float = 1e-4) -> Tensor:
    """Off-diagonal covariance penalty (decorrelate dimensions).  [B, d] → scalar."""
    N, d = z.shape
    z_centered = z - z.mean(dim=0, keepdim=True)
    cov = (z_centered.T @ z_centered) / (N - 1 + eps)  # [d, d]
    off = cov ** 2
    diag_mask = torch.eye(d, device=z.device, dtype=torch.bool)
    return off[~diag_mask].mean()


class VICRegLoss:
    """VICReg future-latent prediction loss for Stage-2 encoder pretraining (SPEC.md §22).

    predictor_out: [B, d]  — output of the prediction MLP applied to e_t  (P(e_t))
    target_e:      [B, d]  — encoded future observation e_{t+gap}  (stop-gradient target)
    """

    def __init__(
        self,
        sim: float = 1.0,
        var: float = 1.0,
        cov: float = 0.04,
        gamma: float = 1.0,
    ) -> None:
        self._sim = sim
        self._var = var
        self._cov = cov
        self._gamma = gamma

    def __call__(self, predictor_out: Tensor, target_e: Tensor) -> dict[str, Tensor]:
        """Returns {l_sim, l_var, l_cov, loss}."""
        t = target_e.detach()                        # stop-gradient on the target branch
        l_sim = F.smooth_l1_loss(predictor_out, t)
        l_var = _var_hinge(predictor_out, self._gamma) + _var_hinge(t, self._gamma)
        l_cov = _offdiag_cov(predictor_out) + _offdiag_cov(t)
        loss = self._sim * l_sim + self._var * l_var + self._cov * l_cov
        return {"l_sim": l_sim, "l_var": l_var, "l_cov": l_cov, "loss": loss}


__all__ = ["VICRegLoss"]
