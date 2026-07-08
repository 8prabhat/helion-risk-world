"""OOD head (SPEC.md §17, §23 innovation 8). Out-of-distribution score for the latent state z_t.

OOD has no labels, so this is a FITTED detector rather than a trained logit: it fits a diagonal
Gaussian over the training latents and scores new latents by their (diagonal) Mahalanobis distance,
calibrated against the training distribution so the score lands in [0, 1]. In-distribution states
score low; states far from the training support score high — which is exactly what
the Risk Shield's ``OODRule`` quarantines. Implemented as an ``nn.Module`` with buffers so it
serialises and moves with the model. ISP: knows nothing of brokers/planner.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

_EPS = 1e-6


class OODHead(nn.Module):
    """Fitted diagonal-Gaussian OOD detector. ``fit`` on training z, ``forward`` scores new z."""

    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.register_buffer("_mean", torch.zeros(latent_dim))
        self.register_buffer("_std", torch.ones(latent_dim))
        self.register_buffer("_boundary", torch.tensor(0.0))
        self.register_buffer("_scale", torch.tensor(1.0))
        self.register_buffer("_fitted", torch.tensor(0.0))

    @torch.no_grad()
    def fit(self, latents: Tensor, boundary_quantile: float = 0.975) -> OODHead:
        """Fit the detector to training latents [N, d]. Returns self."""
        if latents.ndim != 2:
            raise ValueError(f"latents must be [N, d]; got {tuple(latents.shape)}")
        mean = latents.mean(dim=0)
        std = latents.std(dim=0).clamp_min(_EPS)
        raw = (((latents - mean) / std) ** 2).mean(dim=1)         # [N] diagonal Mahalanobis
        boundary = torch.quantile(raw, boundary_quantile)
        scale = (boundary - raw.median()).clamp_min(_EPS)
        self._mean.copy_(mean)
        self._std.copy_(std)
        self._boundary.copy_(boundary)
        self._scale.copy_(scale)
        self._fitted.fill_(1.0)
        return self

    def score(self, z: Tensor) -> Tensor:
        """OOD score in [0, 1] for z [B, d]. Returns zeros if the detector is unfitted."""
        if float(self._fitted) < 0.5:
            return torch.zeros(z.shape[0], device=z.device)
        raw = (((z - self._mean) / self._std) ** 2).mean(dim=1)   # [B]
        return torch.sigmoid((raw - self._boundary) / self._scale)

    def forward(self, z: Tensor) -> Tensor:
        return self.score(z).unsqueeze(-1)  # [B, 1]


__all__ = ["OODHead"]
