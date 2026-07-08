from __future__ import annotations

from torch import Tensor, nn


class DrawdownHead(nn.Module):
    """Emits drawdown probability.

    Input z: [B, d] (or [S, B, |H|, d] rollout). Output: [B, 1].
    ISP: a head knows nothing of brokers/planner (SPEC.md §17, §26).
    """

    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim

    def forward(self, z: Tensor) -> Tensor:
        raise NotImplementedError("DrawdownHead.forward — SPEC.md §17")
