"""Barrier-hit head (SPEC.md §17). Which barrier is touched first within the horizon.

Emits 3-way logits over BARRIER_CLASSES = (stop_first, target_first, neither) — softmax gives
P(stop before target), P(target before stop), and P(neither hit in-horizon), which need NOT sum into
just the two trade-relevant ones (the "neither" mass matters for sizing). Trained with cross-entropy
against triple-barrier labels. ISP: knows nothing of brokers/planner.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

# Fixed class order; index 0 = stop hit first, 1 = target hit first, 2 = neither within horizon.
BARRIER_CLASSES: tuple[str, ...] = ("stop_first", "target_first", "neither")


class BarrierHead(nn.Module):
    """Emit barrier-hit logits. Input z: [B, d]. Output: [B, 3] (BARRIER_CLASSES order)."""

    def __init__(
        self,
        latent_dim: int = 128,
        context_dim: int = 0,
        *,
        normalize_input: bool = False,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.input_norm = nn.LayerNorm(latent_dim) if normalize_input else nn.Identity()
        self.context_gate = nn.Linear(context_dim, latent_dim) if context_dim > 0 else None
        if self.context_gate is not None:
            nn.init.zeros_(self.context_gate.weight)
            nn.init.zeros_(self.context_gate.bias)
        self.linear = nn.Linear(latent_dim, len(BARRIER_CLASSES))

    def forward(self, z: Tensor, context: Tensor | None = None) -> Tensor:
        if self.context_gate is not None and context is not None:
            z = z + torch.tanh(self.context_gate(context))
        return self.linear(self.input_norm(z))  # [B, 3] logits


__all__ = ["BarrierHead", "BARRIER_CLASSES"]
