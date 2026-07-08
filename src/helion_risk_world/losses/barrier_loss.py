from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@runtime_checkable
class LossProtocol(Protocol):
    """Substitutable loss contract (SPEC.md §21, §26 LSP)."""

    def __call__(self, prediction: Any, target: Any) -> Tensor: ...


class BarrierLoss(nn.Module):
    """BCE for barrier-hit head. SRP: one loss term only; composed in CompositeLoss."""

    def forward(self, prediction: Any, target: Any) -> Tensor:
        logits = prediction["barrier_logits"]
        labels = target["barrier"] if isinstance(target, dict) else getattr(target, "barrier")
        if labels is None:
            raise ValueError("BarrierLoss requires barrier targets")
        weights = (
            target.get("barrier_weight")
            if isinstance(target, dict)
            else getattr(target, "barrier_weight", None)
        )
        per_sample = F.cross_entropy(logits, labels.reshape(-1), reduction="none")
        if weights is None:
            return per_sample.mean()
        weights = weights.reshape(-1).to(device=per_sample.device, dtype=per_sample.dtype).clamp_min(0.0)
        denom = weights.sum().clamp_min(1e-8)
        return torch.sum(per_sample * weights) / denom
