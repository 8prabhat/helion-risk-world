from __future__ import annotations

import math
from typing import Any, Protocol, runtime_checkable

from torch import Tensor, nn


@runtime_checkable
class LossProtocol(Protocol):
    """Substitutable loss contract (SPEC.md §21, §26 LSP)."""

    def __call__(self, prediction: Any, target: Any) -> Tensor: ...


class UncertaintyLoss(nn.Module):
    """Uncertainty-aware (heteroscedastic) loss.

    SRP: one loss term only; composed in CompositeLoss.
    """

    def forward(self, prediction: Any, target: Any) -> Tensor:
        ret = target["forward_return"] if isinstance(target, dict) else getattr(target, "forward_return")
        quantiles = prediction["return_quantiles"]
        median = quantiles[:, quantiles.shape[1] // 2]
        sigma = prediction["uncertainty"].reshape(-1).clamp_min(1e-3)
        nll = 0.5 * (sigma.square().log() + math.log(2 * math.pi) + (ret - median).square() / sigma.square())
        return nll.mean()
