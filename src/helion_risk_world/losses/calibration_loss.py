from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor, nn


@runtime_checkable
class LossProtocol(Protocol):
    """Substitutable loss contract (SPEC.md §21, §26 LSP)."""

    def __call__(self, prediction: Any, target: Any) -> Tensor: ...


class CalibrationLoss(nn.Module):
    """Reliability/coverage calibration penalty.

    SRP: one loss term only; composed in CompositeLoss.
    """

    def forward(self, prediction: Any, target: Any) -> Tensor:
        ret = target["forward_return"] if isinstance(target, dict) else getattr(target, "forward_return")
        quantiles = prediction["return_quantiles"]
        if quantiles.shape[1] < 2:
            return quantiles.new_zeros(())
        levels = torch.linspace(
            1.0 / (quantiles.shape[1] + 1),
            quantiles.shape[1] / (quantiles.shape[1] + 1),
            quantiles.shape[1],
            device=quantiles.device,
        )
        coverage = (ret.unsqueeze(-1) <= quantiles).float().mean(dim=0)
        return (coverage - levels).abs().mean()
