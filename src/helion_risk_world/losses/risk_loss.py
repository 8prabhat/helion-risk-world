from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor, nn


@runtime_checkable
class LossProtocol(Protocol):
    """Substitutable loss contract (SPEC.md §21, §26 LSP)."""

    def __call__(self, prediction: Any, target: Any) -> Tensor: ...


class RiskLoss(nn.Module):
    """Drawdown/CVaR-style risk penalty for training.

    SRP: one loss term only; composed in CompositeLoss.
    """

    def forward(self, prediction: Any, target: Any) -> Tensor:
        quantiles = prediction["return_quantiles"]
        downside = torch.relu(-quantiles[:, 0])
        return downside.mean()
