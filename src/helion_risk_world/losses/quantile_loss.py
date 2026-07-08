"""Quantile (pinball) loss for the return distribution (SPEC.md §21, Day 4).

SRP: one loss term; composed in CompositeLoss. The pinball loss for level ``tau`` penalises
under-prediction by ``tau`` and over-prediction by ``1 - tau``, so minimising it fits the ``tau``
conditional quantile.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor, nn

DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)


@runtime_checkable
class LossProtocol(Protocol):
    """Substitutable loss contract (SPEC.md §21, §26 LSP)."""

    def __call__(self, prediction: Any, target: Any) -> Tensor: ...


class QuantileLoss(nn.Module):
    """Pinball loss across quantile levels.

    prediction: [B, Q] predicted quantiles (one column per level)
    target:     [B] or [B, 1] realised value
    returns:    scalar mean pinball loss
    """

    def __init__(self, quantiles: tuple[float, ...] = DEFAULT_QUANTILES) -> None:
        super().__init__()
        if not all(0.0 < q < 1.0 for q in quantiles):
            raise ValueError("quantile levels must lie in (0, 1)")
        self.register_buffer("_levels", torch.tensor(quantiles, dtype=torch.float32))

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        if prediction.ndim != 2:
            raise ValueError(f"prediction must be [B, Q]; got {tuple(prediction.shape)}")
        if prediction.shape[1] != self._levels.numel():
            raise ValueError(
                f"prediction has {prediction.shape[1]} quantiles, expected {self._levels.numel()}"
            )
        target = target.reshape(-1, 1)                       # [B, 1]
        levels = self._levels.to(device=prediction.device, dtype=prediction.dtype).reshape(1, -1)
        error = target - prediction                           # [B, Q]
        loss = torch.maximum(levels * error, (levels - 1.0) * error)  # pinball
        return loss.mean()


__all__ = ["QuantileLoss", "LossProtocol", "DEFAULT_QUANTILES"]
