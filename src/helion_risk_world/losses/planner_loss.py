from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor, nn


@runtime_checkable
class LossProtocol(Protocol):
    """Substitutable loss contract (SPEC.md §21, §26 LSP)."""

    def __call__(self, prediction: Any, target: Any) -> Tensor: ...


class PlannerLoss(nn.Module):
    """Planner objective with cost/CVaR/drawdown/exec penalties.

    SRP: one loss term only; composed in CompositeLoss.
    """

    def forward(self, prediction: Any, target: Any) -> Tensor:
        if isinstance(target, dict):
            reward = target.get("utility")
        else:
            reward = getattr(target, "utility", None)
        if reward is None:
            raise ValueError("PlannerLoss requires utility targets")
        reward_t = reward if isinstance(reward, Tensor) else torch.as_tensor(reward)
        score = prediction["utility"] if isinstance(prediction, dict) and "utility" in prediction else prediction
        return (score.reshape_as(reward_t) - reward_t).square().mean()
