"""Barrier logits derived from barrier-relative excursion geometry."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ExcursionBarrierHead(nn.Module):
    """Map excursion ratios to stop/target/timeout logits.

    Input rows are expected to be ``[stop_ratio, target_ratio, volatility_ratio]`` where
    ``stop_ratio = predicted_mae / |stop_return|`` and
    ``target_ratio = predicted_mfe / target_return``.

    The timeout class is derived from whether both excursion ratios stay below 1.0.
    """

    def __init__(self, *, init_temperature: float = 1.0) -> None:
        super().__init__()
        # 4 inputs: [stop_ratio-1, target_ratio-1, timeout_margin, volatility_ratio-1]
        # (review finding M2: volatility_ratio used to be validated as a required
        # [B, 3] input but never actually read in forward()).
        self.linear = nn.Linear(4, 3)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.zero_()
            self.linear.weight[0, 0] = 1.0
            self.linear.weight[1, 1] = 1.0
            self.linear.weight[2, 2] = 1.0
            # weight[:, 3] (volatility_ratio's column) stays 0 at init, so the head
            # starts out identical to before this fix (raw excursion ratios ARE the
            # logits) and only learns a real volatility-ratio effect during training.
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(init_temperature))))

    def forward(self, ratios: Tensor) -> Tensor:
        if ratios.ndim != 2 or ratios.shape[-1] != 3:
            raise ValueError(f"ratios must be [B, 3]; got {tuple(ratios.shape)}")
        stop_ratio = ratios[:, 0]
        target_ratio = ratios[:, 1]
        volatility_ratio = ratios[:, 2]
        timeout_margin = 1.0 - torch.maximum(stop_ratio, target_ratio)
        features = torch.stack(
            [
                stop_ratio - 1.0,
                target_ratio - 1.0,
                timeout_margin,
                volatility_ratio - 1.0,
            ],
            dim=-1,
        )
        temperature = self.log_temperature.exp().clamp(1.0, 50.0)
        return self.linear(features) * temperature


__all__ = ["ExcursionBarrierHead"]
