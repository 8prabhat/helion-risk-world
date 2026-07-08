"""Differentiable quantile calibration regularizer.

Encourages empirical quantile coverage to match the target quantile levels using a soft
approximation to the indicator ``1[y <= q_tau]``. This complements pinball loss: pinball
fits conditional quantiles per sample, while this term penalizes batch-level coverage drift.
"""

from __future__ import annotations

import torch
from torch import Tensor


def soft_coverage_loss(
    prediction: Tensor,
    target: Tensor,
    quantile_levels: Tensor,
    *,
    sample_weight: Tensor | None = None,
    scale: Tensor | None = None,
    min_scale: float = 1e-4,
) -> Tensor:
    """Return a differentiable batch coverage penalty for predicted quantiles.

    Args:
        prediction: [B, Q] or [B, H, Q] predicted quantiles.
        target: [B] or [B, H] realized targets.
        quantile_levels: [Q] target quantile levels.
        sample_weight: optional [B] weights.
        scale: optional smoothing scale [B] or [B, H]. When omitted, uses detached predicted
            interquantile width.
    """
    if prediction.ndim not in {2, 3}:
        raise ValueError(f"prediction must be [B, Q] or [B, H, Q]; got {tuple(prediction.shape)}")
    if target.shape != prediction.shape[:-1]:
        raise ValueError(
            f"target shape {tuple(target.shape)} must match prediction.shape[:-1] {tuple(prediction.shape[:-1])}"
        )
    levels = quantile_levels.to(device=prediction.device, dtype=prediction.dtype)
    if levels.numel() != prediction.shape[-1]:
        raise ValueError(
            f"quantile_levels has {levels.numel()} entries, expected {prediction.shape[-1]}"
        )

    if scale is None:
        base_scale = (prediction[..., -1] - prediction[..., 0]).detach().abs()
    else:
        if scale.shape != target.shape:
            raise ValueError(f"scale shape {tuple(scale.shape)} must match target shape {tuple(target.shape)}")
        base_scale = scale.to(device=prediction.device, dtype=prediction.dtype).detach().abs()
    temperature = base_scale.clamp_min(min_scale).unsqueeze(-1)
    soft_hits = torch.sigmoid((prediction - target.unsqueeze(-1)) / temperature)

    if sample_weight is None:
        empirical = soft_hits.mean(dim=0)
    else:
        weights = sample_weight.to(device=prediction.device, dtype=prediction.dtype).reshape(-1)
        if weights.shape[0] != prediction.shape[0]:
            raise ValueError(
                f"sample_weight length {weights.shape[0]} must match batch size {prediction.shape[0]}"
            )
        reduce_shape = [weights.shape[0], *([1] * (soft_hits.ndim - 1))]
        weights_view = weights.reshape(*reduce_shape).clamp_min(0.0)
        empirical = (soft_hits * weights_view).sum(dim=0) / weights_view.sum(dim=0).clamp_min(1e-8)

    level_shape = [1] * (empirical.ndim - 1) + [levels.numel()]
    target_levels = levels.reshape(*level_shape)
    return torch.mean((empirical - target_levels) ** 2)


__all__ = ["soft_coverage_loss"]
