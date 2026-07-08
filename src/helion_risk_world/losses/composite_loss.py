"""Composite losses (SPEC.md §21, Day 4).

``ForecasterLoss`` is the concrete Day-4 composite for ``HRWForecaster``: a weighted sum of the
return quantile (pinball) loss and a Gaussian NLL that ties the aleatoric uncertainty head to the
median return. ``CompositeLoss`` remains the future generic, registry-driven weighted-sum that will
fold in volatility/barrier/regime/calibration/OOD terms.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from helion_risk_world.config.model_config import LossWeights
from helion_risk_world.losses.quantile_calibration import soft_coverage_loss
from helion_risk_world.losses.quantile_loss import DEFAULT_QUANTILES, QuantileLoss

_DIRECTION_BAND = 1e-3
_DIRECTION_SCALE_FLOOR = 5e-4
_BARRIER_GEOMETRY_FLOOR = 1e-6
_EXCURSION_EDGE_NEUTRAL_BAND = 0.75


class ForecasterLoss(nn.Module):
    """Weighted quantile + uncertainty (+ optional auxiliaries) loss for the forecaster.

    prediction: model output dict with ``return_quantiles`` [B, Q],
                ``uncertainty`` [B], and (optionally) ``regime_logits`` [B, 6]
    target:     object/dict with ``forward_return`` [B], ``direction`` [B] (0/1/2), and an optional
                ``regime`` [B] (0..5). The regime CE term is added only when both are present.
    returns:    scalar total loss. Per-term values are stashed on ``self.last_components``.
    """

    def __init__(
        self,
        weights: LossWeights | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> None:
        super().__init__()
        self._w = weights or LossWeights()
        self._quantile = QuantileLoss(quantiles)
        self._median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
        self.register_buffer("_levels", torch.tensor(quantiles, dtype=torch.float32))
        class_w = self._w.barrier_class_weights or (1.0, 1.0, 1.0)
        self.register_buffer("_barrier_class_weight", torch.tensor(class_w, dtype=torch.float32))
        self.last_components: dict[str, float] = {}

    @staticmethod
    def _get(target: Any, key: str, default: Any = None) -> Any:
        if isinstance(target, dict):
            return target.get(key, default)
        return getattr(target, key, default)

    def forward(self, prediction: dict[str, Tensor], target: Any) -> Tensor:
        ret = self._get(target, "forward_return").reshape(-1)
        sample_weight = self._get(target, "sample_weight")
        weights = sample_weight.reshape(-1) if sample_weight is not None else None
        return_weight = self._get(target, "return_weight")
        return_weights = weights
        if return_weight is not None:
            return_weight = return_weight.reshape(-1)
            return_weights = (
                return_weight
                if return_weights is None
                else return_weights * return_weight
            )
        barrier_context = self._get(target, "barrier_context")
        geometry = (
            _barrier_geometry(barrier_context, device=ret.device, dtype=ret.dtype)
            if barrier_context is not None
            else None
        )

        q_loss = self._reduce(
            self._quantile_per_sample(prediction["return_quantiles"], ret),
            return_weights,
        )
        calibration_scale = self._get(target, "realized_vol")
        c_loss = soft_coverage_loss(
            prediction["return_quantiles"],
            ret,
            self._levels,
            sample_weight=return_weights,
            scale=(
                calibration_scale.reshape(-1)
                if calibration_scale is not None
                else prediction["uncertainty"].reshape(-1).detach()
            ),
        )

        median = prediction["return_quantiles"][:, self._median_idx]
        sigma = prediction["uncertainty"].reshape(-1).clamp_min(1e-3)  # aleatoric scale
        nll = 0.5 * (torch.log(2 * math.pi * sigma**2) + (ret - median) ** 2 / sigma**2)
        u_loss = self._reduce(nll, return_weights)

        total = (
            self._w.return_ * q_loss
            + self._w.calibration * c_loss
            + self._w.uncertainty * u_loss
        )
        self.last_components = {
            "quantile": float(q_loss.detach()),
            "calibration": float(c_loss.detach()),
            "uncertainty": float(u_loss.detach()),
        }

        # Direction — optional: only applied if the model still has a direction head.
        direction = self._get(target, "direction")
        if direction is not None:
            if "direction_logits" in prediction:
                d_loss = self._reduce(
                    F.cross_entropy(
                        prediction["direction_logits"],
                        direction.reshape(-1),
                        reduction="none",
                    ),
                    weights,
                )
            else:
                d_loss = self._reduce(
                    _direction_surrogate_loss(
                        median,
                        direction.reshape(-1),
                        scale=sigma.detach(),
                    ),
                    return_weights,
                )
            total = total + self._w.direction * d_loss
            self.last_components["direction"] = float(d_loss.detach())

        # Optional auxiliary terms — each active only when both head output and target are present.
        regime = self._get(target, "regime")
        if regime is not None and "regime_logits" in prediction:
            r_loss = self._reduce(
                F.cross_entropy(
                    prediction["regime_logits"],
                    regime.reshape(-1),
                    reduction="none",
                ),
                weights,
            )
            total = total + self._w.regime * r_loss
            self.last_components["regime"] = float(r_loss.detach())

        realized_vol = self._get(target, "realized_vol")
        if realized_vol is not None and "volatility" in prediction:
            # Loss computed in vol-RATIO space (both sides divided by the causal EWMA baseline
            # vol_baseline, i.e. barrier_sigma) when available: OOS diagnostics show forward vol
            # LEVEL is unstable to fit (regime drift in the overall vol scale over the sample
            # dominates the loss) while the RATIO to a trailing baseline is genuinely learnable
            # (positive OOS correlation in every walk-forward fold, 0.34-0.65 depending on
            # horizon). The head itself still outputs/is supervised toward real vol units --
            # only the loss's error metric is rescaled, so all downstream consumers (CVaR,
            # quantile calibration scale) keep receiving an absolute-vol-unit prediction.
            vol_baseline = self._get(target, "vol_baseline")
            if vol_baseline is not None:
                vb = vol_baseline.reshape(-1).clamp_min(1e-6)
                pred_vol = prediction["volatility"].reshape(-1) / vb
                target_vol = realized_vol.reshape(-1) / vb
            else:
                pred_vol = prediction["volatility"].reshape(-1)
                target_vol = realized_vol.reshape(-1)
            v_loss = self._reduce(
                F.smooth_l1_loss(pred_vol, target_vol, reduction="none"),
                weights,
            )
            total = total + self._w.volatility * v_loss
            self.last_components["volatility"] = float(v_loss.detach())

        mae = self._get(target, "mae")
        if mae is not None and "mae" in prediction:
            mae_loss = self._reduce(
                F.smooth_l1_loss(
                    prediction["mae"].reshape(-1),
                    mae.reshape(-1),
                    reduction="none",
                ),
                weights,
            )
            total = total + self._w.mae * mae_loss
            self.last_components["mae"] = float(mae_loss.detach())

        mfe = self._get(target, "mfe")
        if mfe is not None and "mfe" in prediction:
            mfe_loss = self._reduce(
                F.smooth_l1_loss(
                    prediction["mfe"].reshape(-1),
                    mfe.reshape(-1),
                    reduction="none",
                ),
                weights,
            )
            total = total + self._w.mfe * mfe_loss
            self.last_components["mfe"] = float(mfe_loss.detach())

        if mae is not None and mfe is not None and "mae" in prediction and "mfe" in prediction:
            coherence_weight = 0.25 * (self._w.mae + self._w.mfe)
            if coherence_weight != 0.0:
                coherence_loss = self._reduce(
                    _excursion_coherence_per_sample(
                        prediction["return_quantiles"],
                        prediction["mae"].reshape(-1),
                        prediction["mfe"].reshape(-1),
                    ),
                    return_weights,
                )
                total = total + coherence_weight * coherence_loss
                self.last_components["excursion_coherence"] = float(coherence_loss.detach())

        if (
            geometry is not None
            and mae is not None
            and mfe is not None
            and "barrier_logits" in prediction
        ):
            _, stop_scale, target_scale = geometry
            excursion_barrier = _excursion_barrier_labels(
                mae.reshape(-1),
                mfe.reshape(-1),
                stop_scale,
                target_scale,
            )
            barrier_weight = self._get(target, "barrier_weight")
            effective_weights = weights
            if barrier_weight is not None:
                barrier_weight = barrier_weight.reshape(-1)
                effective_weights = (
                    barrier_weight
                    if effective_weights is None
                    else effective_weights * barrier_weight
                )
            excursion_barrier_loss = self._reduce(
                F.cross_entropy(
                    prediction["barrier_logits"],
                    excursion_barrier,
                    weight=self._barrier_class_weight.to(
                        device=prediction["barrier_logits"].device,
                        dtype=prediction["barrier_logits"].dtype,
                    ),
                    reduction="none",
                ),
                effective_weights,
            )
            aux_weight = 0.25 * self._w.barrier
            if aux_weight != 0.0:
                total = total + aux_weight * excursion_barrier_loss
            self.last_components["excursion_barrier"] = float(excursion_barrier_loss.detach())

        barrier = self._get(target, "barrier")
        if barrier is not None and "barrier_logits" in prediction:
            barrier_weight = self._get(target, "barrier_weight")
            effective_weights = weights
            if barrier_weight is not None:
                barrier_weight = barrier_weight.reshape(-1)
                effective_weights = (
                    barrier_weight
                    if effective_weights is None
                    else effective_weights * barrier_weight
                )
            b_loss = self._reduce(
                F.cross_entropy(
                    prediction["barrier_logits"],
                    barrier.reshape(-1),
                    weight=self._barrier_class_weight.to(
                        device=prediction["barrier_logits"].device,
                        dtype=prediction["barrier_logits"].dtype,
                    ),
                    reduction="none",
                ),
                effective_weights,
            )
            total = total + self._w.barrier * b_loss
            self.last_components["barrier"] = float(b_loss.detach())

        self.last_components["total"] = float(total.detach())
        return total

    def _quantile_per_sample(self, prediction: Tensor, target: Tensor) -> Tensor:
        levels = self._levels.to(device=prediction.device, dtype=prediction.dtype).reshape(1, -1)
        error = target.reshape(-1, 1) - prediction
        loss = torch.maximum(levels * error, (levels - 1.0) * error)
        return loss.mean(dim=1)

    @staticmethod
    def _reduce(values: Tensor, weights: Tensor | None) -> Tensor:
        values = values.reshape(-1)
        if weights is None:
            return values.mean()
        weights = weights.to(device=values.device, dtype=values.dtype).reshape(-1).clamp_min(0.0)
        denom = weights.sum().clamp_min(1e-8)
        return torch.sum(values * weights) / denom


class CompositeLoss(nn.Module):
    """Config-driven weighted sum of loss terms (SPEC.md §21). FUTURE generic version.

    OCP: add a term without editing callers. Day 4 uses ``ForecasterLoss``; the full registry-driven
    composite (volatility/barrier/regime/calibration/OOD) lands with those heads.

    total = w_return*quantile + w_direction*direction + w_volatility*vol + w_barrier*barrier
          + w_regime*regime + w_calibration*calib + w_uncertainty*unc + w_ood*ood
    """

    def __init__(self, weights: LossWeights, terms: dict[str, nn.Module]) -> None:
        super().__init__()
        self._weights = weights
        self._terms = nn.ModuleDict(terms)
        self.last_components: dict[str, float] = {}

    def forward(self, prediction: Any, target: Any) -> Tensor:
        total: Tensor | None = None
        components: dict[str, float] = {}
        for name, term in self._terms.items():
            key = "return_" if name == "return" else name
            weight = getattr(self._weights, key, None)
            if weight is None or weight == 0:
                continue
            value = term(prediction, target)
            total = value * weight if total is None else total + value * weight
            components[name] = float(value.detach())
        if total is None:
            raise ValueError("CompositeLoss requires at least one active weighted term")
        components["total"] = float(total.detach())
        self.last_components = components
        return total


__all__ = ["ForecasterLoss", "CompositeLoss"]


def _direction_labels_from_returns(
    returns: Tensor,
    *,
    band: float = _DIRECTION_BAND,
) -> Tensor:
    thresholds = torch.tensor(
        [-band, band],
        device=returns.device,
        dtype=returns.dtype,
    )
    return torch.bucketize(returns, thresholds)


def _barrier_geometry(
    barrier_context: Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor]:
    if barrier_context.ndim != 2 or barrier_context.shape[1] != 3:
        raise ValueError(f"barrier_context must be [N, 3]; got {tuple(barrier_context.shape)}")
    sigma = barrier_context[:, 0].to(device=device, dtype=dtype).clamp_min(_BARRIER_GEOMETRY_FLOOR)
    stop = barrier_context[:, 1].to(device=device, dtype=dtype).abs().clamp_min(_BARRIER_GEOMETRY_FLOOR)
    target = barrier_context[:, 2].to(device=device, dtype=dtype).abs().clamp_min(_BARRIER_GEOMETRY_FLOOR)
    return sigma, stop, target


def _excursion_barrier_labels(
    mae: Tensor,
    mfe: Tensor,
    stop_scale: Tensor,
    target_scale: Tensor,
    *,
    neutral_band: float = _EXCURSION_EDGE_NEUTRAL_BAND,
) -> Tensor:
    edge = mfe.reshape(-1) / target_scale.reshape(-1) - mae.reshape(-1) / stop_scale.reshape(-1)
    labels = torch.full(edge.shape, 2, dtype=torch.long, device=edge.device)
    labels = torch.where(edge <= -neutral_band, torch.zeros_like(labels), labels)
    labels = torch.where(edge >= neutral_band, torch.ones_like(labels), labels)
    return labels


def _excursion_coherence_per_sample(
    return_quantiles: Tensor,
    mae: Tensor,
    mfe: Tensor,
) -> Tensor:
    if return_quantiles.ndim != 2:
        raise ValueError(
            "return_quantiles must be [B, Q] for forecaster excursion coherence; "
            f"got {tuple(return_quantiles.shape)}"
        )
    upper = torch.relu(return_quantiles - mfe.unsqueeze(-1))
    lower = torch.relu(-return_quantiles - mae.unsqueeze(-1))
    return (upper + lower).mean(dim=-1)


def _direction_surrogate_loss(
    median: Tensor,
    direction: Tensor,
    *,
    scale: Tensor | None = None,
    band: float = _DIRECTION_BAND,
) -> Tensor:
    direction = direction.to(device=median.device).reshape(-1)
    median = median.reshape(-1)
    margin_scale = (
        scale.to(device=median.device, dtype=median.dtype).reshape(-1).clamp_min(_DIRECTION_SCALE_FLOOR)
        if scale is not None
        else torch.full_like(median, _DIRECTION_SCALE_FLOOR)
    )
    losses = torch.empty_like(median)
    down = direction == 0
    flat = direction == 1
    up = direction == 2
    if bool(down.any()):
        losses[down] = F.softplus((band + median[down]) / margin_scale[down])
    if bool(flat.any()):
        losses[flat] = F.softplus((median[flat].abs() - band) / margin_scale[flat])
    if bool(up.any()):
        losses[up] = F.softplus((band - median[up]) / margin_scale[up])
    return losses
