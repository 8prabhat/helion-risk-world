"""Supervised head loss for the multi-horizon world model."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from helion_risk_world.config.model_config import LossWeights
from helion_risk_world.losses.composite_loss import (
    _barrier_geometry,
    _direction_labels_from_returns,
    _direction_surrogate_loss,
    _excursion_coherence_per_sample,
    _excursion_barrier_labels,
)
from helion_risk_world.losses.quantile_calibration import soft_coverage_loss
from helion_risk_world.losses.quantile_loss import DEFAULT_QUANTILES
from helion_risk_world.losses.repr_loss import _offdiag_cov, _var_hinge


class WorldModelLoss(nn.Module):
    """Train multi-horizon return/vol heads plus management-horizon auxiliaries."""

    def __init__(
        self,
        *,
        weights: LossWeights | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> None:
        super().__init__()
        if 0.5 not in quantiles:
            raise ValueError("quantiles must include the 0.5 median")
        self._weights = weights or LossWeights()
        self.register_buffer("_levels", torch.tensor(quantiles, dtype=torch.float32))
        class_w = self._weights.barrier_class_weights or (1.0, 1.0, 1.0)
        self.register_buffer("_barrier_class_weight", torch.tensor(class_w, dtype=torch.float32))
        self.last_components: dict[str, float] = {}

    @staticmethod
    def _get(target: Any, key: str, default: Any = None) -> Any:
        if isinstance(target, dict):
            return target.get(key, default)
        return getattr(target, key, default)

    def forward(self, prediction: dict[str, Tensor | tuple[int, ...]], target: Any) -> Tensor:
        returns = self._get(target, "horizon_returns")
        if returns is None:
            raise ValueError("world-model training requires horizon_returns targets")
        vols = self._get(target, "horizon_volatility")
        if vols is None:
            raise ValueError("world-model training requires horizon_volatility targets")
        target_horizons = tuple(int(h) for h in self._get(target, "target_horizons", ()))
        pred_horizons = tuple(int(h) for h in prediction.get("horizons", ()))
        if target_horizons and pred_horizons and target_horizons != pred_horizons:
            raise ValueError(
                f"world-model target horizons {target_horizons} do not match model horizons {pred_horizons}"
            )

        sample_weight = self._get(target, "sample_weight")
        weights = sample_weight.reshape(-1) if sample_weight is not None else None
        barrier_context = self._get(target, "barrier_context")
        geometry = (
            _barrier_geometry(barrier_context, device=returns.device, dtype=returns.dtype)
            if barrier_context is not None
            else None
        )
        rq = prediction["return_quantiles"]
        if not isinstance(rq, Tensor):
            raise ValueError("prediction['return_quantiles'] must be a tensor")
        vol_pred = prediction["volatility"]
        if not isinstance(vol_pred, Tensor):
            raise ValueError("prediction['volatility'] must be a tensor")

        q_loss = self._reduce(
            self._pinball_per_sample(rq, returns),
            weights,
        )
        # Loss computed in vol-RATIO space when a causal baseline (barrier_sigma) is available
        # -- see composite_loss.py's ForecasterLoss for the OOS-diagnostic rationale. vol_baseline
        # is [B] (one causal baseline per decision point); broadcast against the [B, H] per-horizon
        # vol prediction/target.
        vol_baseline = self._get(target, "vol_baseline")
        if vol_baseline is not None:
            vb = vol_baseline.reshape(-1, 1).clamp_min(1e-6)
            vol_pred_for_loss = vol_pred / vb
            vols_for_loss = vols / vb
        else:
            vol_pred_for_loss = vol_pred
            vols_for_loss = vols
        v_loss = self._reduce(
            F.smooth_l1_loss(vol_pred_for_loss, vols_for_loss, reduction="none").mean(dim=-1),
            weights,
        )
        c_loss = soft_coverage_loss(
            rq,
            returns,
            self._levels,
            sample_weight=weights,
            scale=vols,
        )

        total = (
            self._weights.return_ * q_loss
            + self._weights.volatility * v_loss
            + self._weights.calibration * c_loss
        )
        self.last_components = {
            "quantile": float(q_loss.detach()),
            "volatility": float(v_loss.detach()),
            "calibration": float(c_loss.detach()),
        }

        # Anti-collapse regularization on z -- see composite_loss.py::ForecasterLoss's identical
        # term for the full rationale (empirically confirmed z collapse to ~1 effective
        # dimension under supervised fine-tuning without this).
        z = prediction.get("z")
        if isinstance(z, Tensor) and z.shape[0] > 1 and (
            self._weights.repr_var or self._weights.repr_cov
        ):
            var_loss = _var_hinge(z)
            cov_loss = _offdiag_cov(z)
            total = total + self._weights.repr_var * var_loss + self._weights.repr_cov * cov_loss
            self.last_components["repr_var"] = float(var_loss.detach())
            self.last_components["repr_cov"] = float(cov_loss.detach())

        if self._weights.direction != 0.0:
            # World-model training is multi-horizon [B, H]. The legacy target.direction field is
            # single-step [B], so we derive horizon-wise direction labels directly from the
            # realized horizon returns to keep supervision causal and shape-consistent.
            median = rq[..., self._levels.numel() // 2]
            direction = _direction_labels_from_returns(returns)
            direction_weights = (
                weights.repeat_interleave(returns.shape[1]) if weights is not None else None
            )
            d_loss = self._reduce(
                _direction_surrogate_loss(
                    median.reshape(-1),
                    direction.reshape(-1),
                    scale=vol_pred.reshape(-1).detach(),
                ),
                direction_weights,
            )
            total = total + self._weights.direction * d_loss
            self.last_components["direction"] = float(d_loss.detach())

        mae_target = self._get(target, "horizon_mae")
        mae_pred = prediction.get("mae")
        if mae_target is not None and isinstance(mae_pred, Tensor):
            mae_loss = self._reduce(
                F.smooth_l1_loss(mae_pred, mae_target, reduction="none").mean(dim=-1),
                weights,
            )
            total = total + self._weights.mae * mae_loss
            self.last_components["mae"] = float(mae_loss.detach())

        mfe_target = self._get(target, "horizon_mfe")
        mfe_pred = prediction.get("mfe")
        if mfe_target is not None and isinstance(mfe_pred, Tensor):
            mfe_loss = self._reduce(
                F.smooth_l1_loss(mfe_pred, mfe_target, reduction="none").mean(dim=-1),
                weights,
            )
            total = total + self._weights.mfe * mfe_loss
            self.last_components["mfe"] = float(mfe_loss.detach())

        if (
            mae_target is not None
            and mfe_target is not None
            and isinstance(mae_pred, Tensor)
            and isinstance(mfe_pred, Tensor)
        ):
            coherence_weight = 0.25 * (self._weights.mae + self._weights.mfe)
            if coherence_weight != 0.0:
                coherence_loss = self._reduce(
                    _world_excursion_coherence_per_sample(rq, mae_pred, mfe_pred),
                    weights,
                )
                total = total + coherence_weight * coherence_loss
                self.last_components["excursion_coherence"] = float(coherence_loss.detach())

        if (
            geometry is not None
            and mae_target is not None
            and mfe_target is not None
            and isinstance(prediction.get("barrier_logits_intermediate"), Tensor)
            and len(pred_horizons) >= 2
        ):
            # Deep supervision at the non-management horizons (Phase 5b): the real
            # first-crossing label only exists at the management horizon (below), so this
            # excursion-ratio reconstruction is the only available barrier-shaped signal at
            # the shorter horizons — and reusing the same head weights there gives
            # barrier_head/excursion_barrier_head much shorter RSSM gradient paths than the
            # full management-horizon depth. Barrier width scales as sqrt(horizon_bars).
            _, stop_scale, target_scale = geometry
            intermediate_bars = torch.tensor(
                pred_horizons[:-1], device=stop_scale.device, dtype=stop_scale.dtype
            )
            horizon_scale = (intermediate_bars / float(pred_horizons[-1])).sqrt()  # [H']
            scaled_stop = stop_scale.unsqueeze(-1) * horizon_scale.unsqueeze(0)      # [N, H']
            scaled_target = target_scale.unsqueeze(-1) * horizon_scale.unsqueeze(0)  # [N, H']
            barrier_logits_intermediate = prediction.get("barrier_logits_intermediate")
            assert isinstance(barrier_logits_intermediate, Tensor)
            n, h_prime, n_classes = barrier_logits_intermediate.shape
            excursion_intermediate = _excursion_barrier_labels(
                mae_target[:, :-1],
                mfe_target[:, :-1],
                scaled_stop,
                scaled_target,
            )
            barrier_intermediate_loss = self._reduce(
                F.cross_entropy(
                    barrier_logits_intermediate.reshape(n * h_prime, n_classes),
                    excursion_intermediate,
                    weight=self._barrier_class_weight.to(
                        device=barrier_logits_intermediate.device,
                        dtype=barrier_logits_intermediate.dtype,
                    ),
                    reduction="none",
                ).reshape(n, h_prime).mean(dim=-1),
                weights,
            )
            if self._weights.barrier_intermediate != 0.0:
                total = total + self._weights.barrier_intermediate * barrier_intermediate_loss
            self.last_components["barrier_intermediate"] = float(
                barrier_intermediate_loss.detach()
            )

        barrier = self._get(target, "barrier")
        barrier_logits = prediction.get("barrier_logits")
        if barrier is not None and isinstance(barrier_logits, Tensor):
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
                    barrier_logits,
                    barrier.reshape(-1),
                    weight=self._barrier_class_weight.to(
                        device=barrier_logits.device, dtype=barrier_logits.dtype
                    ),
                    reduction="none",
                ),
                effective_weights,
            )
            total = total + self._weights.barrier * b_loss
            self.last_components["barrier"] = float(b_loss.detach())

        regime = self._get(target, "regime")
        regime_logits = prediction.get("regime_logits")
        if regime is not None and isinstance(regime_logits, Tensor):
            r_loss = self._reduce(
                F.cross_entropy(regime_logits, regime.reshape(-1), reduction="none"),
                weights,
            )
            total = total + self._weights.regime * r_loss
            self.last_components["regime"] = float(r_loss.detach())

        self.last_components["total"] = float(total.detach())
        return total

    def _pinball_per_sample(self, prediction: Tensor, target: Tensor) -> Tensor:
        levels = self._levels.to(device=prediction.device, dtype=prediction.dtype).reshape(1, 1, -1)
        error = target.unsqueeze(-1) - prediction
        loss = torch.maximum(levels * error, (levels - 1.0) * error)
        return loss.mean(dim=(1, 2))

    @staticmethod
    def _reduce(values: Tensor, weights: Tensor | None) -> Tensor:
        values = values.reshape(-1)
        if weights is None:
            return values.mean()
        weights = weights.to(device=values.device, dtype=values.dtype).reshape(-1).clamp_min(0.0)
        denom = weights.sum().clamp_min(1e-8)
        return torch.sum(values * weights) / denom


__all__ = ["WorldModelLoss"]


def _world_excursion_coherence_per_sample(
    return_quantiles: Tensor,
    mae: Tensor,
    mfe: Tensor,
) -> Tensor:
    if return_quantiles.ndim != 3:
        raise ValueError(
            "return_quantiles must be [B, H, Q] for world-model excursion coherence; "
            f"got {tuple(return_quantiles.shape)}"
        )
    per_horizon = _excursion_coherence_per_sample(
        return_quantiles.reshape(-1, return_quantiles.shape[-1]),
        mae.reshape(-1),
        mfe.reshape(-1),
    )
    return per_horizon.reshape(return_quantiles.shape[0], return_quantiles.shape[1]).mean(dim=-1)
