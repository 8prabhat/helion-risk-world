"""Uncertainty-gated position sizing (SPEC.md §18, §23 ablation 4, Day 6).

Size shrinks as epistemic uncertainty and OOD score rise, and as calibration degrades
(``confidence_scale`` from the CalibrationMonitor). SRP: maps a raw size + uncertainty signals to an
adjusted size in [0, base_size].

NOTE (review finding H9): the epistemic term is inert whenever
``prediction.epistemic_calibrated`` is False — ``ForecasterPredictor`` (the default,
non-world-model predictor) always emits ``epistemic=0.0`` with no real ensemble
behind it, so ``epi`` below is always 0 (no size reduction) on that path. Use
``model_kind='world_model'`` for this sizing term to be meaningful.
"""

from __future__ import annotations

from helion_risk_world.schemas.prediction_schema import ModelPrediction


class PositionSizer:
    """Shrink position size under uncertainty / OOD / calibration drift."""

    def __init__(self, epistemic_scale: float = 1.0, ood_scale: float = 1.0) -> None:
        self._epistemic_scale = epistemic_scale
        self._ood_scale = ood_scale

    def adjust(
        self, base_size: float, prediction: ModelPrediction, confidence_scale: float = 1.0
    ) -> float:
        """Return adjusted size in [0, base_size]; non-increasing in uncertainty."""
        epi = max(0.0, min(1.0, self._epistemic_scale * prediction.epistemic))
        ood = max(0.0, min(1.0, self._ood_scale * prediction.ood_score))
        gate = max(0.0, min(1.0, confidence_scale)) * (1.0 - epi) * (1.0 - ood)
        return float(max(0.0, base_size * gate))
