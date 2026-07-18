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

    def __init__(
        self, epistemic_scale: float = 1.0, ood_scale: float = 1.0, meta_label_scale: float = 1.0
    ) -> None:
        self._epistemic_scale = epistemic_scale
        self._ood_scale = ood_scale
        self._meta_label_scale = meta_label_scale

    def adjust(
        self, base_size: float, prediction: ModelPrediction, confidence_scale: float = 1.0
    ) -> float:
        """Return adjusted size in [0, base_size]; non-increasing in uncertainty.

        Meta-label gate (2026-07-18, see heads/meta_label_head.py): ``meta_label_prob``
        answers "is a trade worth it" ONLY for the momentum-based ``primary_side`` the
        head was conditioned on -- it says nothing about the opposite side. This method
        is called ONCE per decision (before the planner enumerates both long and short
        candidates), so it cannot know which side an individual candidate will end up
        being. As an MVP integration this applies the gate uniformly to the shared size
        CAP rather than per-candidate: when the model has no opinion (``primary_side ==
        0``) or no meta-label head output (older artifact, ``meta_label_prob is None``),
        the gate is a neutral 1.0 -- exactly the same "inert unless populated" contract
        the epistemic/OOD terms already follow, so this is backward compatible with
        every existing artifact.
        """
        epi = max(0.0, min(1.0, self._epistemic_scale * prediction.epistemic))
        ood = max(0.0, min(1.0, self._ood_scale * prediction.ood_score))
        meta_gate = 1.0
        if prediction.primary_side != 0 and prediction.meta_label_prob is not None:
            scaled = self._meta_label_scale * prediction.meta_label_prob
            meta_gate = max(0.0, min(1.0, scaled + (1.0 - self._meta_label_scale)))
        gate = max(0.0, min(1.0, confidence_scale)) * (1.0 - epi) * (1.0 - ood) * meta_gate
        return float(max(0.0, base_size * gate))
