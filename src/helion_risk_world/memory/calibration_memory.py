from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from helion_risk_world.evaluation.calibration_metrics import compute as calibration_compute
from helion_risk_world.schemas.prediction_schema import ModelPrediction


class CalibrationMonitor:
    """Tracks coverage/ECE; shrinks confidence (and thus size) when calibration degrades.

    SPEC.md §23 innovation 7 + Appendix A (CalibrationMonitor.check).
    """

    def __init__(self, ece_threshold: float = 0.1, coverage_threshold: float = 0.1,
                 decay: float = 0.9, min_scale: float = 0.25, min_samples: int = 16) -> None:
        self.confidence_scale = 1.0
        self._ece_thr = ece_threshold
        self._cov_thr = coverage_threshold
        self._decay = decay
        self._min_scale = min_scale
        self._min_samples = min_samples

    def check(
        self, predictions: Sequence[ModelPrediction], outcomes: Sequence[object]
    ) -> dict[str, float]:
        n = min(len(predictions), len(outcomes))
        if n == 0:
            return {"samples": 0.0, "confidence_scale": self.confidence_scale}

        pred_quantiles: list[list[float]] = []
        realized: list[float] = []
        barrier_probs: list[list[float]] = []
        barrier_labels: list[int] = []

        for prediction, outcome in zip(predictions[-n:], outcomes[-n:], strict=False):
            hp = prediction.longest_horizon
            pred_quantiles.append([hp.return_quantiles[q] for q in sorted(hp.return_quantiles)])
            realized.append(float(_read(outcome, "realized_return", "exit_return", default=0.0)))
            barrier = _read(outcome, "barrier", default=None)
            if barrier is not None:
                barrier_probs.append([
                    float(prediction.barrier.stop),
                    float(prediction.barrier.target),
                    float(prediction.barrier.timeout),
                ])
                barrier_labels.append(_barrier_idx(barrier))

        metrics = calibration_compute(
            pred_quantiles=np.asarray(pred_quantiles, dtype=float),
            realized=np.asarray(realized, dtype=float),
            barrier_probs=np.asarray(barrier_probs, dtype=float) if barrier_probs else None,
            barrier_labels=np.asarray(barrier_labels, dtype=int) if barrier_labels else None,
            quantile_levels=np.asarray(sorted(predictions[-1].longest_horizon.return_quantiles), dtype=float),
        )
        metrics["samples"] = float(n)

        if n < self._min_samples:
            metrics["confidence_scale"] = float(self.confidence_scale)
            return metrics

        degraded = (
            metrics.get("coverage_error", 0.0) > self._cov_thr
            or metrics.get("barrier_ece", 0.0) > self._ece_thr
        )
        if degraded:
            self.confidence_scale = max(self._min_scale, self.confidence_scale * self._decay)
        else:
            recovery = (1.0 - self._decay) * 0.5
            self.confidence_scale = min(1.0, self.confidence_scale + recovery)
        metrics["confidence_scale"] = float(self.confidence_scale)
        return metrics


def _read(outcome: object, *names: str, default: object = None) -> object:
    if isinstance(outcome, dict):
        for name in names:
            if name in outcome:
                return outcome[name]
        return default
    for name in names:
        if hasattr(outcome, name):
            return getattr(outcome, name)
    return default


def _barrier_idx(value: object) -> int:
    raw = str(value).lower()
    if raw.endswith("stop"):
        return 0
    if raw.endswith("target"):
        return 1
    return 2
