"""Artifact-backed post-hoc calibration for runtime predictions.

The model heads learn a first-pass predictive distribution. This module fits small,
chronology-safe post-hoc corrections that can be serialized into the promoted artifact
and applied consistently in predict/backtest/paper-trading flows.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from helion_risk_world.schemas.prediction_schema import (
    BarrierProbabilities,
    HorizonPrediction,
    ModelPrediction,
)

_EPS = 1e-9

# Safeguards for temperature fitting (2026-07-13): the original grid search picked whatever
# temperature minimized loss on the SAME sample it was evaluated against -- pure in-sample
# optimization. Confirmed empirically to overfit: a fitted barrier_temperature=3.76 from ~2588
# validation samples made held-out test barrier_ece WORSE (0.040 -> 0.114) than no correction
# at all (T=1.0), on the identical underlying model. These constants gate a chronological
# fit/check split (see `_fit_temperature_with_holdout`) so a temperature is only ever applied
# if it demonstrably generalizes to samples it wasn't fit on.
_MIN_TEMPERATURE_FIT_SAMPLES = 200  # below this, don't trust any split-based fit at all
_TEMPERATURE_HOLDOUT_FRACTION = 0.4
_TEMPERATURE_SHRINKAGE = 0.5  # even a holdout-validated temperature is a noisy point estimate
# from a few-hundred/thousand-sample split; partially shrinking back to 1.0 is a cheap extra
# safeguard against a holdout split that happened to validate a fluke.


@dataclass(frozen=True)
class HorizonPredictionCalibration:
    """Per-horizon quantile and volatility corrections."""

    horizon_bars: int
    quantile_offsets: dict[float, float]
    volatility_scale: float = 1.0
    volatility_bias: float = 0.0
    sample_count: int = 0

    def apply(self, prediction: HorizonPrediction) -> HorizonPrediction:
        levels = sorted(prediction.return_quantiles)
        adjusted = [
            float(prediction.return_quantiles[level] + self.quantile_offsets.get(level, 0.0))
            for level in levels
        ]
        monotone = np.maximum.accumulate(np.asarray(adjusted, dtype=float))
        calibrated_quantiles = {
            level: float(monotone[idx])
            for idx, level in enumerate(levels)
        }
        calibrated_vol = max(
            float(self.volatility_scale * prediction.volatility + self.volatility_bias),
            1e-6,
        )
        return HorizonPrediction(
            horizon_bars=prediction.horizon_bars,
            return_quantiles=calibrated_quantiles,
            volatility=calibrated_vol,
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "horizon_bars": int(self.horizon_bars),
            "quantile_offsets": {
                str(level): float(offset)
                for level, offset in sorted(self.quantile_offsets.items())
            },
            "volatility_scale": float(self.volatility_scale),
            "volatility_bias": float(self.volatility_bias),
            "sample_count": int(self.sample_count),
        }

    @classmethod
    def from_metadata(cls, payload: Mapping[str, object]) -> "HorizonPredictionCalibration":
        return cls(
            horizon_bars=int(payload["horizon_bars"]),
            quantile_offsets={
                float(level): float(offset)
                for level, offset in dict(payload.get("quantile_offsets", {})).items()
            },
            volatility_scale=float(payload.get("volatility_scale", 1.0)),
            volatility_bias=float(payload.get("volatility_bias", 0.0)),
            sample_count=int(payload.get("sample_count", 0)),
        )


@dataclass(frozen=True)
class PredictionCalibration:
    """Serialized runtime calibration bundle."""

    quantile_levels: tuple[float, ...]
    horizons: dict[int, HorizonPredictionCalibration]
    barrier_temperature: float = 1.0
    regime_temperature: float = 1.0
    barrier_prior_offsets: tuple[float, float, float] | None = None
    """Per-class log-prior correction for barrier probabilities (2026-07-18).

    Training uses class-weighted cross-entropy (``LossWeights.barrier_class_weights``,
    auto-computed from the label distribution) to prevent majority-class collapse. That
    upweighting systematically inflates minority-class predicted probabilities relative
    to their true base rates — a PRIOR SHIFT, which temperature scaling structurally
    cannot fix (temperature only sharpens/flattens toward uniform; it has no per-class
    direction). Diagnosed empirically 2026-07-11: mean predicted P(target)=27% against a
    0.9% true base rate, and the crude divide-by-class-weight correction cut Brier 37%.
    This field is the properly-fit version of that correction: additive per-class
    offsets in logit space, fit on a chronological split with the same
    holdout-validation + shrinkage safeguards as the temperature (see
    ``_fit_barrier_prior_offsets``). Applied BEFORE temperature. ``None`` = no
    correction (not validated to generalize, or never fit)."""
    source: str = "unknown"
    sample_count: int = 0

    def apply(self, prediction: ModelPrediction) -> ModelPrediction:
        calibrated_horizons: list[HorizonPrediction] = []
        original_management = prediction.longest_horizon
        management = original_management
        for horizon_pred in prediction.horizon_preds:
            calibrated = self.horizons.get(horizon_pred.horizon_bars, None)
            updated = calibrated.apply(horizon_pred) if calibrated is not None else horizon_pred
            calibrated_horizons.append(updated)
            if horizon_pred.horizon_bars == original_management.horizon_bars:
                management = updated

        barrier = _apply_barrier_temperature(
            _apply_barrier_prior_offsets(prediction.barrier, self.barrier_prior_offsets),
            self.barrier_temperature,
        )
        regime_probs = _apply_probability_temperature(prediction.regime_probs, self.regime_temperature)
        aleatoric_scale = _interval_width(management) / max(_interval_width(original_management), _EPS)
        mae = max(float(prediction.mae) * aleatoric_scale, 0.0)
        mfe = max(float(prediction.mfe) * aleatoric_scale, 0.0)
        sigma_h = float(management.volatility)
        aleatoric = max(float(prediction.aleatoric) * aleatoric_scale, 0.0)
        return ModelPrediction(
            symbol=prediction.symbol,
            ts=prediction.ts,
            horizon_preds=calibrated_horizons,
            barrier=barrier,
            mae=mae,
            mfe=mfe,
            sigma_H=sigma_h,
            stop_return=prediction.stop_return,
            target_return=prediction.target_return,
            regime_probs=regime_probs,
            # Meta-labeling fields (2026-07-18): not recalibrated (no fitted correction for
            # them yet), but must be carried through unchanged -- same "don't silently drop
            # a field on rebuild" bug class the epistemic_calibrated fix below addresses.
            primary_side=int(prediction.primary_side),
            meta_label_prob=(
                float(prediction.meta_label_prob)
                if prediction.meta_label_prob is not None
                else None
            ),
            epistemic=float(prediction.epistemic),
            aleatoric=aleatoric,
            ood_score=float(prediction.ood_score),
            # Preserve the placeholder flag (2026-07-18 fix): omitting this field
            # defaulted it back to True, silently re-marking ForecasterPredictor's
            # hardcoded epistemic=0.0 as "calibrated" after calibration was applied.
            epistemic_calibrated=bool(prediction.epistemic_calibrated),
        )

    def to_metadata(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "version": 1,
            "source": self.source,
            "sample_count": int(self.sample_count),
            "quantile_levels": [float(level) for level in self.quantile_levels],
            "barrier_temperature": float(self.barrier_temperature),
            "regime_temperature": float(self.regime_temperature),
            "horizons": {
                str(horizon): calibration.to_metadata()
                for horizon, calibration in sorted(self.horizons.items())
            },
        }
        if self.barrier_prior_offsets is not None:
            payload["barrier_prior_offsets"] = [float(v) for v in self.barrier_prior_offsets]
        return payload

    @classmethod
    def from_metadata(cls, payload: Mapping[str, object] | None) -> "PredictionCalibration | None":
        if payload is None:
            return None
        horizon_payloads = dict(payload.get("horizons", {}))
        raw_offsets = payload.get("barrier_prior_offsets")
        prior_offsets: tuple[float, float, float] | None = None
        if raw_offsets is not None:
            values = [float(v) for v in raw_offsets]  # type: ignore[union-attr]
            if len(values) == 3:
                prior_offsets = (values[0], values[1], values[2])
        return cls(
            quantile_levels=tuple(float(level) for level in payload.get("quantile_levels", ())),
            horizons={
                int(horizon): HorizonPredictionCalibration.from_metadata(
                    dict(calibration_payload)
                )
                for horizon, calibration_payload in horizon_payloads.items()
            },
            barrier_temperature=float(payload.get("barrier_temperature", 1.0)),
            regime_temperature=float(payload.get("regime_temperature", 1.0)),
            barrier_prior_offsets=prior_offsets,
            source=str(payload.get("source", "unknown")),
            sample_count=int(payload.get("sample_count", 0)),
        )


def fit_prediction_calibration(
    *,
    quantile_levels: Sequence[float],
    horizon_payloads: Mapping[int, Mapping[str, Sequence[Sequence[float]] | Sequence[float]]],
    barrier_probs: Sequence[Sequence[float]] | None = None,
    barrier_labels: Sequence[int] | None = None,
    regime_probs: Sequence[Sequence[float]] | None = None,
    regime_labels: Sequence[int] | None = None,
    source: str,
) -> PredictionCalibration | None:
    levels = tuple(float(level) for level in quantile_levels)
    if not levels:
        return None

    calibrated_horizons: dict[int, HorizonPredictionCalibration] = {}
    sample_count = 0
    for horizon, payload in horizon_payloads.items():
        pred_quantiles = np.asarray(payload.get("pred_quantiles", []), dtype=float)
        realized = np.asarray(payload.get("realized", []), dtype=float).reshape(-1)
        if pred_quantiles.ndim != 2 or pred_quantiles.shape[0] == 0 or pred_quantiles.shape[1] != len(levels):
            continue
        if realized.shape[0] != pred_quantiles.shape[0]:
            continue
        offsets = _fit_conformal_offsets(pred_quantiles, realized, levels)
        pred_vol = np.asarray(payload.get("predicted_volatility", []), dtype=float).reshape(-1)
        real_vol = np.asarray(payload.get("realized_volatility", []), dtype=float).reshape(-1)
        vol_scale, vol_bias = _fit_volatility_affine(pred_vol, real_vol)
        sample_count = max(sample_count, int(realized.shape[0]))
        calibrated_horizons[int(horizon)] = HorizonPredictionCalibration(
            horizon_bars=int(horizon),
            quantile_offsets=offsets,
            volatility_scale=vol_scale,
            volatility_bias=vol_bias,
            sample_count=int(realized.shape[0]),
        )

    if not calibrated_horizons:
        return None

    # Fit the per-class prior correction FIRST (it targets the class-weight-induced
    # prior shift temperature cannot express), then fit the temperature on the
    # prior-corrected probabilities so the two compose the same way apply() applies them.
    barrier_prior_offsets = _fit_barrier_prior_offsets(barrier_probs, barrier_labels)
    temperature_input_probs = barrier_probs
    if barrier_prior_offsets is not None and barrier_probs is not None:
        temperature_input_probs = _apply_prior_offsets_array(
            np.asarray(barrier_probs, dtype=float), np.asarray(barrier_prior_offsets)
        )
    barrier_temperature = _fit_barrier_temperature(temperature_input_probs, barrier_labels)
    regime_temperature = _fit_temperature(regime_probs, regime_labels)
    return PredictionCalibration(
        quantile_levels=levels,
        horizons=calibrated_horizons,
        barrier_temperature=barrier_temperature,
        regime_temperature=regime_temperature,
        barrier_prior_offsets=barrier_prior_offsets,
        source=source,
        sample_count=sample_count,
    )


def _fit_conformal_offsets(
    pred_quantiles: np.ndarray,
    realized: np.ndarray,
    levels: Sequence[float],
) -> dict[float, float]:
    """Conformalized quantile regression (CQR; Romano, Patterson, Candes 2019), pairing
    symmetric levels around the median into intervals and computing a single conformal
    margin per pair that WIDENS (or narrows) the interval, rather than just shifting its
    location.

    Diagnostic finding (2026-07-05): the previous approach here -- an independent additive
    shift per level, `offset = quantile_tau(realized - pred_tau)` -- cannot fix a
    systematically too-narrow interval, only its centering. Empirically, every quantile
    level (0.1 through 0.9) had 40-58% of realized outcomes falling below it regardless of
    the nominal target, the signature of predicted quantiles clustered too close together,
    not miscentered. CQR's nonconformity score `max(pred_lo - y, y - pred_hi)` targets
    interval width directly and gives a finite-sample marginal coverage guarantee (under
    exchangeability) for each paired interval. Two location-shift-only alternatives (fit on
    a more recent window, fit per-regime) were tried first and neither beat the original
    per-level shift; CQR was the one approach that visibly corrected the outer/inner
    interval coverage in isolation, even though the aggregate coverage_error landed in the
    same range as before -- adopted here as the more principled mechanism going forward.

    Any level without a symmetric partner (e.g. the median in an odd-length level set) falls
    back to the original single-quantile conformal correction (`quantile_tau(residual)`),
    itself a valid, simpler conformal method for an unpaired point.
    """
    sorted_levels = sorted(levels)
    n = pred_quantiles.shape[0]
    index = {level: i for i, level in enumerate(sorted_levels)}
    offsets: dict[float, float] = {}
    paired: set[float] = set()
    lo_ptr, hi_ptr = 0, len(sorted_levels) - 1
    while lo_ptr < hi_ptr:
        lo, hi = sorted_levels[lo_ptr], sorted_levels[hi_ptr]
        alpha = max(1e-6, min(1.0, 1.0 - (hi - lo)))
        nonconformity = np.maximum(
            pred_quantiles[:, index[lo]] - realized,
            realized - pred_quantiles[:, index[hi]],
        )
        q_level = min(np.ceil((n + 1) * (1.0 - alpha)) / max(n, 1), 1.0) if n > 0 else 1.0
        margin = float(np.quantile(nonconformity, q_level)) if n > 0 else 0.0
        offsets[lo] = -margin
        offsets[hi] = margin
        paired.add(lo)
        paired.add(hi)
        lo_ptr += 1
        hi_ptr -= 1
    for level in sorted_levels:
        if level not in paired:
            offsets[level] = float(np.quantile(realized - pred_quantiles[:, index[level]], level))
    return offsets


def _fit_volatility_affine(
    predicted: np.ndarray,
    realized: np.ndarray,
) -> tuple[float, float]:
    if predicted.size == 0 or realized.size == 0 or predicted.shape != realized.shape:
        return 1.0, 0.0
    if float(np.nanstd(predicted)) < 1e-8:
        baseline = float(np.nanmedian(predicted))
        if abs(baseline) >= 1e-8:
            return max(float(np.nanmedian(realized / baseline)), 0.0), 0.0
        return 0.0, float(np.nanmedian(realized))
    design = np.column_stack([predicted, np.ones_like(predicted)])
    coef, *_ = np.linalg.lstsq(design, realized, rcond=None)
    scale = float(max(coef[0], 0.0))
    bias = float(coef[1])
    return scale, bias


def _apply_prior_offsets_array(probs: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Apply additive log-space per-class offsets to a [N, C] probability array."""
    logits = np.log(np.clip(probs, _EPS, 1.0)) + offsets.reshape(1, -1)
    logits = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True).clip(min=_EPS)


_PRIOR_OFFSET_CLIP = 3.0  # |log prior ratio| cap: a >20x prior correction is more likely
# a degenerate split (a class absent from the fit window) than a real, stable shift.


def _fit_barrier_prior_offsets(
    probs: Sequence[Sequence[float]] | None,
    labels: Sequence[int] | None,
) -> tuple[float, float, float] | None:
    """Fit per-class log-prior offsets correcting the class-weighted-CE prior shift.

    Analytic fit on the chronological FIT split: ``offset_c = log(freq_c / mean_pred_c)``
    (the standard prior-correction for a model trained under class reweighting), clipped
    and mean-centered (softmax is shift-invariant, centering keeps magnitudes honest).
    Accepted only if it improves Brier+ECE on the held-out CHECK split it wasn't fit on,
    then shrunk halfway toward zero — the exact same safeguard pattern
    ``_fit_temperature_with_holdout`` uses, and for the same empirically-confirmed
    reason (in-sample calibration fits on this data overfit).
    """
    if probs is None or labels is None:
        return None
    prob_array = np.asarray(probs, dtype=float)
    label_array = np.asarray(labels, dtype=int).reshape(-1)
    n = prob_array.shape[0]
    if prob_array.ndim != 2 or prob_array.shape[1] != 3 or n == 0 or n != label_array.shape[0]:
        return None
    if n < _MIN_TEMPERATURE_FIT_SAMPLES:
        return None

    split = int(round(n * (1.0 - _TEMPERATURE_HOLDOUT_FRACTION)))
    split = max(1, min(n - 1, split))
    fit_probs, fit_labels = prob_array[:split], label_array[:split]
    check_probs, check_labels = prob_array[split:], label_array[split:]

    n_classes = prob_array.shape[1]
    freq = np.bincount(fit_labels, minlength=n_classes).astype(float)
    if np.any(freq == 0):
        return None  # a class absent from the fit window: any offset for it is a guess
    freq = freq / freq.sum()
    mean_pred = fit_probs.mean(axis=0).clip(min=_EPS)
    offsets = np.clip(np.log(freq / mean_pred), -_PRIOR_OFFSET_CLIP, _PRIOR_OFFSET_CLIP)
    offsets = offsets - offsets.mean()

    def _score(p: np.ndarray, y: np.ndarray) -> float:
        metrics = _classification_calibration_metrics(p, y)
        return metrics["brier"] + metrics["ece"]

    baseline_check = _score(check_probs, check_labels)
    candidate_check = _score(
        _apply_prior_offsets_array(check_probs, offsets), check_labels
    )
    if candidate_check >= baseline_check - 1e-12:
        return None  # didn't generalize to held-out data — reject, same as temperature

    shrunk = _TEMPERATURE_SHRINKAGE * offsets
    return (float(shrunk[0]), float(shrunk[1]), float(shrunk[2]))


def _fit_temperature(
    probs: Sequence[Sequence[float]] | None,
    labels: Sequence[int] | None,
) -> float:
    return _fit_temperature_with_holdout(probs, labels, loss_fn=_nll_loss)


def _fit_barrier_temperature(
    probs: Sequence[Sequence[float]] | None,
    labels: Sequence[int] | None,
) -> float:
    def _brier_plus_ece(p: np.ndarray, y: np.ndarray) -> float:
        metrics = _classification_calibration_metrics(p, y)
        return metrics["brier"] + metrics["ece"]

    return _fit_temperature_with_holdout(probs, labels, loss_fn=_brier_plus_ece)


def _fit_temperature_with_holdout(
    probs: Sequence[Sequence[float]] | None,
    labels: Sequence[int] | None,
    *,
    loss_fn: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    """Grid-search a temperature on a chronological FIT split, only accept it if it also
    improves (not just doesn't-get-worse) on a held-out CHECK split it wasn't fit on, then
    shrink halfway back to 1.0 as a further safeguard -- see the module-level constants'
    docstring for why (a pure in-sample fit was confirmed to overfit on real data).

    ``probs``/``labels`` are assumed already in their natural (typically chronological)
    collection order, e.g. from a per-timestamp validation loop -- the fit/check split is a
    plain prefix/suffix split, not a shuffle, so the check split is never "from the past"
    relative to what the temperature was fit on.
    """
    if probs is None or labels is None:
        return 1.0
    prob_array = np.asarray(probs, dtype=float)
    label_array = np.asarray(labels, dtype=int).reshape(-1)
    n = prob_array.shape[0]
    if prob_array.ndim != 2 or n == 0 or n != label_array.shape[0]:
        return 1.0
    if n < _MIN_TEMPERATURE_FIT_SAMPLES:
        return 1.0

    split = int(round(n * (1.0 - _TEMPERATURE_HOLDOUT_FRACTION)))
    split = max(1, min(n - 1, split))
    fit_probs, fit_labels = prob_array[:split], label_array[:split]
    check_probs, check_labels = prob_array[split:], label_array[split:]

    baseline_fit = loss_fn(fit_probs, fit_labels)
    best_temp, best_score = 1.0, baseline_fit
    for temp in np.geomspace(0.35, 4.0, 41):
        score = loss_fn(_apply_temperature(fit_probs, float(temp)), fit_labels)
        if score < best_score - 1e-12:
            best_temp, best_score = float(temp), score
    if best_temp == 1.0:
        return 1.0

    baseline_check = loss_fn(check_probs, check_labels)
    candidate_check = loss_fn(_apply_temperature(check_probs, best_temp), check_labels)
    if candidate_check >= baseline_check - 1e-12:
        return 1.0  # didn't generalize to held-out data it wasn't fit on -- reject

    return float(1.0 + _TEMPERATURE_SHRINKAGE * (best_temp - 1.0))


def _nll_loss(probs: np.ndarray, labels: np.ndarray) -> float:
    rows = np.arange(labels.shape[0])
    chosen = probs[rows, labels].clip(min=_EPS)
    return float(-np.log(chosen).mean())


def _classification_calibration_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    one_hot = np.eye(probs.shape[1], dtype=float)[labels]
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:], strict=False):
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if bool(mask.any()):
            ece += abs(correct[mask].mean() - conf[mask].mean()) * mask.mean()
    return {"brier": brier, "ece": float(ece)}


def _apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    temp = max(float(temperature), 1e-3)
    logits = np.log(np.clip(probs, _EPS, 1.0)) / temp
    logits = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True).clip(min=_EPS)


def _apply_barrier_prior_offsets(
    barrier: BarrierProbabilities,
    offsets: tuple[float, float, float] | None,
) -> BarrierProbabilities:
    if offsets is None:
        return barrier
    corrected = _apply_prior_offsets_array(
        np.asarray([[barrier.stop, barrier.target, barrier.timeout]], dtype=float),
        np.asarray(offsets, dtype=float),
    )[0]
    return BarrierProbabilities(
        stop=float(corrected[0]),
        target=float(corrected[1]),
        timeout=float(corrected[2]),
    )


def _apply_barrier_temperature(
    barrier: BarrierProbabilities,
    temperature: float,
) -> BarrierProbabilities:
    scaled = _apply_temperature(
        np.asarray([[barrier.stop, barrier.target, barrier.timeout]], dtype=float),
        temperature,
    )[0]
    return BarrierProbabilities(
        stop=float(scaled[0]),
        target=float(scaled[1]),
        timeout=float(scaled[2]),
    )


def _apply_probability_temperature(
    probs: Mapping[Any, float] | None,
    temperature: float,
) -> dict[Any, float] | None:
    if probs is None:
        return None
    if not probs:
        return {}
    items = list(probs.items())
    scaled = _apply_temperature(
        np.asarray([[float(value) for _, value in items]], dtype=float),
        temperature,
    )[0]
    return {
        key: float(scaled[idx])
        for idx, (key, _) in enumerate(items)
    }


def _interval_width(prediction: HorizonPrediction) -> float:
    levels = sorted(prediction.return_quantiles)
    return float(prediction.return_quantiles[levels[-1]] - prediction.return_quantiles[levels[0]])


__all__ = [
    "HorizonPredictionCalibration",
    "PredictionCalibration",
    "fit_prediction_calibration",
]
