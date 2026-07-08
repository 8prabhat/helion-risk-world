"""Predictive diagnostics for forecaster/world-model evaluation.

This extends the raw calibration gate with:
  * point-forecast accuracy from the median quantile,
  * causal expanding-window baselines,
  * regime parity summaries,
  * uncertainty robustness slices.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from helion_risk_world.evaluation.calibration_metrics import compute as calibration_compute
from helion_risk_world.evaluation.world_model_metrics import compute as world_model_compute
from helion_risk_world.schemas.prediction_schema import QUANTILE_LEVELS

_EPS = 1e-12


def evaluate_predictive_outputs(
    *,
    pred_quantiles: Any,
    realized: Any,
    barrier_probs: Any = None,
    barrier_labels: Any = None,
    quantile_levels: Any = None,
    predicted_volatility: Any = None,
    realized_volatility: Any = None,
    regime_labels: Any = None,
    epistemic: Any = None,
    ood_scores: Any = None,
    baseline_min_history: int = 32,
) -> dict[str, Any]:
    """Build a richer predictive-evaluation report from saved model outputs."""
    q = np.asarray(pred_quantiles, dtype=float)
    y = np.asarray(realized, dtype=float).reshape(-1)
    if q.ndim != 2:
        raise ValueError(f"pred_quantiles must be [N, Q]; got shape {q.shape}")
    if q.shape[0] != y.shape[0]:
        raise ValueError("pred_quantiles and realized must share the same first dimension")

    levels = _resolve_quantile_levels(q, quantile_levels)
    probs = _maybe_2d(barrier_probs, y.size, "barrier_probs")
    labels = _maybe_1d(barrier_labels, y.size, "barrier_labels", dtype=int)
    pred_vol = _maybe_1d(predicted_volatility, y.size, "predicted_volatility")
    real_vol = _maybe_1d(realized_volatility, y.size, "realized_volatility")
    regimes = _maybe_1d(regime_labels, y.size, "regime_labels", dtype=object)
    epistemic_arr = _maybe_1d(epistemic, y.size, "epistemic")
    ood_arr = _maybe_1d(ood_scores, y.size, "ood_scores")

    metrics = _slice_metrics(
        pred_quantiles=q,
        realized=y,
        quantile_levels=levels,
        barrier_probs=probs,
        barrier_labels=labels,
        predicted_volatility=pred_vol,
        realized_volatility=real_vol,
        epistemic=epistemic_arr,
    )
    report: dict[str, Any] = {
        "samples": int(y.size),
        "metrics": metrics,
        "baseline_comparison": _baseline_comparison(
            pred_quantiles=q,
            realized=y,
            quantile_levels=levels,
            barrier_probs=probs,
            barrier_labels=labels,
            predicted_volatility=pred_vol,
            realized_volatility=real_vol,
            baseline_min_history=baseline_min_history,
        ),
    }

    if regimes is not None:
        regimes = np.asarray([_label_key(label) for label in regimes], dtype=object)
        breakdown = _group_breakdown(
            pred_quantiles=q,
            realized=y,
            quantile_levels=levels,
            group_labels=regimes,
            barrier_probs=probs,
            barrier_labels=labels,
            predicted_volatility=pred_vol,
            realized_volatility=real_vol,
        )
        if breakdown:
            report["regime_breakdown"] = breakdown
            report["regime_parity"] = _parity_summary(breakdown)

    uncertainty_breakdown: dict[str, Any] = {}
    if epistemic_arr is not None:
        uncertainty_breakdown["epistemic"] = _uncertainty_breakdown(
            name="epistemic",
            scores=epistemic_arr,
            pred_quantiles=q,
            realized=y,
            quantile_levels=levels,
            barrier_probs=probs,
            barrier_labels=labels,
        )
    if ood_arr is not None:
        uncertainty_breakdown["ood_score"] = _uncertainty_breakdown(
            name="ood_score",
            scores=ood_arr,
            pred_quantiles=q,
            realized=y,
            quantile_levels=levels,
            barrier_probs=probs,
            barrier_labels=labels,
        )
    if uncertainty_breakdown:
        report["uncertainty_breakdown"] = uncertainty_breakdown

    return report


def _baseline_comparison(
    *,
    pred_quantiles: np.ndarray,
    realized: np.ndarray,
    quantile_levels: np.ndarray,
    barrier_probs: np.ndarray | None,
    barrier_labels: np.ndarray | None,
    predicted_volatility: np.ndarray | None,
    realized_volatility: np.ndarray | None,
    baseline_min_history: int,
) -> dict[str, Any]:
    baseline_quantiles = _expanding_quantile_baseline(
        realized,
        quantile_levels,
        min_history=baseline_min_history,
    )
    model_point = _median_prediction(pred_quantiles, quantile_levels)
    baseline_point = _median_prediction(baseline_quantiles, quantile_levels)

    model_metrics = _slice_metrics(
        pred_quantiles=pred_quantiles,
        realized=realized,
        quantile_levels=quantile_levels,
        barrier_probs=barrier_probs,
        barrier_labels=barrier_labels,
        predicted_volatility=predicted_volatility,
        realized_volatility=realized_volatility,
    )
    baseline_metrics = _slice_metrics(
        pred_quantiles=baseline_quantiles,
        realized=realized,
        quantile_levels=quantile_levels,
        barrier_probs=(
            _expanding_barrier_prior_baseline(
                barrier_labels,
                barrier_probs.shape[1],
                baseline_min_history,
            )
            if barrier_probs is not None and barrier_labels is not None
            else None
        ),
        barrier_labels=barrier_labels,
        predicted_volatility=(
            _expanding_mean_baseline(realized_volatility, baseline_min_history)
            if realized_volatility is not None
            else None
        ),
        realized_volatility=realized_volatility,
    )

    out: dict[str, Any] = {
        "point": {
            "model_rollout_mae": float(model_metrics["rollout_mae"]),
            "baseline_rollout_mae": float(baseline_metrics["rollout_mae"]),
            "mae_improvement": float(baseline_metrics["rollout_mae"] - model_metrics["rollout_mae"]),
            "mae_skill": _lower_is_better_skill(
                float(model_metrics["rollout_mae"]),
                float(baseline_metrics["rollout_mae"]),
            ),
            "model_rollout_rmse": float(model_metrics["rollout_rmse"]),
            "baseline_rollout_rmse": float(baseline_metrics["rollout_rmse"]),
            "rmse_improvement": float(
                baseline_metrics["rollout_rmse"] - model_metrics["rollout_rmse"]
            ),
            "rmse_skill": _lower_is_better_skill(
                float(model_metrics["rollout_rmse"]),
                float(baseline_metrics["rollout_rmse"]),
            ),
            "model_point_bias": float(np.mean(model_point - realized)),
            "baseline_point_bias": float(np.mean(baseline_point - realized)),
        },
        "quantiles": {
            "model_coverage_error": float(model_metrics["coverage_error"]),
            "baseline_coverage_error": float(baseline_metrics["coverage_error"]),
            "coverage_improvement": float(
                baseline_metrics["coverage_error"] - model_metrics["coverage_error"]
            ),
            "model_interval_width": float(model_metrics["interval_width"]),
            "baseline_interval_width": float(baseline_metrics["interval_width"]),
        },
    }

    if "barrier_brier" in model_metrics and "barrier_brier" in baseline_metrics:
        out["barrier"] = {
            "model_barrier_brier": float(model_metrics["barrier_brier"]),
            "baseline_barrier_brier": float(baseline_metrics["barrier_brier"]),
            "brier_improvement": float(
                baseline_metrics["barrier_brier"] - model_metrics["barrier_brier"]
            ),
            "brier_skill": _lower_is_better_skill(
                float(model_metrics["barrier_brier"]),
                float(baseline_metrics["barrier_brier"]),
            ),
            "model_barrier_ece": float(model_metrics["barrier_ece"]),
            "baseline_barrier_ece": float(baseline_metrics["barrier_ece"]),
        }

    if "volatility_mae" in model_metrics and "volatility_mae" in baseline_metrics:
        out["volatility"] = {
            "model_volatility_mae": float(model_metrics["volatility_mae"]),
            "baseline_volatility_mae": float(baseline_metrics["volatility_mae"]),
            "volatility_mae_improvement": float(
                baseline_metrics["volatility_mae"] - model_metrics["volatility_mae"]
            ),
            "volatility_mae_skill": _lower_is_better_skill(
                float(model_metrics["volatility_mae"]),
                float(baseline_metrics["volatility_mae"]),
            ),
            "model_volatility_rmse": float(model_metrics["volatility_rmse"]),
            "baseline_volatility_rmse": float(baseline_metrics["volatility_rmse"]),
        }

    return out


def _slice_metrics(
    *,
    pred_quantiles: np.ndarray,
    realized: np.ndarray,
    quantile_levels: np.ndarray,
    barrier_probs: np.ndarray | None = None,
    barrier_labels: np.ndarray | None = None,
    predicted_volatility: np.ndarray | None = None,
    realized_volatility: np.ndarray | None = None,
    epistemic: np.ndarray | None = None,
) -> dict[str, float]:
    point = _median_prediction(pred_quantiles, quantile_levels)
    out = world_model_compute(predicted=point, target=realized, epistemic=epistemic)
    out.update(
        calibration_compute(
            pred_quantiles=pred_quantiles,
            realized=realized,
            barrier_probs=barrier_probs,
            barrier_labels=barrier_labels,
            quantile_levels=quantile_levels,
        )
    )
    if predicted_volatility is not None and realized_volatility is not None:
        diff = np.asarray(predicted_volatility - realized_volatility, dtype=float)
        out["volatility_mae"] = float(np.abs(diff).mean())
        out["volatility_rmse"] = float(np.sqrt(np.square(diff).mean()))
        out["volatility_bias"] = float(diff.mean())
    return out


def _group_breakdown(
    *,
    pred_quantiles: np.ndarray,
    realized: np.ndarray,
    quantile_levels: np.ndarray,
    group_labels: np.ndarray,
    barrier_probs: np.ndarray | None = None,
    barrier_labels: np.ndarray | None = None,
    predicted_volatility: np.ndarray | None = None,
    realized_volatility: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for label in np.unique(group_labels):
        mask = group_labels == label
        key = _label_key(label)
        metrics = _slice_metrics(
            pred_quantiles=pred_quantiles[mask],
            realized=realized[mask],
            quantile_levels=quantile_levels,
            barrier_probs=barrier_probs[mask] if barrier_probs is not None else None,
            barrier_labels=barrier_labels[mask] if barrier_labels is not None else None,
            predicted_volatility=(
                predicted_volatility[mask] if predicted_volatility is not None else None
            ),
            realized_volatility=(
                realized_volatility[mask] if realized_volatility is not None else None
            ),
        )
        metrics["count"] = float(mask.sum())
        out[key] = metrics
    return out


def _parity_summary(groups: dict[str, dict[str, float]]) -> dict[str, float | str]:
    out: dict[str, float | str] = {}
    for key in ("rollout_mae", "coverage_error", "barrier_brier", "volatility_mae"):
        values = {name: metrics[key] for name, metrics in groups.items() if key in metrics}
        if len(values) < 2:
            continue
        worst_name = max(values, key=values.get)
        best_name = min(values, key=values.get)
        out[f"{key}_gap"] = float(values[worst_name] - values[best_name])
        out[f"{key}_worst_group"] = worst_name
        out[f"{key}_best_group"] = best_name
    return out


def _uncertainty_breakdown(
    *,
    name: str,
    scores: np.ndarray,
    pred_quantiles: np.ndarray,
    realized: np.ndarray,
    quantile_levels: np.ndarray,
    barrier_probs: np.ndarray | None = None,
    barrier_labels: np.ndarray | None = None,
) -> dict[str, Any]:
    point_errors = np.abs(_median_prediction(pred_quantiles, quantile_levels) - realized)
    out: dict[str, Any] = {
        "correlation_abs_error": _correlation(scores, point_errors),
        "buckets": {},
    }
    edges = np.quantile(scores, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    if np.unique(np.round(edges, 12)).size < 4:
        out["buckets"]["all"] = _bucket_metrics(
            name=name,
            scores=scores,
            mask=np.ones(scores.shape[0], dtype=bool),
            pred_quantiles=pred_quantiles,
            realized=realized,
            quantile_levels=quantile_levels,
            barrier_probs=barrier_probs,
            barrier_labels=barrier_labels,
        )
        return out

    bucket_names = ("low", "mid", "high")
    for idx, bucket_name in enumerate(bucket_names):
        lo = edges[idx]
        hi = edges[idx + 1]
        if idx == len(bucket_names) - 1:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        if not mask.any():
            continue
        out["buckets"][bucket_name] = _bucket_metrics(
            name=name,
            scores=scores,
            mask=mask,
            pred_quantiles=pred_quantiles,
            realized=realized,
            quantile_levels=quantile_levels,
            barrier_probs=barrier_probs,
            barrier_labels=barrier_labels,
        )
    return out


def _bucket_metrics(
    *,
    name: str,
    scores: np.ndarray,
    mask: np.ndarray,
    pred_quantiles: np.ndarray,
    realized: np.ndarray,
    quantile_levels: np.ndarray,
    barrier_probs: np.ndarray | None = None,
    barrier_labels: np.ndarray | None = None,
) -> dict[str, float]:
    metrics = _slice_metrics(
        pred_quantiles=pred_quantiles[mask],
        realized=realized[mask],
        quantile_levels=quantile_levels,
        barrier_probs=barrier_probs[mask] if barrier_probs is not None else None,
        barrier_labels=barrier_labels[mask] if barrier_labels is not None else None,
    )
    metrics["count"] = float(mask.sum())
    metrics[f"{name}_min"] = float(scores[mask].min())
    metrics[f"{name}_max"] = float(scores[mask].max())
    return metrics


def _expanding_quantile_baseline(
    realized: np.ndarray,
    quantile_levels: np.ndarray,
    *,
    min_history: int,
) -> np.ndarray:
    out = np.zeros((realized.shape[0], quantile_levels.shape[0]), dtype=float)
    fallback = np.zeros(quantile_levels.shape[0], dtype=float)
    for idx in range(realized.shape[0]):
        history = realized[:idx]
        if history.size >= max(1, min_history):
            out[idx] = np.quantile(history, quantile_levels)
        else:
            out[idx] = fallback
    return out


def _expanding_barrier_prior_baseline(
    labels: np.ndarray,
    n_classes: int,
    min_history: int,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    out = np.full((labels.shape[0], n_classes), 1.0 / n_classes, dtype=float)
    counts = np.zeros(n_classes, dtype=float)
    for idx, label in enumerate(labels):
        if idx >= max(1, min_history):
            out[idx] = (counts + alpha) / (counts.sum() + alpha * n_classes)
        counts[int(label)] += 1.0
    return out


def _expanding_mean_baseline(values: np.ndarray, min_history: int) -> np.ndarray:
    out = np.zeros(values.shape[0], dtype=float)
    for idx in range(values.shape[0]):
        history = values[:idx]
        if history.size >= max(1, min_history):
            out[idx] = float(history.mean())
        else:
            out[idx] = float(values[0])
    return out


def _median_prediction(pred_quantiles: np.ndarray, quantile_levels: np.ndarray) -> np.ndarray:
    idx = _median_index(quantile_levels)
    return pred_quantiles[:, idx]


def _median_index(quantile_levels: np.ndarray) -> int:
    exact = np.where(np.isclose(quantile_levels, 0.5))[0]
    if exact.size:
        return int(exact[0])
    return int(np.argmin(np.abs(quantile_levels - 0.5)))


def _resolve_quantile_levels(pred_quantiles: np.ndarray, quantile_levels: Any) -> np.ndarray:
    if quantile_levels is None:
        if pred_quantiles.shape[1] == len(QUANTILE_LEVELS):
            return np.asarray(QUANTILE_LEVELS, dtype=float)
        return np.linspace(
            1.0 / (pred_quantiles.shape[1] + 1),
            pred_quantiles.shape[1] / (pred_quantiles.shape[1] + 1),
            pred_quantiles.shape[1],
        )
    levels = np.asarray(quantile_levels, dtype=float).reshape(-1)
    if levels.shape != (pred_quantiles.shape[1],):
        raise ValueError(
            f"quantile_levels shape {levels.shape} does not match q.shape[1]={pred_quantiles.shape[1]}"
        )
    return levels


def _maybe_1d(
    value: Any,
    expected_len: int,
    name: str,
    *,
    dtype: Any = float,
) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=dtype).reshape(-1)
    if arr.shape[0] != expected_len:
        raise ValueError(f"{name} must have length {expected_len}; got {arr.shape[0]}")
    return arr


def _maybe_2d(value: Any, expected_len: int, name: str) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != expected_len:
        raise ValueError(f"{name} must be [N, K] with N={expected_len}; got shape {arr.shape}")
    return arr


def _lower_is_better_skill(model_value: float, baseline_value: float) -> float:
    if abs(baseline_value) <= _EPS:
        return 0.0
    return float(1.0 - (model_value / baseline_value))


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    corr = np.corrcoef(x, y)[0, 1]
    return float(0.0 if np.isnan(corr) else corr)


def _label_key(label: Any) -> str:
    if label is None:
        return "unknown"
    if isinstance(label, (float, np.floating)) and np.isnan(label):
        return "unknown"
    if hasattr(label, "value"):
        return str(label.value)
    return str(label)


__all__ = ["evaluate_predictive_outputs"]
