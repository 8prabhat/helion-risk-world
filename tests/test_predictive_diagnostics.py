"""Predictive diagnostics coverage."""

from __future__ import annotations

import numpy as np

from helion_risk_world.evaluation.predictive_diagnostics import evaluate_predictive_outputs


def test_predictive_diagnostics_reports_baselines_regimes_and_uncertainty() -> None:
    n = 90
    realized = np.linspace(-0.03, 0.03, n)
    noise = np.linspace(0.0005, 0.004, n)
    median = realized + noise
    width = 0.01
    pred_quantiles = np.column_stack(
        [
            median - 2.0 * width,
            median - width,
            median,
            median + width,
            median + 2.0 * width,
        ]
    )

    barrier_labels = np.where(realized > 0.007, 1, np.where(realized < -0.007, 0, 2))
    barrier_probs = np.tile(np.array([0.1, 0.1, 0.1], dtype=float), (n, 1))
    for idx, label in enumerate(barrier_labels):
        barrier_probs[idx, label] = 0.8

    realized_volatility = 0.01 + np.linspace(0.0, 0.006, n)
    predicted_volatility = realized_volatility + noise / 4.0
    regime_labels = np.array(["trend"] * 30 + ["range"] * 30 + ["high_vol"] * 30, dtype=object)
    epistemic = np.linspace(0.05, 0.35, n)
    ood_scores = np.linspace(0.1, 1.0, n)

    report = evaluate_predictive_outputs(
        pred_quantiles=pred_quantiles,
        realized=realized,
        barrier_probs=barrier_probs,
        barrier_labels=barrier_labels,
        quantile_levels=np.array([0.1, 0.25, 0.5, 0.75, 0.9], dtype=float),
        predicted_volatility=predicted_volatility,
        realized_volatility=realized_volatility,
        regime_labels=regime_labels,
        epistemic=epistemic,
        ood_scores=ood_scores,
        baseline_min_history=8,
    )

    assert report["samples"] == n
    assert report["baseline_comparison"]["point"]["mae_skill"] > 0.0
    assert report["baseline_comparison"]["barrier"]["brier_skill"] > 0.0
    assert set(report["regime_breakdown"]) == {"trend", "range", "high_vol"}
    assert report["regime_breakdown"]["trend"]["count"] == 30.0
    assert "rollout_mae_gap" in report["regime_parity"]
    assert report["uncertainty_breakdown"]["ood_score"]["correlation_abs_error"] > 0.0
    assert set(report["uncertainty_breakdown"]["epistemic"]["buckets"]) == {"low", "mid", "high"}
