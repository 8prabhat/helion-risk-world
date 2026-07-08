"""Calibration metrics and Stage-5 gate (SPEC.md §21 Stage 5, §26).

``compute()`` — raw Brier score, ECE, quantile coverage error.
``CalibrationGate`` — PASS/FAIL gate that blocks backtesting until calibration criteria are met.

Gate criteria (all must pass):
  1. Mean quantile coverage error < coverage_tol for each quantile level.
  2. Barrier Brier score < barrier_brier_max.
  3. Barrier ECE < barrier_ece_max.
  4. Per-regime coverage parity (optional) — when regime_labels supplied, each regime's
     coverage error must also satisfy the tolerance.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from helion_risk_world.schemas.prediction_schema import QUANTILE_LEVELS


def compute(*args: Any, **kwargs: Any) -> dict[str, float]:
    """Calibration: Brier, ECE, quantile coverage.

    Keyword args:
        pred_quantiles: [N, Q] predicted quantile values.
        realized:       [N]    realized outcomes.
        barrier_probs:  [N, 3] predicted barrier probabilities (stop/target/timeout).
        barrier_labels: [N]    integer barrier outcome labels (0=stop, 1=target, 2=timeout).

    Returns dict with keys: coverage_error, interval_width, barrier_brier, barrier_ece.
    """
    quantiles = kwargs.get("pred_quantiles")
    realized = kwargs.get("realized")
    barrier_probs = kwargs.get("barrier_probs")
    barrier_labels = kwargs.get("barrier_labels")
    out: dict[str, float] = {}

    if quantiles is not None and realized is not None:
        q = np.asarray(quantiles, dtype=float)
        y = np.asarray(realized, dtype=float).reshape(-1)
        levels_arg = kwargs.get("quantile_levels")
        if levels_arg is None:
            levels = (
                np.asarray(QUANTILE_LEVELS, dtype=float)
                if q.shape[1] == len(QUANTILE_LEVELS)
                else np.linspace(1.0 / (q.shape[1] + 1), q.shape[1] / (q.shape[1] + 1), q.shape[1])
            )
        else:
            levels = np.asarray(levels_arg, dtype=float)
            if levels.shape != (q.shape[1],):
                raise ValueError(
                    f"quantile_levels shape {levels.shape} does not match q.shape[1]={q.shape[1]}"
                )
        coverage = (y[:, None] <= q).mean(axis=0)
        out["coverage_error"] = float(np.abs(coverage - levels).mean())
        out["interval_width"] = float(np.mean(q[:, -1] - q[:, 0])) if q.shape[1] > 1 else 0.0

    if barrier_probs is not None and barrier_labels is not None:
        probs = np.asarray(barrier_probs, dtype=float)
        labels = np.asarray(barrier_labels, dtype=int).reshape(-1)
        one_hot = np.eye(probs.shape[1])[labels]
        out["barrier_brier"] = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
        correct = (pred == labels).astype(float)
        bins = np.linspace(0.0, 1.0, 11)
        ece = 0.0
        for lo, hi in zip(bins[:-1], bins[1:], strict=False):
            mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
            if mask.any():
                ece += abs(correct[mask].mean() - conf[mask].mean()) * mask.mean()
        out["barrier_ece"] = float(ece)
    return out


class CalibrationGate:
    """Stage-5 pass/fail gate before backtest (SPEC.md §21 Stage 5).

    Checks that the model is calibrated before allowing the backtest to proceed.
    Returns PASS only when ALL active criteria are satisfied.

    Args:
        coverage_tol:      maximum allowed mean quantile coverage error (default 0.05).
        barrier_brier_max: maximum Brier score for barrier outcome predictions (default 0.25).
        barrier_ece_max:   maximum ECE for barrier outcome predictions (default 0.10).
    """

    def __init__(
        self,
        coverage_tol: float = 0.05,
        barrier_brier_max: float = 0.25,
        barrier_ece_max: float = 0.10,
    ) -> None:
        if coverage_tol <= 0 or barrier_brier_max <= 0 or barrier_ece_max <= 0:
            raise ValueError("All thresholds must be positive")
        self._coverage_tol = coverage_tol
        self._barrier_brier_max = barrier_brier_max
        self._barrier_ece_max = barrier_ece_max

    def check(
        self,
        pred_quantiles: Any = None,
        realized: Any = None,
        barrier_probs: Any = None,
        barrier_labels: Any = None,
        regime_labels: Any = None,
        quantile_levels: Any = None,
    ) -> tuple[bool, dict[str, str]]:
        """Run all active calibration checks.

        Args:
            pred_quantiles:  [N, Q] predicted quantile values.
            realized:        [N]    realized outcomes.
            barrier_probs:   [N, 3] predicted barrier probabilities.
            barrier_labels:  [N]    integer barrier outcome labels.
            regime_labels:   [N]    optional regime ids — enables per-regime coverage check.

        Returns:
            (passed: bool, reasons: dict[str, str])
            ``reasons`` maps check name → "PASS <value>" or "FAIL <value> > <threshold>".
        """
        metrics = compute(
            pred_quantiles=pred_quantiles,
            realized=realized,
            barrier_probs=barrier_probs,
            barrier_labels=barrier_labels,
            quantile_levels=quantile_levels,
        )
        reasons: dict[str, str] = {}
        passed = True

        if not metrics:
            return False, {"no_metrics": "FAIL no calibration inputs supplied"}

        # Check 1: quantile coverage error
        if "coverage_error" in metrics:
            val = metrics["coverage_error"]
            ok = val < self._coverage_tol
            reasons["coverage_error"] = (
                f"PASS {val:.4f} < {self._coverage_tol}"
                if ok else
                f"FAIL {val:.4f} >= {self._coverage_tol}"
            )
            passed = passed and ok

        # Check 2: barrier Brier
        if "barrier_brier" in metrics:
            val = metrics["barrier_brier"]
            ok = val < self._barrier_brier_max
            reasons["barrier_brier"] = (
                f"PASS {val:.4f} < {self._barrier_brier_max}"
                if ok else
                f"FAIL {val:.4f} >= {self._barrier_brier_max}"
            )
            passed = passed and ok

        # Check 3: barrier ECE
        if "barrier_ece" in metrics:
            val = metrics["barrier_ece"]
            ok = val < self._barrier_ece_max
            reasons["barrier_ece"] = (
                f"PASS {val:.4f} < {self._barrier_ece_max}"
                if ok else
                f"FAIL {val:.4f} >= {self._barrier_ece_max}"
            )
            passed = passed and ok

        # Check 4 (optional): per-regime coverage parity
        if (
            regime_labels is not None
            and pred_quantiles is not None
            and realized is not None
        ):
            q = np.asarray(pred_quantiles, dtype=float)
            y = np.asarray(realized, dtype=float).reshape(-1)
            levels = (
                np.asarray(quantile_levels, dtype=float)
                if quantile_levels is not None
                else (
                    np.asarray(QUANTILE_LEVELS, dtype=float)
                    if q.shape[1] == len(QUANTILE_LEVELS)
                    else np.linspace(
                        1.0 / (q.shape[1] + 1),
                        q.shape[1] / (q.shape[1] + 1),
                        q.shape[1],
                    )
                )
            )
            regimes = np.asarray(regime_labels, dtype=object).reshape(-1)
            for regime_id in np.unique(regimes):
                if regime_id is None or (isinstance(regime_id, float) and np.isnan(regime_id)):
                    continue
                mask = regimes == regime_id
                if mask.sum() < 10:
                    reasons[f"coverage_regime_{regime_id}"] = f"SKIP n={mask.sum()} < 10"
                    continue
                cov = (y[mask, None] <= q[mask]).mean(axis=0)
                err = float(np.abs(cov - levels).mean())
                ok = err < self._coverage_tol
                reasons[f"coverage_regime_{regime_id}"] = (
                    f"PASS {err:.4f}" if ok else f"FAIL {err:.4f} >= {self._coverage_tol}"
                )
                passed = passed and ok

        return passed, reasons


__all__ = ["compute", "CalibrationGate"]
