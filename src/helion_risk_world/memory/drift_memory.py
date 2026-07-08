from __future__ import annotations

from typing import Any

import numpy as np


class DriftMonitor:
    """Detects when paper/live performance deviates from backtest expectations (SPEC.md §23 #15)."""

    def __init__(self, alert_threshold: float = 0.3) -> None:
        self._alert_threshold = alert_threshold

    def check(
        self, backtest_stats: dict[str, float], live_stats: dict[str, float]
    ) -> dict[str, float]:
        if not backtest_stats or not live_stats:
            return {"drift_score": 0.0, "drift_alert": 0.0}

        higher_is_better = (
            "net_pnl",
            "total_return",
            "annualized_return",
            "sharpe",
            "hit_rate",
            "profit_factor",
        )
        lower_is_better = ("max_drawdown", "avg_slippage", "reject_rate")
        out: dict[str, float] = {}
        penalties: list[float] = []

        for key in higher_is_better:
            if key not in backtest_stats or key not in live_stats:
                continue
            base = abs(float(backtest_stats[key])) or 1.0
            delta = float(live_stats[key]) - float(backtest_stats[key])
            out[f"{key}_delta"] = delta
            penalty = max(0.0, -delta / base)
            penalties.append(penalty)

        for key in lower_is_better:
            if key not in backtest_stats or key not in live_stats:
                continue
            base = abs(float(backtest_stats[key])) or 1.0
            delta = float(live_stats[key]) - float(backtest_stats[key])
            out[f"{key}_delta"] = delta
            penalty = max(0.0, delta / base)
            penalties.append(penalty)

        if "turnover" in backtest_stats and "turnover" in live_stats:
            base = abs(float(backtest_stats["turnover"])) or 1.0
            delta = float(live_stats["turnover"]) - float(backtest_stats["turnover"])
            out["turnover_delta"] = delta
            penalties.append(abs(delta) / base)

        score = float(sum(penalties) / len(penalties)) if penalties else 0.0
        out["drift_score"] = score
        out["drift_alert"] = 1.0 if score >= self._alert_threshold else 0.0
        return out

    def check_distributions(
        self,
        reference: dict[str, Any],
        current: dict[str, Any],
    ) -> dict[str, float]:
        """Monitor feature/label/prediction/confidence distribution drift.

        Expected keys are optional. Numeric arrays are compared with PSI and a
        lightweight two-sample KS statistic; categorical arrays are compared by total
        variation distance. Missing keys are skipped.
        """
        out: dict[str, float] = {}
        penalties: list[float] = []
        for key in ("features", "confidence", "prediction", "label"):
            if key not in reference or key not in current:
                continue
            ref = np.asarray(reference[key])
            cur = np.asarray(current[key])
            if ref.size == 0 or cur.size == 0:
                continue
            if np.issubdtype(ref.dtype, np.number) and np.issubdtype(cur.dtype, np.number):
                ref_1d = ref.astype(float).reshape(-1)
                cur_1d = cur.astype(float).reshape(-1)
                psi = _psi(ref_1d, cur_1d)
                ks = _ks_stat(ref_1d, cur_1d)
                out[f"{key}_psi"] = psi
                out[f"{key}_ks"] = ks
                penalties.append(max(psi, ks))
            else:
                tvd = _categorical_tvd(ref.reshape(-1), cur.reshape(-1))
                out[f"{key}_tvd"] = tvd
                penalties.append(tvd)

        if "regime_performance" in reference and "regime_performance" in current:
            degradation = _regime_degradation(
                reference["regime_performance"],
                current["regime_performance"],
            )
            out["regime_performance_degradation"] = degradation
            penalties.append(degradation)

        score = float(sum(penalties) / len(penalties)) if penalties else 0.0
        out["distribution_drift_score"] = score
        out["distribution_drift_alert"] = 1.0 if score >= self._alert_threshold else 0.0
        return out


def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if reference.size == 0 or current.size == 0:
        return 0.0
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if edges.size < 2:
        return 0.0
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_pct = np.maximum(ref_counts / max(ref_counts.sum(), 1), 1e-6)
    cur_pct = np.maximum(cur_counts / max(cur_counts.sum(), 1), 1e-6)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _ks_stat(reference: np.ndarray, current: np.ndarray) -> float:
    ref = np.sort(reference[np.isfinite(reference)])
    cur = np.sort(current[np.isfinite(current)])
    if ref.size == 0 or cur.size == 0:
        return 0.0
    values = np.sort(np.unique(np.concatenate([ref, cur])))
    ref_cdf = np.searchsorted(ref, values, side="right") / ref.size
    cur_cdf = np.searchsorted(cur, values, side="right") / cur.size
    return float(np.max(np.abs(ref_cdf - cur_cdf)))


def _categorical_tvd(reference: np.ndarray, current: np.ndarray) -> float:
    keys = sorted(set(reference.tolist()) | set(current.tolist()))
    if not keys:
        return 0.0
    ref_counts = np.array([np.sum(reference == key) for key in keys], dtype=float)
    cur_counts = np.array([np.sum(current == key) for key in keys], dtype=float)
    ref_pct = ref_counts / max(ref_counts.sum(), 1.0)
    cur_pct = cur_counts / max(cur_counts.sum(), 1.0)
    return float(0.5 * np.abs(ref_pct - cur_pct).sum())


def _regime_degradation(reference: Any, current: Any) -> float:
    if not isinstance(reference, dict) or not isinstance(current, dict):
        return 0.0
    penalties: list[float] = []
    for regime, ref_payload in reference.items():
        cur_payload = current.get(regime)
        if not isinstance(ref_payload, dict) or not isinstance(cur_payload, dict):
            continue
        ref_ret = float(ref_payload.get("mean_step_return", ref_payload.get("return", 0.0)))
        cur_ret = float(cur_payload.get("mean_step_return", cur_payload.get("return", 0.0)))
        base = abs(ref_ret) or 1.0
        penalties.append(max(0.0, (ref_ret - cur_ret) / base))
    return float(sum(penalties) / len(penalties)) if penalties else 0.0
