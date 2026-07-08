"""Deflated Sharpe Ratio (DSR) for trial-corrected significance (SPEC.md §23.3, §26).

Reference: López de Prado (2018) "Advances in Financial Machine Learning", Ch. 14.

DSR answers: "Is the strategy's Sharpe ratio real after correcting for the number
of backtested strategies (trial selection bias) and non-normality?"

This is SEPARATE from the baseline-comparison test (§23.3):
  (a) DSR > 0 — the strategy's own Sharpe is real after trial-deflation.
  (b) Paired block-bootstrap — the strategy beats buy-and-hold/flat AFTER costs.

DSR alone does not prove baseline-beating.

SRP: DSR computation only — baseline bootstrap lives in evaluation/baselines.py.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats  # type: ignore[import-untyped]


def _sr_from_returns(returns: np.ndarray, annualisation: float = 252.0) -> float:
    """Annualised Sharpe Ratio from a 1-D array of per-period returns."""
    mu = float(returns.mean())
    sigma = float(returns.std(ddof=1))
    if sigma < 1e-12:
        return 0.0
    return (mu / sigma) * math.sqrt(annualisation)


def deflated_sharpe_ratio(
    returns: np.ndarray,
    benchmark_sharpe: float = 0.0,
    n_trials: int = 1,
    annualisation: float = 252.0,
) -> dict[str, float]:
    """Compute the Deflated Sharpe Ratio (PSR variant).

    Parameters
    ----------
    returns:          per-period strategy returns (e.g. per-bar, per-day)
    benchmark_sharpe: the Sharpe ratio expected by chance from the best of
                      ``n_trials`` independent trials (use 0 for a single trial)
    n_trials:         number of total strategy variants tested
    annualisation:    periods per year (252 for daily, 78 for 5-min NSE session bars)

    Returns
    -------
    dict with keys: observed_sr, benchmark_sr, psr, dsr, z_stat, p_value
    """
    returns = np.asarray(returns, dtype=float)
    T = len(returns)
    if T < 4:
        raise ValueError("Need at least 4 return observations for DSR")

    observed_sr = _sr_from_returns(returns, annualisation)

    # Adjust benchmark for trials: E[max SR over n_trials i.i.d. standard normals]
    # Approximation: E[max Z_n] ≈ (1 - γ) * Φ^{-1}(1 - 1/n) + γ * Φ^{-1}(1 - 1/(n*e))
    # where γ ≈ 0.5772 (Euler–Mascheroni). Simplified version used here:
    if n_trials > 1:
        z_max = stats.norm.ppf(1.0 - 1.0 / n_trials)
    else:
        z_max = 0.0
    adjusted_benchmark = max(benchmark_sharpe, z_max / math.sqrt(annualisation))

    # Higher-order adjustment for non-normality (skew/excess-kurtosis)
    mu3 = float(stats.skew(returns))
    mu4 = float(stats.kurtosis(returns, fisher=True))  # excess kurtosis
    # Variance of the SR estimator (Mertens 2002 correction)
    var_sr_ratio = (
        1.0
        - mu3 * (observed_sr / math.sqrt(annualisation))
        + (mu4 - 1.0) / 4.0 * (observed_sr / math.sqrt(annualisation)) ** 2
    ) / T

    if var_sr_ratio <= 0:
        var_sr_ratio = 1.0 / T

    # PSR (Probabilistic Sharpe Ratio): P(true SR > benchmark)
    sr_annual = observed_sr
    sr_bench_annual = adjusted_benchmark * math.sqrt(annualisation)
    z_stat = (sr_annual - sr_bench_annual) / (math.sqrt(var_sr_ratio * annualisation))
    psr = float(stats.norm.cdf(z_stat))

    # DSR = observed SR × (1 − PSR_needed)  — just report PSR and let caller threshold
    dsr = observed_sr if psr > 0.95 else 0.0

    return {
        "observed_sr": observed_sr,
        "benchmark_sr": adjusted_benchmark * math.sqrt(annualisation),
        "psr": psr,
        "dsr": dsr,
        "z_stat": z_stat,
        "p_value": 1.0 - psr,
        "n_obs": T,
        "n_trials": n_trials,
    }


__all__ = ["deflated_sharpe_ratio"]
