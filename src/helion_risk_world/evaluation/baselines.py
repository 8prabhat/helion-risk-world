"""Trading baseline comparisons and paired block-bootstrap tests.

These metrics are separate from DSR. DSR answers whether a strategy's own Sharpe survives
trial-deflation; the baseline tests answer whether it beats simple alternatives after costs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def paired_block_bootstrap(
    strategy_returns: Sequence[float],
    baseline_returns: Sequence[float],
    *,
    block_size: int = 5,
    n_bootstrap: int = 1000,
    seed: int = 7,
) -> dict[str, float | int]:
    """Bootstrap the mean return difference using contiguous resampled blocks."""
    strat = np.asarray(strategy_returns, dtype=float)
    base = np.asarray(baseline_returns, dtype=float)
    if strat.shape != base.shape:
        raise ValueError("strategy_returns and baseline_returns must have the same shape")
    if strat.size == 0:
        return {
            "observed_diff_mean": 0.0,
            "bootstrap_diff_mean": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "p_value": 1.0,
            "n_obs": 0,
            "block_size": block_size,
            "n_bootstrap": n_bootstrap,
        }

    diff = strat - base
    observed = float(diff.mean())
    if diff.size == 1:
        return {
            "observed_diff_mean": observed,
            "bootstrap_diff_mean": observed,
            "ci_low": observed,
            "ci_high": observed,
            "p_value": 0.0 if observed > 0 else 1.0,
            "n_obs": 1,
            "block_size": 1,
            "n_bootstrap": 1,
        }

    rng = np.random.default_rng(seed)
    size = int(max(1, min(block_size, diff.size)))
    samples = np.empty(n_bootstrap, dtype=float)
    max_start = max(1, diff.size - size + 1)
    for i in range(n_bootstrap):
        pieces: list[np.ndarray] = []
        while sum(piece.size for piece in pieces) < diff.size:
            start = int(rng.integers(0, max_start))
            pieces.append(diff[start : start + size])
        sample = np.concatenate(pieces)[: diff.size]
        samples[i] = float(sample.mean())

    return {
        "observed_diff_mean": observed,
        "bootstrap_diff_mean": float(samples.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "p_value": float((samples <= 0.0).mean()),
        "n_obs": int(diff.size),
        "block_size": size,
        "n_bootstrap": int(n_bootstrap),
    }


def flat_baseline(n_steps: int) -> np.ndarray:
    """Always-flat baseline returns."""
    return np.zeros(int(n_steps), dtype=float)


def buy_and_hold_baseline(
    market_returns: Sequence[float],
    *,
    cost_rate_frac: float = 0.0,
) -> np.ndarray:
    """Long-only buy-and-hold baseline with one entry and one exit cost."""
    baseline = np.asarray(market_returns, dtype=float).copy()
    if baseline.size == 0:
        return baseline
    baseline[0] -= cost_rate_frac
    baseline[-1] -= cost_rate_frac
    return baseline


def random_matched_turnover_baseline(
    market_returns: Sequence[float],
    exposure_path: Sequence[float],
    turnover_fractions: Sequence[float],
    *,
    cost_rate_frac: float = 0.0,
    n_trials: int = 32,
    seed: int = 7,
) -> np.ndarray:
    """Random-sign baseline with the same exposure magnitude and turnover cost path."""
    market = np.asarray(market_returns, dtype=float)
    exposure = np.clip(np.abs(np.asarray(exposure_path, dtype=float)), 0.0, 1.0)
    turnover = np.asarray(turnover_fractions, dtype=float)
    if not (market.shape == exposure.shape == turnover.shape):
        raise ValueError("market_returns, exposure_path, and turnover_fractions must have the same shape")
    if market.size == 0:
        return np.zeros((int(n_trials), 0), dtype=float)
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(int(n_trials), market.size))
    baseline = signs * exposure[None, :] * market[None, :]
    baseline -= cost_rate_frac * turnover[None, :]
    baseline[:, exposure <= 1e-12] = 0.0
    return baseline


def compare_to_baselines(
    strategy_returns: Sequence[float],
    market_returns: Sequence[float],
    turnover_fractions: Sequence[float],
    exposure_path: Sequence[float],
    *,
    total_cost: float,
    capital0: float,
    block_size: int = 5,
    n_bootstrap: int = 1000,
    random_trials: int = 32,
    seed: int = 7,
) -> dict[str, Any]:
    """Compare strategy returns against flat, buy-hold, and random matched-turnover baselines."""
    strat = np.asarray(strategy_returns, dtype=float)
    market = np.asarray(market_returns, dtype=float)
    turnover = np.asarray(turnover_fractions, dtype=float)
    exposure = np.asarray(exposure_path, dtype=float)
    if not (strat.shape == market.shape == turnover.shape == exposure.shape):
        raise ValueError("all baseline comparison inputs must have the same shape")

    turnover_total = float(turnover.sum())
    if capital0 <= 0:
        raise ValueError("capital0 must be positive")
    cost_rate_frac = (total_cost / capital0) / max(turnover_total, 1e-12) if turnover_total > 0 else 0.0

    flat = flat_baseline(strat.size)
    buy_hold = buy_and_hold_baseline(market, cost_rate_frac=cost_rate_frac)
    random_trials_returns = random_matched_turnover_baseline(
        market,
        exposure,
        turnover,
        cost_rate_frac=cost_rate_frac,
        n_trials=random_trials,
        seed=seed,
    )

    flat_stats = paired_block_bootstrap(
        strat, flat, block_size=block_size, n_bootstrap=n_bootstrap, seed=seed
    )
    buy_hold_stats = paired_block_bootstrap(
        strat, buy_hold, block_size=block_size, n_bootstrap=n_bootstrap, seed=seed + 1
    )
    random_stats = [
        paired_block_bootstrap(
            strat,
            random_trials_returns[i],
            block_size=block_size,
            n_bootstrap=n_bootstrap,
            seed=seed + 2 + i,
        )
        for i in range(random_trials_returns.shape[0])
    ]

    return {
        "cost_rate_frac": cost_rate_frac,
        "flat": {
            "baseline_mean_return": float(flat.mean()) if flat.size else 0.0,
            **flat_stats,
        },
        "buy_hold": {
            "baseline_mean_return": float(buy_hold.mean()) if buy_hold.size else 0.0,
            **buy_hold_stats,
        },
        "random_matched_turnover": {
            "baseline_mean_return": float(random_trials_returns.mean())
            if random_trials_returns.size
            else 0.0,
            "median_observed_diff_mean": float(
                np.median([float(stat["observed_diff_mean"]) for stat in random_stats])
            )
            if random_stats
            else 0.0,
            "median_p_value": float(np.median([float(stat["p_value"]) for stat in random_stats]))
            if random_stats
            else 1.0,
            "trials": int(random_trials_returns.shape[0]),
        },
    }


__all__ = [
    "buy_and_hold_baseline",
    "compare_to_baselines",
    "flat_baseline",
    "paired_block_bootstrap",
    "random_matched_turnover_baseline",
]
