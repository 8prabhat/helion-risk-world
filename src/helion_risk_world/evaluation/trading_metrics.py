"""Trading metrics (SPEC.md §22, Day 7).

Pure, stateless functions over per-step returns / PnL / an equity curve. No I/O. Used by the
backtest report and any evaluation harness so the definitions live in exactly one place (DRY).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np

_SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60
_MIN_ANNUALIZATION_SPAN_SECONDS = 24 * 60 * 60


def sharpe(step_returns: Sequence[float], periods_per_year: int = 1) -> float:
    """Annualised Sharpe of per-step returns. Returns 0.0 if there is no dispersion."""
    r = np.asarray(step_returns, dtype=float)
    if r.size < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))


def sortino(step_returns: Sequence[float], periods_per_year: int = 1) -> float:
    """Annualized Sortino ratio using downside deviation only."""
    r = np.asarray(step_returns, dtype=float)
    if r.size < 2:
        return 0.0
    downside = r[r < 0]
    if downside.size == 0:
        return float("inf") if r.mean() > 0 else 0.0
    downside_std = downside.std(ddof=1) if downside.size > 1 else abs(float(downside[0]))
    if downside_std == 0:
        return 0.0
    return float(r.mean() / downside_std * np.sqrt(periods_per_year))


def infer_periods_per_year(timestamps: Sequence[datetime]) -> int:
    """Infer annualisation periods from the median bar spacing.

    Uses calendar time so intraday bars annualise correctly without requiring a separate exchange
    calendar dependency. Returns 1 when the spacing cannot be inferred safely.
    """
    if len(timestamps) < 2:
        return 1
    deltas = np.array(
        [
            (b - a).total_seconds()
            for a, b in zip(timestamps, timestamps[1:], strict=False)
            if b > a
        ],
        dtype=float,
    )
    if deltas.size == 0:
        return 1
    median_seconds = float(np.median(deltas))
    if median_seconds <= 0:
        return 1
    return max(1, int(round((365.25 * 24 * 60 * 60) / median_seconds)))


def annualized_return(equity: Sequence[float], timestamps: Sequence[datetime]) -> float:
    """Compound annual growth rate over the observed time span.

    Falls back to total return when the elapsed span cannot be inferred safely or is too short to
    annualize stably.
    """
    e = np.asarray(equity, dtype=float)
    if e.size < 2:
        return 0.0
    initial = float(e[0])
    final = float(e[-1])
    if initial <= 0:
        return 0.0
    total_return = final / initial - 1.0
    if len(timestamps) < 2:
        return float(total_return)
    elapsed_seconds = float((timestamps[-1] - timestamps[0]).total_seconds())
    if elapsed_seconds <= 0 or elapsed_seconds < _MIN_ANNUALIZATION_SPAN_SECONDS:
        return float(total_return)
    if final <= 0:
        return -1.0
    years = elapsed_seconds / _SECONDS_PER_YEAR
    if years <= 0:
        return float(total_return)
    return float((final / initial) ** (1.0 / years) - 1.0)


def max_drawdown(equity: Sequence[float]) -> float:
    """Maximum peak-to-trough drawdown of an equity curve, as a positive fraction in [0, 1]."""
    e = np.asarray(equity, dtype=float)
    if e.size == 0:
        return 0.0
    peak = np.maximum.accumulate(e)
    dd = (peak - e) / np.where(peak == 0, 1.0, peak)
    return float(dd.max())


def drawdown_duration(equity: Sequence[float]) -> float:
    """Longest consecutive drawdown spell in bars."""
    e = np.asarray(equity, dtype=float)
    if e.size == 0:
        return 0.0
    peak = np.maximum.accumulate(e)
    dd = (peak - e) / np.where(peak == 0, 1.0, peak)
    duration = 0
    current = 0
    for value in dd:
        if value > 0:
            current += 1
            duration = max(duration, current)
        else:
            current = 0
    return float(duration)


def profit_factor(pnls: Sequence[float]) -> float:
    """Gross profit / gross loss. inf if there are no losses (and some profit)."""
    p = np.asarray(pnls, dtype=float)
    gains = p[p > 0].sum()
    losses = -p[p < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def payoff_ratio(pnls: Sequence[float]) -> float:
    """Average winner size divided by average loser size magnitude."""
    p = np.asarray(pnls, dtype=float)
    wins = p[p > 0]
    losses = -p[p < 0]
    if wins.size == 0 or losses.size == 0:
        return 0.0
    return float(wins.mean() / losses.mean())


def hit_rate(pnls: Sequence[float]) -> float:
    """Fraction of non-zero outcomes that are profitable."""
    p = np.asarray(pnls, dtype=float)
    nonzero = p[p != 0]
    if nonzero.size == 0:
        return 0.0
    return float((nonzero > 0).mean())


def expectancy(trade_returns: Sequence[float]) -> float:
    """Average realized return per trade; 0.0 when there are no trades."""
    r = np.asarray(trade_returns, dtype=float)
    if r.size == 0:
        return 0.0
    return float(r.mean())


def turnover(turnover_fractions: Sequence[float]) -> float:
    """Cumulative traded notional as a fraction of capital over the run."""
    t = np.asarray(turnover_fractions, dtype=float)
    if t.size == 0:
        return 0.0
    return float(t.sum())


def calmar(annualized_return_value: float, max_drawdown_value: float) -> float:
    """Annualized return divided by max drawdown."""
    if max_drawdown_value <= 0:
        return float("inf") if annualized_return_value > 0 else 0.0
    return float(annualized_return_value / max_drawdown_value)
