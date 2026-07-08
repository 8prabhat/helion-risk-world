"""Higher-level diagnostics for backtest reports."""

from __future__ import annotations

from typing import Any

import numpy as np

from helion_risk_world.backtesting.backtest_engine import BacktestReport
from helion_risk_world.backtesting.deflated_sharpe import deflated_sharpe_ratio
from helion_risk_world.backtesting.event_stress_test import EventStressTest
from helion_risk_world.evaluation.baselines import compare_to_baselines
from helion_risk_world.evaluation import trading_metrics as tm
from helion_risk_world.schemas.action_schema import ActionType


def evaluate_backtest_report(
    report: BacktestReport,
    *,
    n_trials: int = 1,
    block_size: int = 5,
    n_bootstrap: int = 1000,
    random_trials: int = 32,
    seed: int = 7,
) -> dict[str, Any]:
    """Compute DSR, baseline comparisons, and stress slices for one report."""
    annualisation = tm.infer_periods_per_year(report.timestamps)
    if len(report.step_returns) >= 4:
        dsr = deflated_sharpe_ratio(
            np.asarray(report.step_returns, dtype=float),
            n_trials=max(1, int(n_trials)),
            annualisation=float(annualisation),
        )
    else:
        dsr = {
            "observed_sr": 0.0,
            "benchmark_sr": 0.0,
            "psr": 0.0,
            "dsr": 0.0,
            "z_stat": 0.0,
            "p_value": 1.0,
            "n_obs": len(report.step_returns),
            "n_trials": int(max(1, n_trials)),
        }

    baselines = compare_to_baselines(
        report.step_returns,
        report.market_returns,
        report.step_turnover,
        report.step_exposure,
        total_cost=report.total_cost,
        capital0=report.equity_curve[0] if report.equity_curve else 1.0,
        block_size=block_size,
        n_bootstrap=n_bootstrap,
        random_trials=random_trials,
        seed=seed,
    )
    stress = EventStressTest().run(report)

    return {
        "deflated_sharpe": dsr,
        "baseline_comparison": baselines,
        "stress": stress,
        "trading_quality": _trading_quality(report),
        "confidence_buckets": _confidence_bucket_returns(report),
        "regime_returns": _regime_returns(report),
        "cost_sensitivity": _cost_sensitivity(report),
        "promotion_checks": _promotion_checks(report, baselines),
    }


def _trading_quality(report: BacktestReport) -> dict[str, float]:
    elapsed_days = 0.0
    if len(report.timestamps) >= 2:
        elapsed_days = max(
            (report.timestamps[-1] - report.timestamps[0]).total_seconds() / 86_400.0,
            1e-9,
        )
    return {
        "average_trade_return": float(np.mean(report.trade_returns)) if report.trade_returns else 0.0,
        "trade_frequency_per_day": float(report.n_trades / elapsed_days) if elapsed_days > 0 else 0.0,
        "cost_adjusted_return": float(report.total_return),
        "no_trade_fraction": float(report.no_trade_fraction),
    }


def _decision_confidence(decision) -> float:
    summary = getattr(decision, "market_summary", {}) or {}
    return float(max(summary.get("p_stop", 0.0), summary.get("p_target", 0.0)))


def _confidence_bucket_returns(report: BacktestReport, n_bins: int = 5) -> list[dict[str, float]]:
    if not report.decisions or not report.step_returns:
        return []
    conf = np.asarray([_decision_confidence(decision) for decision in report.decisions], dtype=float)
    returns = np.asarray(report.step_returns[: conf.size], dtype=float)
    traded = np.asarray(
        [decision.final_action.action_type is not ActionType.NO_TRADE for decision in report.decisions],
        dtype=bool,
    )
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[dict[str, float]] = []
    for lo, hi in zip(bins[:-1], bins[1:], strict=False):
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if not bool(mask.any()):
            continue
        trade_mask = mask & traded
        out.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "n": float(mask.sum()),
                "trades": float(trade_mask.sum()),
                "mean_confidence": float(conf[mask].mean()),
                "mean_step_return": float(returns[mask].mean()),
                "mean_trade_step_return": float(returns[trade_mask].mean()) if bool(trade_mask.any()) else 0.0,
                "hit_rate": float((returns[trade_mask] > 0.0).mean()) if bool(trade_mask.any()) else 0.0,
            }
        )
    return out


def _regime_returns(report: BacktestReport) -> dict[str, dict[str, float]]:
    if not report.decisions or not report.step_returns:
        return {}
    out: dict[str, dict[str, float]] = {}
    for decision, step_return in zip(report.decisions, report.step_returns, strict=False):
        regime = str(decision.latent_regime)
        bucket = out.setdefault(
            regime,
            {"n": 0.0, "trades": 0.0, "mean_step_return": 0.0, "hit_rate": 0.0},
        )
        n = bucket["n"]
        bucket["mean_step_return"] = (bucket["mean_step_return"] * n + float(step_return)) / (n + 1.0)
        bucket["hit_rate"] = (bucket["hit_rate"] * n + (1.0 if step_return > 0 else 0.0)) / (n + 1.0)
        bucket["n"] = n + 1.0
        if decision.final_action.action_type is not ActionType.NO_TRADE:
            bucket["trades"] += 1.0
    return out


def _cost_sensitivity(report: BacktestReport) -> dict[str, object]:
    capital0 = float(report.equity_curve[0]) if report.equity_curve else 1.0
    cost_return = float(report.total_cost / max(capital0, 1e-12))
    out: dict[str, object] = {}
    for multiplier in (0.0, 0.5, 1.0, 1.5, 2.0):
        out[f"total_return_cost_{multiplier:.1f}x"] = float(
            report.total_return + (1.0 - multiplier) * cost_return
        )

    step_returns = np.asarray(report.step_returns, dtype=float)
    turnover = np.asarray(report.step_turnover, dtype=float)
    step_cost_returns = np.asarray(report.step_cost_returns, dtype=float)
    if step_returns.size == 0:
        out.update(
            {
                "sharpe_at_5bps": 0.0,
                "sharpe_at_25bps": 0.0,
                "total_return_at_5bps": 0.0,
                "total_return_at_25bps": 0.0,
                "max_drawdown_at_5bps": 0.0,
                "max_drawdown_at_25bps": 0.0,
            }
        )
        return out
    if turnover.size != step_returns.size:
        turnover = np.zeros_like(step_returns)
    if step_cost_returns.size != step_returns.size:
        step_cost_returns = np.zeros_like(step_returns)

    gross_returns = step_returns + step_cost_returns
    periods = tm.infer_periods_per_year(report.timestamps)
    out["average_step_cost_return"] = float(step_cost_returns.mean())
    out["turnover_cost_basis"] = "observed_step_turnover"
    for bps in (5, 25):
        stressed_returns = gross_returns - (bps / 10_000.0) * turnover
        stressed_equity = [capital0]
        for step_return in stressed_returns:
            stressed_equity.append(stressed_equity[-1] * (1.0 + float(step_return)))
        out[f"sharpe_at_{bps}bps"] = float(tm.sharpe(stressed_returns, periods_per_year=periods))
        out[f"total_return_at_{bps}bps"] = float(stressed_equity[-1] / max(capital0, 1e-12) - 1.0)
        out[f"max_drawdown_at_{bps}bps"] = float(tm.max_drawdown(stressed_equity))
    return out


def _promotion_checks(report: BacktestReport, baselines: dict[str, object]) -> dict[str, str]:
    checks: dict[str, str] = {}
    checks["nonzero_trades"] = "PASS" if report.n_trades > 0 else "FAIL n_trades=0"
    checks["cost_adjusted_return"] = (
        "PASS" if report.total_return > 0.0 else f"FAIL total_return={report.total_return:.6f}"
    )
    checks["profit_factor"] = (
        "PASS" if report.profit_factor > 1.0 else f"FAIL profit_factor={report.profit_factor:.6f}"
    )
    checks["drawdown_control"] = (
        "PASS" if report.max_drawdown < 0.20 else f"FAIL max_drawdown={report.max_drawdown:.6f}"
    )
    buy_hold = baselines.get("buy_hold", {}) if isinstance(baselines, dict) else {}
    observed_diff = (
        float(buy_hold.get("observed_diff_mean", 0.0))
        if isinstance(buy_hold, dict)
        else 0.0
    )
    checks["buy_hold_edge"] = (
        "PASS" if observed_diff > 0.0 else f"FAIL observed_diff_mean={observed_diff:.8f}"
    )
    cost = _cost_sensitivity(report)
    sharpe_25bps = float(cost.get("sharpe_at_25bps", 0.0))
    checks["positive_25bps_sharpe"] = (
        "PASS" if sharpe_25bps > 0.0 else f"FAIL sharpe_at_25bps={sharpe_25bps:.6f}"
    )
    return checks


__all__ = ["evaluate_backtest_report"]
