"""Diagnostics and stress-test coverage."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from helion_risk_world.backtesting import BacktestEngine, BacktestStep, EventStressTest, evaluate_backtest_report
from helion_risk_world.backtesting.transaction_costs import TransactionCosts
from helion_risk_world.config.execution_config import CostModelConfig
from helion_risk_world.evaluation.baselines import compare_to_baselines
from helion_risk_world.planner.mpc_planner import MPCPlanner
from helion_risk_world.schemas import PortfolioState, RiskProfile
from helion_risk_world.schemas.execution_schema import ExecutionState
from helion_risk_world.schemas.prediction_schema import (
    BarrierProbabilities,
    HorizonPrediction,
    ModelPrediction,
)
from helion_risk_world.data.expiry_calendar import monthly_expiry

TS0 = datetime(2026, 6, 16, 9, 20)
RISK = RiskProfile(
    name="balanced", max_risk_per_trade=0.01, max_daily_loss=0.02, max_weekly_loss=0.05,
    max_drawdown=0.10, max_exposure=1.0, max_trades_per_day=100, consecutive_loss_cooldown=99,
    cvar_alpha=0.05, n_paths=256,
)


def _account() -> PortfolioState:
    cap = 500_000.0
    return PortfolioState(ts=TS0, capital0=cap, capital=cap, cash=cap, free_margin=cap)


def _market(ts: datetime) -> ExecutionState:
    return ExecutionState(symbol="BANKNIFTY", ts=ts, available_at=ts, bid=99.995, ask=100.005, spread=0.01)


def _pred(ts: datetime, mean: float, sigma: float = 0.005) -> ModelPrediction:
    q = {
        0.1: mean - 2 * sigma,
        0.25: mean - sigma,
        0.5: mean,
        0.75: mean + sigma,
        0.9: mean + 2 * sigma,
    }
    return ModelPrediction(
        symbol="BANKNIFTY",
        ts=ts,
        horizon_preds=[HorizonPrediction(horizon_bars=12, return_quantiles=q, volatility=sigma)],
        barrier=BarrierProbabilities(stop=0.05, target=0.85, timeout=0.10),
        mae=sigma,
        sigma_H=sigma,
        epistemic=0.0,
        aleatoric=sigma,
        ood_score=0.0,
    )


def _steps(means: list[float], realized: list[float]) -> list[BacktestStep]:
    return [
        BacktestStep(
            prediction=_pred(TS0 + timedelta(minutes=5 * i), mean),
            market=_market(TS0 + timedelta(minutes=5 * i)),
            realized_return=ret,
        )
        for i, (mean, ret) in enumerate(zip(means, realized, strict=True))
    ]


def test_baseline_comparison_reports_flat_buy_hold_and_random() -> None:
    stats = compare_to_baselines(
        strategy_returns=[0.01, 0.012, 0.008, 0.011, 0.009, 0.013],
        market_returns=[0.006, 0.007, 0.004, 0.006, 0.005, 0.007],
        turnover_fractions=[0.2, 0.0, 0.1, 0.0, 0.15, 0.0],
        exposure_path=[0.2, 0.2, 0.3, 0.3, 0.45, 0.45],
        total_cost=250.0,
        capital0=500_000.0,
        block_size=2,
        n_bootstrap=200,
        random_trials=8,
        seed=11,
    )

    assert stats["flat"]["observed_diff_mean"] > 0
    assert stats["buy_hold"]["baseline_mean_return"] > 0
    assert stats["random_matched_turnover"]["trials"] == 8
    assert 0.0 <= stats["flat"]["p_value"] <= 1.0


def test_event_stress_test_separates_event_expiry_and_regular_days() -> None:
    expiry_day = monthly_expiry(2024, 2)
    timestamps = [
        datetime(2024, 2, 8, 10, 0),                       # RBI event day
        datetime.combine(expiry_day, datetime.min.time()).replace(hour=10),
        datetime(2024, 2, 14, 10, 0),                      # regular day
    ]
    out = EventStressTest().run_series(timestamps, [0.02, -0.03, 0.01])

    assert out["event_days"]["count"] == 1.0
    assert out["expiry_days"]["count"] == 1.0
    assert out["regular_days"]["count"] == 1.0
    assert "rbi" in out["by_event_type"]
    assert "expiry" in out["by_event_type"]


def test_evaluate_backtest_report_includes_dsr_baselines_and_stress() -> None:
    report = BacktestEngine(
        MPCPlanner.default(),
        TransactionCosts(CostModelConfig()),
    ).run(
        _steps(
            [0.05, 0.04, 0.06, 0.03, 0.05, 0.04],
            [0.02, 0.015, 0.025, 0.01, 0.02, 0.015],
        ),
        _account(),
        RISK,
    )
    diagnostics = evaluate_backtest_report(report, n_trials=3, n_bootstrap=100, random_trials=4)

    assert "deflated_sharpe" in diagnostics
    assert "baseline_comparison" in diagnostics
    assert "stress" in diagnostics
    assert diagnostics["deflated_sharpe"]["n_trials"] == 3
    assert diagnostics["baseline_comparison"]["random_matched_turnover"]["trials"] == 4
    assert "cost_sensitivity" in diagnostics
    assert "sharpe_at_5bps" in diagnostics["cost_sensitivity"]
    assert "sharpe_at_25bps" in diagnostics["cost_sensitivity"]
    assert "total_return_at_5bps" in diagnostics["cost_sensitivity"]
    assert "total_return_at_25bps" in diagnostics["cost_sensitivity"]
    assert "positive_25bps_sharpe" in diagnostics["promotion_checks"]
