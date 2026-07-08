"""Strategy comparison backtests."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from helion_risk_world.backtesting import BacktestStep
from helion_risk_world.backtesting.strategy_comparison import (
    StrategyBacktestCase,
    StrategyComparisonRunner,
)
from helion_risk_world.schemas import PortfolioState, RiskProfile
from helion_risk_world.schemas.execution_schema import ExecutionState
from helion_risk_world.schemas.prediction_schema import (
    BarrierProbabilities,
    HorizonPrediction,
    ModelPrediction,
)
from helion_risk_world.strategy import get_strategy_profile

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


def _pred(ts: datetime, horizon: int, mean: float, sigma: float) -> ModelPrediction:
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
        horizon_preds=[HorizonPrediction(horizon_bars=horizon, return_quantiles=q, volatility=sigma)],
        barrier=BarrierProbabilities(stop=0.05, target=0.85, timeout=0.10),
        mae=sigma,
        sigma_H=sigma,
        epistemic=0.0,
        aleatoric=sigma,
        ood_score=0.0,
    )


def _steps(horizon: int, mean: float, realized: float) -> list[BacktestStep]:
    return [
        BacktestStep(
            prediction=_pred(TS0 + timedelta(minutes=5 * i), horizon, mean, 0.005),
            market=_market(TS0 + timedelta(minutes=5 * i)),
            realized_return=realized,
        )
        for i in range(3)
    ]


def test_strategy_comparison_runner_produces_ranked_summary() -> None:
    cases = [
        StrategyBacktestCase(get_strategy_profile("scalping"), _steps(3, 0.08, 0.03)),
        StrategyBacktestCase(get_strategy_profile("medium_frequency"), _steps(192, 0.04, 0.02)),
    ]

    report = StrategyComparisonRunner.default().run(cases, _account(), RISK)

    assert len(report.results) == 2
    assert report.best() is not None
    rows = report.summary_rows()
    assert rows[0]["strategy"] == report.best().strategy_name
    assert all("expectancy" in row and "turnover" in row for row in rows)
    payload = report.to_dict()
    assert "diagnostics" in payload["strategies"][0]


def test_strategy_comparison_validates_horizon_coverage() -> None:
    bad_case = StrategyBacktestCase(get_strategy_profile("scalping"), _steps(6, 0.08, 0.03))

    with pytest.raises(ValueError, match="requires horizon 3"):
        StrategyComparisonRunner.default().run([bad_case], _account(), RISK)
