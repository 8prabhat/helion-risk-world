"""Walk-forward evaluation tests."""

from __future__ import annotations

from datetime import datetime, timedelta

from helion_risk_world.backtesting import BacktestStep, WalkForward
from helion_risk_world.backtesting.strategy_comparison import StrategyBacktestCase
from helion_risk_world.backtesting.walk_forward_evaluation import WalkForwardStrategyRunner
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


def _steps(horizon: int, mean: float, realized: float, n: int = 36) -> list[BacktestStep]:
    return [
        BacktestStep(
            prediction=_pred(TS0 + timedelta(minutes=5 * i), horizon, mean, 0.005),
            market=_market(TS0 + timedelta(minutes=5 * i)),
            realized_return=realized,
        )
        for i in range(n)
    ]


def test_embargo_bars_property_exposes_configured_value() -> None:
    assert WalkForward(n_folds=3, embargo_bars=7).embargo_bars == 7


def test_walk_forward_indices_are_chronological_and_contiguous() -> None:
    folds = WalkForward(n_folds=3, embargo_bars=1).split_indices(
        30,
        test_size=4,
        val_size=2,
        min_train_size=8,
    )

    assert len(folds) == 3
    assert folds[0].train_end <= folds[0].val_start
    assert folds[0].val_end <= folds[0].test_start
    assert folds[1].test_start == folds[0].test_end
    assert folds[2].test_start == folds[1].test_end


def test_walk_forward_runner_produces_in_and_out_of_sample_results() -> None:
    walker = WalkForward(n_folds=2, embargo_bars=1)
    runner = WalkForwardStrategyRunner.default(walker)
    case = StrategyBacktestCase(get_strategy_profile("scalping"), _steps(3, 0.08, 0.03))

    comparison = runner.run([case], _account(), RISK)
    result = comparison.results[0]

    assert len(result.folds) == 2
    assert result.in_sample_report.n_steps > result.out_of_sample_report.n_steps
    assert result.out_of_sample_report.ending_state is not None
    assert result.out_of_sample_report.n_steps == sum(
        fold.out_of_sample.n_steps for fold in result.folds
    )
    assert comparison.best() is not None
    payload = comparison.to_dict()
    assert "diagnostics" in payload["strategies"][0]
