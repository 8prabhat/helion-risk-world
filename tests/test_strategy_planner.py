"""Strategy profile tests."""

from __future__ import annotations

from datetime import datetime

from helion_risk_world.config.risk_profiles import load_account_risk_profile
from helion_risk_world.schemas import ActionType, PortfolioState, RiskProfile
from helion_risk_world.schemas.execution_schema import ExecutionState
from helion_risk_world.schemas.prediction_schema import (
    BarrierProbabilities,
    HorizonPrediction,
    ModelPrediction,
)
from helion_risk_world.strategy import StrategyPlanner, get_strategy_profile

TS = datetime(2026, 6, 16, 10, 0)
BASE_RISK = RiskProfile(
    name="balanced",
    max_risk_per_trade=0.01,
    max_daily_loss=0.02,
    max_weekly_loss=0.05,
    max_drawdown=0.10,
    max_exposure=1.0,
    max_trades_per_day=10,
    consecutive_loss_cooldown=4,
    cvar_alpha=0.05,
    n_paths=256,
)


def _account() -> PortfolioState:
    cap = 500_000.0
    return PortfolioState(ts=TS, capital0=cap, capital=cap, cash=cap, free_margin=cap)


def _market() -> ExecutionState:
    return ExecutionState(
        symbol="BANKNIFTY",
        ts=TS,
        available_at=TS,
        bid=99.995,
        ask=100.005,
        spread=0.01,
    )


def _hp(horizon_bars: int, mean: float, sigma: float) -> HorizonPrediction:
    return HorizonPrediction(
        horizon_bars=horizon_bars,
        return_quantiles={
            0.1: mean - 2 * sigma,
            0.25: mean - sigma,
            0.5: mean,
            0.75: mean + sigma,
            0.9: mean + 2 * sigma,
        },
        volatility=sigma,
    )


def _multi_horizon_prediction() -> ModelPrediction:
    return ModelPrediction(
        symbol="BANKNIFTY",
        ts=TS,
        horizon_preds=[
            _hp(3, 0.040, 0.006),
            _hp(6, -0.006, 0.010),
            _hp(12, -0.200, 0.005),
            _hp(192, -0.006, 0.010),
        ],
        barrier=BarrierProbabilities(stop=0.05, target=0.25, timeout=0.70),
        mae=0.02,
        sigma_H=0.02,
        epistemic=0.0,
        aleatoric=0.01,
        ood_score=0.0,
    )


def test_strategy_profiles_change_decision_horizon_behavior() -> None:
    prediction = _multi_horizon_prediction()
    market = _market()

    scalping = StrategyPlanner.default(get_strategy_profile("scalping")).plan(
        prediction, _account(), BASE_RISK, market
    )
    medium = StrategyPlanner.default(get_strategy_profile("medium_frequency")).plan(
        prediction, _account(), BASE_RISK, market
    )
    low = StrategyPlanner.default(get_strategy_profile("low_frequency")).plan(
        prediction, _account(), BASE_RISK, market
    )

    assert scalping.strategy_name == "scalping"
    assert medium.strategy_name == "medium_frequency"
    assert low.strategy_name == "low_frequency"
    assert scalping.final_action.action_type is ActionType.ENTER_LONG
    assert medium.final_action.action_type is ActionType.NO_TRADE
    assert low.final_action.action_type is ActionType.ENTER_SHORT
    assert scalping.market_summary["return_p50"] == 0.040
    assert medium.market_summary["return_p50"] == -0.006
    assert low.market_summary["return_p50"] == -0.200


def test_strategy_profiles_apply_distinct_risk_overlays() -> None:
    scalping = get_strategy_profile("scalping").apply_risk(BASE_RISK)
    medium = get_strategy_profile("medium_frequency").apply_risk(BASE_RISK)
    low = get_strategy_profile("low_frequency").apply_risk(BASE_RISK)

    assert scalping.max_exposure < medium.max_exposure
    assert low.max_exposure < BASE_RISK.max_exposure
    assert scalping.max_trades_per_day > low.max_trades_per_day
    assert low.cvar_alpha < medium.cvar_alpha


def test_strategy_planner_enforces_profile_risk_limits() -> None:
    planner = StrategyPlanner.default(get_strategy_profile("scalping"))
    risk = get_strategy_profile("scalping").apply_risk(BASE_RISK)
    state = _account().model_copy(update={"exposure": risk.max_exposure})
    decision = planner.plan(_multi_horizon_prediction(), state, BASE_RISK, _market())
    assert decision.final_action.action_type is ActionType.NO_TRADE
    assert decision.reason_code == "EXPOSURE_LIMIT"


def test_load_account_risk_profile_reads_yaml_registry() -> None:
    profile = load_account_risk_profile("balanced")

    assert profile.capital0 == 500_000.0
    assert profile.risk.name == "balanced"
    assert profile.risk.max_drawdown == 0.10


def test_load_banknifty_futures_account_profile_reads_yaml_registry() -> None:
    profile = load_account_risk_profile("banknifty_futures_conservative")

    assert profile.capital0 == 2_500_000.0
    assert profile.risk.name == "banknifty_futures_conservative"
    assert profile.risk.max_exposure == 1.0
