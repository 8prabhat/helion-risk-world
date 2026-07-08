"""MPC planner: mean–CVaR objective, management loop, audit record (SPEC.md §19, §29)."""

from __future__ import annotations

from datetime import datetime

from helion_risk_world.config.planner_config import PlannerConfig
from helion_risk_world.planner.mpc_planner import MPCPlanner
from helion_risk_world.planner.reward_scorer import RewardScorer
from helion_risk_world.schemas import (
    ActionType,
    CandidateAction,
    PortfolioState,
    RiskProfile,
)
from helion_risk_world.schemas.execution_schema import CostEstimate, ExecutionRealism
from helion_risk_world.schemas.execution_schema import ExecutionState
from helion_risk_world.schemas.portfolio_schema import Consequence, PositionSide
from helion_risk_world.schemas.prediction_schema import (
    BarrierProbabilities,
    HorizonPrediction,
    ModelPrediction,
)

TS = datetime(2026, 6, 16, 10, 0)
RISK = RiskProfile(
    name="balanced", max_risk_per_trade=0.01, max_daily_loss=0.02, max_weekly_loss=0.05,
    max_drawdown=0.10, max_exposure=1.0, max_trades_per_day=10, consecutive_loss_cooldown=4,
    cvar_alpha=0.05, n_paths=512,
)


def _pred(
    mean: float,
    sigma: float = 0.02,
    epistemic: float = 0.05,
    ood: float = 0.0,
    p_stop: float = 0.3,
    p_target: float = 0.4,
) -> ModelPrediction:
    q = {
        0.1: mean - 2 * sigma, 0.25: mean - sigma, 0.5: mean,
        0.75: mean + sigma, 0.9: mean + 2 * sigma,
    }
    hp = HorizonPrediction(horizon_bars=12, return_quantiles=q, volatility=sigma)
    barrier = BarrierProbabilities(stop=p_stop, target=p_target, timeout=1.0 - p_stop - p_target)
    return ModelPrediction(
        symbol="BANKNIFTY", ts=TS,
        horizon_preds=[hp], barrier=barrier, mae=2 * sigma, sigma_H=sigma,
        epistemic=epistemic, aleatoric=0.05, ood_score=ood,
    )


def _state(**kw: object) -> PortfolioState:
    base = dict(ts=TS, capital0=500_000.0, capital=500_000.0, cash=500_000.0, free_margin=500_000.0)
    base.update(kw)
    return PortfolioState(**base)  # type: ignore[arg-type]


def test_plan_returns_full_audit_record() -> None:
    decision = MPCPlanner.default().plan(_pred(0.01), _state(), RISK)
    assert decision.symbol == "BANKNIFTY"
    assert len(decision.candidates) >= 3  # NO_TRADE + entries
    assert any(c.action.action_type is ActionType.NO_TRADE for c in decision.candidates)
    assert decision.reason_code
    assert decision.final_action.action_type in set(ActionType)


def test_no_edge_prefers_no_trade() -> None:
    # Zero drift, real downside → NO_TRADE should win
    decision = MPCPlanner.default().plan(_pred(0.0, sigma=0.03, p_stop=0.5, p_target=0.2), _state(), RISK)
    assert decision.chosen_before_shield.action_type is ActionType.NO_TRADE


def test_strong_positive_edge_enters_long() -> None:
    # High target prob (85%), low stop prob (5%), low lambda → should enter long
    from helion_risk_world.config.planner_config import PlannerConfig
    low_lambda = PlannerConfig(risk_aversion_lambda=0.5)
    planner = MPCPlanner.default(planner_cfg=low_lambda)
    decision = planner.plan(
        _pred(0.05, sigma=0.01, epistemic=0.0, p_stop=0.05, p_target=0.85),
        _state(), RISK,
    )
    assert decision.chosen_before_shield.action_type is ActionType.ENTER_LONG


def test_risk_shield_overrides_planner_in_drawdown() -> None:
    decision = MPCPlanner.default().plan(
        _pred(0.02, sigma=0.01, epistemic=0.0),
        _state(position=PositionSide.LONG, exposure=0.5, drawdown=0.5),
        RISK,
    )
    assert decision.risk_shield.allowed is False
    assert decision.final_action.action_type is ActionType.EXIT


def test_planner_respects_event_blackout_for_entries() -> None:
    planner = MPCPlanner.default(planner_cfg=PlannerConfig(risk_aversion_lambda=0.5))
    pred = _pred(0.03, sigma=0.01, epistemic=0.0, p_stop=0.05, p_target=0.85).model_copy(
        update={"ts": datetime(2024, 2, 8, 10, 0)}
    )
    decision = planner.plan(pred, _state(), RISK)
    assert decision.chosen_before_shield.action_type is ActionType.ENTER_LONG
    assert decision.risk_shield.allowed is False
    assert decision.reason_code == "EVENT_BLACKOUT"
    assert decision.final_action.action_type is ActionType.NO_TRADE


def test_management_loop_exits_on_barrier_flip() -> None:
    planner = MPCPlanner.default()
    held = _state(position=PositionSide.LONG, exposure=0.5, margin_used=250_000.0)
    pred = _pred(0.01, sigma=0.02, epistemic=0.0, p_stop=0.7, p_target=0.1)
    decision = planner.plan(pred, held, RISK)
    assert decision.final_action.action_type is ActionType.EXIT


def test_no_trade_flat_scores_exactly_zero_baseline() -> None:
    scorer = RewardScorer(PlannerConfig())
    zero_cost = CostEstimate(total_cost=0.0, fill_prob=1.0, realism=ExecutionRealism.HIGH)
    cons = Consequence(exp_dW=0.0, cvar_dW=0.0, p_drawdown_breach=0.0, d_margin=0.0, d_exposure=0.0)
    score, _ = scorer.score(
        CandidateAction(action_type=ActionType.NO_TRADE, size_fraction=0.0),
        _pred(0.0), cons, zero_cost, RISK, _state(),
    )
    assert score == 0.0


def test_unexecutable_futures_sizes_are_flagged_explicitly() -> None:
    planner = MPCPlanner.default(
        planner_cfg=PlannerConfig(risk_aversion_lambda=0.5, sizes=(0.0, 0.1, 0.2, 0.5)),
    )
    market = ExecutionState(
        symbol="BANKNIFTY_FUT_continuous",
        ts=TS,
        available_at=TS,
        bid=53_300.0,
        ask=53_350.0,
        spread=50.0,
    )

    decision = planner.plan(
        _pred(0.05, sigma=0.01, epistemic=0.0, p_stop=0.05, p_target=0.85),
        _state(),
        RISK.model_copy(update={"max_exposure": 0.6}),
        market,
    )

    assert decision.chosen_before_shield.action_type is ActionType.NO_TRADE
    blocked = [
        cand for cand in decision.candidates
        if cand.action.action_type is not ActionType.NO_TRADE
    ]
    assert blocked
    assert all(cand.components.get("unexecutable") == 1.0 for cand in blocked)


def test_planner_tie_break_prefers_executable_blocked_trade_over_tiny_unexecutable_one() -> None:
    planner = MPCPlanner.default(
        planner_cfg=PlannerConfig(risk_aversion_lambda=0.5, sizes=(0.0, 0.1, 0.2, 0.35, 0.5)),
    )
    market = ExecutionState(
        symbol="BANKNIFTY_FUT_continuous",
        ts=TS,
        available_at=TS,
        bid=53_300.0,
        ask=53_350.0,
        spread=50.0,
    )
    state = _state(
        capital0=2_500_000.0,
        capital=2_500_000.0,
        cash=2_500_000.0,
        free_margin=2_500_000.0,
    )

    decision = planner.plan(
        _pred(0.0008, sigma=0.01, epistemic=0.0, p_stop=0.40, p_target=0.37),
        state,
        RISK.model_copy(update={"max_exposure": 0.6}),
        market,
    )

    trade_candidates = [
        cand for cand in decision.candidates
        if cand.action.action_type is not ActionType.NO_TRADE
    ]
    best_trade = max(trade_candidates, key=planner._candidate_rank)

    assert best_trade.action.size_fraction >= 0.2
    assert best_trade.components.get("unexecutable") is None
    assert best_trade.components.get("blocked") == 1.0
    assert "raw_utility" in best_trade.components
    assert any(
        key.startswith("blocked_") and key != "blocked"
        for key in best_trade.components
    )
