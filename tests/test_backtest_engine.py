"""Backtest engine, walk-forward, leakage report, trading metrics (SPEC.md §23, §27, Day 7)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from helion_risk_world.backtesting.backtest_engine import BacktestEngine, BacktestStep
from helion_risk_world.backtesting.leakage_report import LeakageReport
from helion_risk_world.backtesting.transaction_costs import TransactionCosts
from helion_risk_world.config.execution_config import CostModelConfig
from helion_risk_world.data.leakage_checks import LeakageError
from helion_risk_world.evaluation import trading_metrics as tm
from helion_risk_world.planner.mpc_planner import MPCPlanner
from helion_risk_world.schemas.action_schema import (
    ActionType,
    CandidateAction,
    FinalDecision,
    RiskDecision,
    ScoredCandidate,
)
from helion_risk_world.schemas import PortfolioState, RiskProfile
from helion_risk_world.schemas.execution_schema import ExecutionState
from helion_risk_world.schemas.prediction_schema import (
    BarrierProbabilities,
    HorizonPrediction,
    ModelPrediction,
)

TS0 = datetime(2026, 6, 16, 9, 20)
RISK = RiskProfile(
    name="balanced", max_risk_per_trade=0.01, max_daily_loss=0.02, max_weekly_loss=0.05,
    max_drawdown=0.10, max_exposure=1.0, max_trades_per_day=100, consecutive_loss_cooldown=99,
    cvar_alpha=0.05, n_paths=256,
)


def _pred(ts: datetime, mean: float, sigma: float = 0.01) -> ModelPrediction:
    q = {
        0.1: mean - 2 * sigma, 0.25: mean - sigma, 0.5: mean,
        0.75: mean + sigma, 0.9: mean + 2 * sigma,
    }
    hp = HorizonPrediction(horizon_bars=12, return_quantiles=q, volatility=sigma)
    barrier = BarrierProbabilities(stop=0.3, target=0.4, timeout=0.3)
    return ModelPrediction(
        symbol="BANKNIFTY", ts=ts,
        horizon_preds=[hp],
        barrier=barrier,
        mae=2 * sigma,
        sigma_H=sigma,
        epistemic=0.0, aleatoric=sigma, ood_score=0.0,
    )


def _market(ts: datetime) -> ExecutionState:
    return ExecutionState(symbol="BANKNIFTY", ts=ts, available_at=ts,
                          bid=99.95, ask=100.05, spread=0.1)


def _account(cap: float = 500_000.0) -> PortfolioState:
    return PortfolioState(ts=TS0, capital0=cap, capital=cap, cash=cap, free_margin=cap)


def _steps(means: list[float], realized: list[float]) -> list[BacktestStep]:
    out = []
    for i, (m, r) in enumerate(zip(means, realized, strict=True)):
        ts = TS0 + timedelta(minutes=5 * i)
        out.append(BacktestStep(prediction=_pred(ts, m), market=_market(ts), realized_return=r))
    return out


def _engine() -> BacktestEngine:
    return BacktestEngine(MPCPlanner.default(), TransactionCosts(CostModelConfig()))


def test_walk_forward_constructs() -> None:
    from helion_risk_world.backtesting.walk_forward import WalkForward

    assert WalkForward(n_folds=5, embargo_bars=12)._n_folds == 5


def test_metric_helpers() -> None:
    assert tm.max_drawdown([100, 120, 90, 95]) == pytest.approx((120 - 90) / 120)
    assert tm.hit_rate([1.0, -1.0, 2.0, 0.0]) == pytest.approx(2 / 3)
    assert tm.profit_factor([2.0, -1.0, 1.0]) == pytest.approx(3.0)
    assert tm.expectancy([0.02, -0.01, 0.03]) == pytest.approx(0.0133333333)
    assert tm.turnover([0.1, 0.0, 0.25]) == pytest.approx(0.35)
    assert tm.sharpe([0.01, 0.01, 0.01]) == 0.0  # no dispersion
    annualized = tm.annualized_return(
        [100.0, 110.0],
        [TS0, TS0 + timedelta(days=365)],
    )
    assert annualized == pytest.approx(0.10, rel=1e-2)


def test_run_produces_audited_report() -> None:
    report = _engine().run(_steps([0.03, 0.03, 0.03], [0.02, -0.01, 0.015]), _account(), RISK)
    assert report.n_steps == 3 and len(report.decisions) == 3
    assert all(d.reason_code for d in report.decisions)
    assert all(d.paper_result is not None for d in report.decisions)
    assert report.gross_pnl == pytest.approx(report.net_pnl + report.total_cost, rel=1e-6)
    assert "annualized_return" in report.summary()
    assert len(report.step_returns) == report.n_steps
    assert len(report.equity_curve) == report.n_steps + 1
    # regime breakdown keys are latent_regime strings from the auditor
    assert isinstance(report.regime_breakdown, dict)


def test_run_is_reproducible() -> None:
    steps = _steps([0.02, 0.0, 0.03, -0.01], [0.01, -0.02, 0.02, -0.01])
    a = _engine().run(steps, _account(), RISK)
    b = _engine().run(steps, _account(), RISK)
    assert a.net_pnl == b.net_pnl and a.final_capital == b.final_capital
    assert a.sharpe == b.sharpe and a.n_trades == b.n_trades


def test_no_edge_run_does_not_trade() -> None:
    report = _engine().run(_steps([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]), _account(), RISK)
    assert report.n_trades == 0
    assert report.no_trade_fraction == 1.0
    assert report.total_cost == 0.0
    assert report.expectancy == 0.0
    assert report.turnover == 0.0


def _engine_low_lambda() -> BacktestEngine:
    """Planner with λ=0.5 so moderate edges produce trades (tests the engine machinery)."""
    from helion_risk_world.config.planner_config import PlannerConfig
    cfg = PlannerConfig(risk_aversion_lambda=0.5, cvar_alpha=0.05)
    return BacktestEngine(MPCPlanner.default(planner_cfg=cfg), TransactionCosts(CostModelConfig()))


class _FixedPlanner:
    def __init__(self, actions: list[CandidateAction]) -> None:
        self._actions = actions
        self._idx = 0

    def adapt_risk(self, risk: RiskProfile) -> RiskProfile:
        return risk

    def plan(self, prediction: ModelPrediction, state: PortfolioState, risk: RiskProfile, market=None) -> FinalDecision:
        action = self._actions[self._idx]
        self._idx += 1
        scored = ScoredCandidate(action=action, score=1.0, components={})
        shield = RiskDecision(
            allowed=True,
            reason_code="test",
            final_action=action,
            adjusted_size=action.size_fraction,
        )
        return FinalDecision(
            ts=prediction.ts,
            symbol=prediction.symbol,
            strategy_name="test",
            market_summary={"return_p50": prediction.longest_horizon.return_quantiles[0.5]},
            latent_regime="range",
            uncertainty=prediction.epistemic,
            ood_score=prediction.ood_score,
            portfolio_summary={"capital": state.capital, "drawdown": state.drawdown, "exposure": state.exposure},
            candidates=[scored],
            chosen_before_shield=action,
            execution_realism="high",
            risk_shield=shield,
            final_action=action,
            reason_code="test",
            expected_cost=0.0,
            expected_risk=0.0,
            expected_reward=0.0,
        )


def _pred_strong(ts: datetime) -> ModelPrediction:
    """Strong-edge prediction: high target probability, near-zero stop probability."""
    sigma = 0.01
    mean = 0.05
    q = {0.1: 0.02, 0.25: 0.035, 0.5: mean, 0.75: 0.065, 0.9: 0.08}
    hp = HorizonPrediction(horizon_bars=12, return_quantiles=q, volatility=sigma)
    barrier = BarrierProbabilities(stop=0.05, target=0.85, timeout=0.10)
    return ModelPrediction(
        symbol="BANKNIFTY", ts=ts,
        horizon_preds=[hp], barrier=barrier, mae=sigma, sigma_H=sigma,
        epistemic=0.0, aleatoric=sigma, ood_score=0.0,
    )


def test_costs_reduce_net_below_gross_when_trading() -> None:
    ts = [TS0 + timedelta(minutes=5 * i) for i in range(2)]
    steps = [
        BacktestStep(prediction=_pred_strong(t), market=_market(t), realized_return=0.03)
        for t in ts
    ]
    report = _engine_low_lambda().run(steps, _account(), RISK)
    assert report.n_trades > 0
    assert report.total_cost > 0.0
    assert report.turnover > 0.0
    assert report.net_pnl < report.gross_pnl


def test_trade_metrics_use_completed_trade_lifecycle_not_fill_steps() -> None:
    ts = [TS0 + timedelta(minutes=5 * i) for i in range(3)]
    planner = _FixedPlanner(
        [
            CandidateAction(action_type=ActionType.ENTER_LONG, size_fraction=1.0),
            CandidateAction(action_type=ActionType.NO_TRADE, size_fraction=0.0),
            CandidateAction(action_type=ActionType.EXIT, size_fraction=0.0),
        ]
    )
    steps = [
        BacktestStep(prediction=_pred_strong(ts[0]), market=_market(ts[0]), realized_return=0.001),
        BacktestStep(prediction=_pred_strong(ts[1]), market=_market(ts[1]), realized_return=0.03),
        BacktestStep(prediction=_pred(ts[2], mean=-0.02), market=_market(ts[2]), realized_return=0.0),
    ]

    report = BacktestEngine(planner, TransactionCosts(CostModelConfig())).run(steps, _account(), RISK)

    assert report.n_trades == 1
    assert len(report.trade_pnls) == 1
    assert report.trade_pnls[0] > 0.0
    assert report.trade_returns[0] > 0.0
    assert report.hit_rate == 1.0
    assert report.expectancy > 0.0


def test_backtest_uses_execution_market_for_costs() -> None:
    ts = [TS0 + timedelta(minutes=5 * i) for i in range(2)]
    low_spread_market = ExecutionState(
        symbol="BANKNIFTY_FUT_continuous",
        ts=ts[1],
        available_at=ts[1],
        bid=99.99,
        ask=100.01,
        spread=0.02,
    )
    high_spread_market = ExecutionState(
        symbol="BANKNIFTY_FUT_continuous",
        ts=ts[1],
        available_at=ts[1],
        bid=99.5,
        ask=100.5,
        spread=1.0,
    )
    decision_market = _market(ts[0])
    low = BacktestStep(
        prediction=_pred_strong(ts[0]),
        market=decision_market,
        execution_market=low_spread_market,
        realized_return=0.03,
    )
    high = BacktestStep(
        prediction=_pred_strong(ts[0]),
        market=decision_market,
        execution_market=high_spread_market,
        realized_return=0.03,
    )

    report_low = _engine_low_lambda().run([low], _account(), RISK)
    report_high = _engine_low_lambda().run([high], _account(), RISK)

    assert report_high.total_cost > report_low.total_cost


def test_overnight_hold_charges_financing_but_same_day_hold_does_not() -> None:
    """Feature/label overhaul Phase 4a: a position held ACROSS a session boundary
    accrues overnight NRML-carry financing; the identical hold within one session
    does not."""
    same_day_ts = [TS0, TS0 + timedelta(minutes=5)]
    next_day_open = datetime(TS0.year, TS0.month, TS0.day + 1, 9, 20)
    cross_day_ts = [TS0, next_day_open]

    def _run(ts_pair: list[datetime]) -> float:
        planner = _FixedPlanner(
            [
                CandidateAction(action_type=ActionType.ENTER_LONG, size_fraction=1.0),
                CandidateAction(action_type=ActionType.NO_TRADE, size_fraction=0.0),
            ]
        )
        steps = [
            BacktestStep(prediction=_pred_strong(ts_pair[0]), market=_market(ts_pair[0]), realized_return=0.0),
            BacktestStep(prediction=_pred_strong(ts_pair[1]), market=_market(ts_pair[1]), realized_return=0.0),
        ]
        report = BacktestEngine(planner, TransactionCosts(CostModelConfig())).run(steps, _account(), RISK)
        return report.total_cost

    same_day_cost = _run(same_day_ts)
    cross_day_cost = _run(cross_day_ts)
    assert cross_day_cost > same_day_cost


def test_backtest_fails_fast_when_account_cannot_open_one_contract() -> None:
    ts = TS0
    prediction = _pred_strong(ts).model_copy(update={"symbol": "BANKNIFTY_FUT_continuous"})
    market = ExecutionState(
        symbol="BANKNIFTY_FUT_continuous",
        ts=ts,
        available_at=ts,
        bid=49_990.0,
        ask=50_010.0,
        spread=20.0,
    )
    step = BacktestStep(
        prediction=prediction,
        market=market,
        execution_market=market,
        realized_return=0.01,
    )
    impossible_risk = RISK.model_copy(update={"max_exposure": 0.60})

    with pytest.raises(ValueError, match="cannot open one contract"):
        _engine().run([step], _account(), impossible_risk)


def test_leakage_report_passes_clean_and_raises_on_portfolio_leak() -> None:
    lr = LeakageReport()
    assert lr.run(["close", "atr", "pcr", "vix"])["passed"] is True
    with pytest.raises(LeakageError):
        lr.run(["close", "drawdown"])  # portfolio field in market features


def test_leakage_report_rejects_future_label() -> None:
    with pytest.raises(LeakageError):
        LeakageReport().run(["close"], labels=[(TS0, TS0 - timedelta(minutes=5))])


def test_backtest_script_style_leakage_check_catches_bad_label_realized_at() -> None:
    """Review finding H8: scripts/backtest.py used to call LeakageReport().run()
    with no feature_rows/labels, silently skipping the real per-row checks. This
    reproduces the exact feature_rows/labels construction backtest.py now uses
    from a list of BacktestSteps, confirming it actually catches a bad step."""
    good_steps = _steps([0.01, 0.02], [0.01, 0.02])
    good_steps = [
        BacktestStep(
            prediction=s.prediction, market=s.market, realized_return=s.realized_return,
            label_realized_at=s.prediction.ts + timedelta(minutes=15),
        )
        for s in good_steps
    ]
    feature_rows = [{"ts": s.market.ts, "available_at": s.market.available_at} for s in good_steps]
    labels = [(s.prediction.ts, s.label_realized_at) for s in good_steps]
    assert LeakageReport().run(["close"], feature_rows=feature_rows, labels=labels)["passed"] is True

    bad_steps = list(good_steps)
    bad_steps[0] = BacktestStep(
        prediction=bad_steps[0].prediction,
        market=bad_steps[0].market,
        realized_return=bad_steps[0].realized_return,
        label_realized_at=bad_steps[0].prediction.ts - timedelta(minutes=5),  # leaks into the past
    )
    bad_labels = [(s.prediction.ts, s.label_realized_at) for s in bad_steps]
    with pytest.raises(LeakageError):
        LeakageReport().run(["close"], feature_rows=feature_rows, labels=bad_labels)


# ---------------------------------------------------------------------------
# Per-bar settlement (multi-count bug fix, 2026-07-18)
# ---------------------------------------------------------------------------

def _pred_nospec(ts: datetime, mean: float = 0.0, sigma: float = 0.01) -> ModelPrediction:
    """Prediction for a symbol with NO instrument spec (continuous-notional fallback)."""
    q = {
        0.1: mean - 2 * sigma, 0.25: mean - sigma, 0.5: mean,
        0.75: mean + sigma, 0.9: mean + 2 * sigma,
    }
    hp = HorizonPrediction(horizon_bars=12, return_quantiles=q, volatility=sigma)
    return ModelPrediction(
        symbol="TESTSYM", ts=ts,
        horizon_preds=[hp],
        barrier=BarrierProbabilities(stop=0.3, target=0.4, timeout=0.3),
        mae=2 * sigma, sigma_H=sigma,
        epistemic=0.0, aleatoric=sigma, ood_score=0.0,
    )


def _market_nospec(ts: datetime) -> ExecutionState:
    return ExecutionState(symbol="TESTSYM", ts=ts, available_at=ts,
                          bid=99.95, ask=100.05, spread=0.1)


def test_held_position_settles_per_bar_marks_not_overlapping_h_bar_labels() -> None:
    """Regression for the settlement multi-count bug: the real-data step builders put
    the FULL H-bar forward label in realized_return and the engine settled
    notional x that label EVERY held bar — a K-bar hold accrued K overlapping H-bar
    returns. With carry_return/fill_to_mark_return populated, settlement must follow
    the per-bar price path and the (poisoned, absurd) H-bar label must never reach P&L."""
    ts = [TS0 + timedelta(minutes=5 * i) for i in range(4)]
    planner = _FixedPlanner(
        [
            CandidateAction(action_type=ActionType.ENTER_LONG, size_fraction=1.0),
            CandidateAction(action_type=ActionType.NO_TRADE, size_fraction=0.0),
            CandidateAction(action_type=ActionType.NO_TRADE, size_fraction=0.0),
            CandidateAction(action_type=ActionType.EXIT, size_fraction=0.0),
        ]
    )
    # Price path: entry fill 100.0 -> marks 101 / 102 / 103 -> exit fill 103.5 (~+3.5%).
    # realized_return is poisoned with an absurd +50% H-bar label on every step: under
    # the pre-fix settlement a 3-bar hold would realize ~150% of notional.
    poisoned_label = 0.5
    steps = [
        BacktestStep(
            prediction=_pred_nospec(ts[0]), market=_market_nospec(ts[0]),
            realized_return=poisoned_label,
            carry_return=0.0,                                  # flat before entry
            fill_to_mark_return=101.0 / 100.0 - 1.0,           # fill 100 -> mark 101
        ),
        BacktestStep(
            prediction=_pred_nospec(ts[1]), market=_market_nospec(ts[1]),
            realized_return=poisoned_label,
            carry_return=101.5 / 101.0 - 1.0,                  # mark 101 -> open 101.5
            fill_to_mark_return=102.0 / 101.5 - 1.0,           # open 101.5 -> mark 102
        ),
        BacktestStep(
            prediction=_pred_nospec(ts[2]), market=_market_nospec(ts[2]),
            realized_return=poisoned_label,
            carry_return=102.5 / 102.0 - 1.0,
            fill_to_mark_return=103.0 / 102.5 - 1.0,
        ),
        BacktestStep(
            prediction=_pred_nospec(ts[3]), market=_market_nospec(ts[3]),
            realized_return=poisoned_label,
            carry_return=103.5 / 103.0 - 1.0,                  # final mark -> exit fill
            fill_to_mark_return=0.0,                            # post-exit notional is 0
        ),
    ]
    cap = 500_000.0
    report = BacktestEngine(planner, TransactionCosts(CostModelConfig())).run(
        steps, _account(cap), RISK
    )
    assert report.n_trades == 1
    # True path P&L at full-capital notional is ~3.5% of capital (minus costs, plus
    # minor per-bar compounding). The poisoned label would produce ~150%.
    assert 0.020 * cap < report.net_pnl < 0.050 * cap
    # trade_pnls accumulate net_step over the trade's life, so the single trade's
    # P&L must equal the whole run's net P&L (cost-inclusive).
    assert report.trade_pnls[0] == pytest.approx(report.net_pnl, rel=1e-6)


def test_exit_realizes_final_carry_leg_on_old_notional() -> None:
    """An EXIT's post-fill notional is zero — the carried position must still realize
    its final mark->exit-fill move (pre-fix code realized exactly 0 on the exit bar)."""
    ts = [TS0 + timedelta(minutes=5 * i) for i in range(2)]
    planner = _FixedPlanner(
        [
            CandidateAction(action_type=ActionType.ENTER_LONG, size_fraction=1.0),
            CandidateAction(action_type=ActionType.EXIT, size_fraction=0.0),
        ]
    )
    steps = [
        BacktestStep(
            prediction=_pred_nospec(ts[0]), market=_market_nospec(ts[0]),
            realized_return=0.0, carry_return=0.0, fill_to_mark_return=0.01,
        ),
        BacktestStep(
            prediction=_pred_nospec(ts[1]), market=_market_nospec(ts[1]),
            realized_return=0.0, carry_return=0.02, fill_to_mark_return=0.0,
        ),
    ]
    cap = 500_000.0
    report = BacktestEngine(planner, TransactionCosts(CostModelConfig())).run(
        steps, _account(cap), RISK
    )
    # Entry bar: +1% on notional; exit bar: +2% carry on the OLD notional. Without the
    # carry leg the exit bar contributes only -cost and total lands near 1%.
    assert report.net_pnl > 0.025 * cap
