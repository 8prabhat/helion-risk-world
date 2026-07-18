"""Portfolio World transition + consequence (SPEC.md §17, §27, Day 5).

CVaR convention: cvar_dW is a POSITIVE shortfall (expected worst-α loss, ≥ 0).
ΔW is analytic from the heads — no price-path simulation.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from helion_risk_world.data.leakage_checks import PORTFOLIO_FEATURE_NAMES
from helion_risk_world.data.portfolio_state_builder import PortfolioStateBuilder
from helion_risk_world.schemas import (
    ActionType,
    CandidateAction,
    PortfolioState,
    PositionSide,
    RiskProfile,
)
from helion_risk_world.schemas.prediction_schema import (
    BarrierProbabilities,
    HorizonPrediction,
    ModelPrediction,
)
from helion_risk_world.worlds.portfolio_world import PortfolioWorld

TS = datetime(2026, 6, 25, 10, 0)
RISK = RiskProfile(
    name="balanced", max_risk_per_trade=0.01, max_daily_loss=0.02, max_weekly_loss=0.05,
    max_drawdown=0.10, max_exposure=1.0, max_trades_per_day=10, consecutive_loss_cooldown=4,
    cvar_alpha=0.05, n_paths=512,
)


def _prediction(mean: float, sigma: float, p_stop: float = 0.3, p_target: float = 0.4) -> ModelPrediction:
    q = {
        0.1: mean - 2 * sigma, 0.25: mean - sigma, 0.5: mean,
        0.75: mean + sigma, 0.9: mean + 2 * sigma,
    }
    hp = HorizonPrediction(horizon_bars=12, return_quantiles=q, volatility=sigma)
    barrier = BarrierProbabilities(stop=p_stop, target=p_target, timeout=1.0 - p_stop - p_target)
    return ModelPrediction(
        symbol="BANKNIFTY", ts=TS,
        horizon_preds=[hp], barrier=barrier, mae=2 * sigma, sigma_H=sigma,
        epistemic=0.1, aleatoric=0.1, ood_score=0.0,
    )


def _fresh(capital: float = 500_000.0) -> PortfolioState:
    return PortfolioState(
        ts=TS, capital0=capital, capital=capital, cash=capital, free_margin=capital
    )


def test_portfolio_state_is_portfolio_plane_only() -> None:
    assert "capital" in PORTFOLIO_FEATURE_NAMES
    assert _fresh().consecutive_losses == 0


def test_no_trade_flat_has_zero_consequence() -> None:
    pw = PortfolioWorld()
    nxt, cons = pw.step(
        _fresh(), CandidateAction(action_type=ActionType.NO_TRADE, size_fraction=0.0),
        _prediction(0.01, 0.02), RISK,
    )
    assert cons.exp_dW == 0.0
    assert cons.p_drawdown_breach == 0.0
    assert nxt.position is PositionSide.FLAT


def test_long_profits_on_positive_drift_short_loses() -> None:
    pw = PortfolioWorld()
    # Strong upward prediction: high target prob, low stop prob
    pred_up = _prediction(0.03, 0.01, p_stop=0.05, p_target=0.85)
    long = CandidateAction(action_type=ActionType.ENTER_LONG, size_fraction=0.5)
    short = CandidateAction(action_type=ActionType.ENTER_SHORT, size_fraction=0.5)
    _, long_cons = pw.step(_fresh(), long, pred_up, RISK)
    _, short_cons = pw.step(_fresh(), short, pred_up, RISK)
    assert long_cons.exp_dW > 0
    assert short_cons.exp_dW < 0


def test_cvar_dW_is_positive_shortfall() -> None:
    """CVaR is the expected worst-α loss reported as a POSITIVE number (SPEC.md §17, §19)."""
    pw = PortfolioWorld()
    _, cons = pw.step(
        _fresh(), CandidateAction(action_type=ActionType.ENTER_LONG, size_fraction=1.0),
        _prediction(0.0, 0.03), RISK,
    )
    assert cons.cvar_dW >= 0.0           # always non-negative (positive shortfall)
    assert cons.cvar_dW >= cons.exp_dW   # shortfall >= mean (by definition of left tail)


def test_step_is_deterministic_for_same_seed_and_common_noise() -> None:
    """Candidate comparisons should be reproducible under common random numbers."""
    pw = PortfolioWorld(n_samples=500, seed=7)
    action = CandidateAction(action_type=ActionType.ENTER_LONG, size_fraction=0.4)
    pred = _prediction(0.005, 0.02)
    noise = pw.sample_noise()
    a = pw.step(_fresh(), action, pred, RISK, common_noise=noise)[1]
    b = pw.step(_fresh(), action, pred, RISK, common_noise=noise)[1]
    assert a.exp_dW == b.exp_dW
    assert a.cvar_dW == b.cvar_dW


def test_same_future_different_accounts_imply_different_breach_risk() -> None:
    """Counterfactual accounts: in-drawdown account breaches more often than the fresh one."""
    pw = PortfolioWorld(n_samples=2000)
    built = PortfolioStateBuilder().synthetic_profiles(500_000.0, RISK, TS)
    profiles = {p.label: p.state for p in built}
    action = CandidateAction(action_type=ActionType.ENTER_LONG, size_fraction=1.0)
    pred = _prediction(0.0, 0.04)
    _, fresh_cons = pw.step(profiles["fresh"], action, pred, RISK)
    _, dd_cons = pw.step(profiles["in_drawdown"], action, pred, RISK)
    assert dd_cons.p_drawdown_breach >= fresh_cons.p_drawdown_breach


def test_exit_goes_flat() -> None:
    pw = PortfolioWorld()
    held = _fresh().model_copy(update={"position": PositionSide.LONG, "exposure": 0.5,
                                       "margin_used": 250_000.0})
    nxt, _ = pw.step(held, CandidateAction(action_type=ActionType.EXIT, size_fraction=0.0),
                     _prediction(0.0, 0.02), RISK)
    assert nxt.position is PositionSide.FLAT and nxt.exposure == 0.0


def test_explicit_barrier_returns_override_sigma_fallback() -> None:
    pw = PortfolioWorld()
    base = _prediction(0.0, 0.02, p_stop=0.5, p_target=0.5)
    explicit = base.model_copy(update={"stop_return": -0.01, "target_return": 0.03})
    action = CandidateAction(action_type=ActionType.ENTER_LONG, size_fraction=1.0)

    _, fallback_cons = pw.step(_fresh(), action, base, RISK)
    _, explicit_cons = pw.step(_fresh(), action, explicit, RISK)

    assert explicit_cons.exp_dW > fallback_cons.exp_dW


def test_short_side_swaps_barrier_semantics_and_underlying_barrier_returns() -> None:
    pred = _prediction(0.0, 0.02, p_stop=0.2, p_target=0.6).model_copy(
        update={"stop_return": -0.01, "target_return": 0.03}
    )

    short_barrier = pred.barrier_for_side("short")

    assert short_barrier.stop == pytest.approx(0.6)
    assert short_barrier.target == pytest.approx(0.2)
    assert pred.resolved_stop_return_for_side("short") == pytest.approx(0.03)
    assert pred.resolved_target_return_for_side("short") == pytest.approx(-0.01)


def test_quantile_stop_target_mode_produces_different_consequence_than_barrier_context() -> None:
    """2026-07-16: stop_target_mode="quantile" must actually change scoring behavior
    (proves the wiring reaches PortfolioWorld, not just that it doesn't crash) --
    skewed quantiles should shift exp_dW/cvar_dW relative to the fixed symmetric
    BarrierContext-derived sizing used by the default mode."""
    base = _prediction(0.0, 0.02, p_stop=0.4, p_target=0.4)
    # Explicit symmetric BarrierContext (what "barrier_context" mode will use) alongside
    # asymmetric quantiles (what "quantile" mode will use instead) on the SAME prediction.
    skewed_quantiles = {0.1: -0.003, 0.25: -0.001, 0.5: 0.001, 0.75: 0.006, 0.9: 0.01}
    pred = base.model_copy(
        update={
            "stop_return": -0.01, "target_return": 0.01,
            "horizon_preds": [base.horizon_preds[0].model_copy(update={"return_quantiles": skewed_quantiles})],
        }
    )
    action = CandidateAction(action_type=ActionType.ENTER_LONG, size_fraction=1.0)

    pw_barrier = PortfolioWorld(n_samples=2000, seed=7)
    pw_quantile = PortfolioWorld(n_samples=2000, seed=7, stop_target_mode="quantile")

    _, cons_barrier = pw_barrier.step(_fresh(), action, pred, RISK)
    _, cons_quantile = pw_quantile.step(_fresh(), action, pred, RISK)

    assert cons_barrier.exp_dW != cons_quantile.exp_dW
    assert cons_barrier.cvar_dW != cons_quantile.cvar_dW


def test_quantile_mode_rejects_unsupported_value() -> None:
    with pytest.raises(ValueError):
        PortfolioWorld(stop_target_mode="bogus")
