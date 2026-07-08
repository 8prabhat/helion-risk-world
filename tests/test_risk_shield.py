"""Risk Shield: deterministic, override-capable, de-risking always allowed (SPEC.md §20, §27)."""

from __future__ import annotations

from datetime import datetime

import pytest

from helion_risk_world.config.risk_config import RiskShieldConfig
from helion_risk_world.risk import DrawdownGuard, EventBlackout, ExposureManager, MarginSimulator
from helion_risk_world.risk.risk_shield import RiskShield
from helion_risk_world.schemas import ActionType, CandidateAction, PortfolioState, RiskProfile
from helion_risk_world.schemas.market_schema import EventContext, EventType
from helion_risk_world.schemas.prediction_schema import (
    BarrierProbabilities,
    HorizonPrediction,
    ModelPrediction,
)

TS = datetime(2026, 6, 16, 10, 0)
RISK = RiskProfile(
    name="balanced", max_risk_per_trade=0.01, max_daily_loss=0.02, max_weekly_loss=0.05,
    max_drawdown=0.10, max_exposure=1.0, max_trades_per_day=10, consecutive_loss_cooldown=4,
)
LONG = CandidateAction(action_type=ActionType.ENTER_LONG, size_fraction=0.5)
LARGE_LONG = CandidateAction(action_type=ActionType.ENTER_LONG, size_fraction=1.0)
LARGE_SHORT = CandidateAction(action_type=ActionType.ENTER_SHORT, size_fraction=1.0)
EXIT = CandidateAction(action_type=ActionType.EXIT, size_fraction=0.0)


def _pred(epistemic: float = 0.0, ood: float = 0.0) -> ModelPrediction:
    q = {0.1: -0.02, 0.25: -0.01, 0.5: 0.0, 0.75: 0.01, 0.9: 0.02}
    hp = HorizonPrediction(horizon_bars=12, return_quantiles=q, volatility=0.02)
    barrier = BarrierProbabilities(stop=0.33, target=0.34, timeout=0.33)
    return ModelPrediction(
        symbol="BANKNIFTY", ts=TS,
        horizon_preds=[hp], barrier=barrier, mae=0.04, sigma_H=0.02,
        epistemic=epistemic, aleatoric=0.1, ood_score=ood,
    )


def _state(**kw: object) -> PortfolioState:
    base = dict(ts=TS, capital0=500_000.0, capital=500_000.0, cash=500_000.0, free_margin=500_000.0)
    base.update(kw)
    return PortfolioState(**base)  # type: ignore[arg-type]


def test_no_trade_helper_blocks_and_returns_no_trade() -> None:
    d = RiskShield._no_trade("DAILY_LOSS_LIMIT")
    assert d.allowed is False and d.final_action.action_type is ActionType.NO_TRADE


def test_risk_config_rejects_out_of_range_thresholds() -> None:
    with pytest.raises(ValueError):
        RiskShieldConfig(daily_loss_limit=1.5)


def test_clean_state_allows_entry() -> None:
    d = RiskShield(RiskShieldConfig()).validate(LONG, _state(), RISK, _pred())
    assert d.allowed and d.final_action == LONG


def test_max_drawdown_forces_exit_even_for_entry() -> None:
    d = RiskShield(RiskShieldConfig()).validate(LONG, _state(drawdown=0.2), RISK, _pred())
    assert not d.allowed
    assert d.final_action.action_type is ActionType.EXIT
    assert d.reason_code == "MAX_DRAWDOWN_BREACHED"


def test_daily_loss_blocks_entry_but_allows_exit() -> None:
    shield = RiskShield(RiskShieldConfig(daily_loss_limit=0.02))
    breached = _state(daily_pnl=-20_000.0)  # -4% < -2% limit
    assert not shield.validate(LONG, breached, RISK, _pred()).allowed
    # De-risking is always permitted.
    assert shield.validate(EXIT, breached, RISK, _pred()).allowed


def test_uncertainty_and_ood_quarantine_entries() -> None:
    shield = RiskShield(RiskShieldConfig(uncertainty_block_threshold=0.8, ood_block_threshold=0.9))
    high_unc = shield.validate(LONG, _state(), RISK, _pred(epistemic=0.95))
    assert high_unc.reason_code == "UNCERTAINTY_TOO_HIGH"
    assert shield.validate(LONG, _state(), RISK, _pred(ood=0.95)).reason_code == "OOD_QUARANTINE"
    # But not exit.
    assert shield.validate(EXIT, _state(), RISK, _pred(ood=0.95)).allowed


def test_ml_cannot_override_shield() -> None:
    """Whatever the planner proposes, a breached account is overridden — the shield is final."""
    shield = RiskShield(RiskShieldConfig())
    for action_type in (ActionType.ENTER_LONG, ActionType.INCREASE):
        a = CandidateAction(action_type=action_type, size_fraction=1.0)
        d = shield.validate(a, _state(drawdown=0.5), RISK, _pred())
        assert not d.allowed and d.final_action.action_type is ActionType.EXIT


def test_max_risk_per_trade_resizes_entry() -> None:
    decision = RiskShield(RiskShieldConfig()).validate(LARGE_LONG, _state(), RISK, _pred())
    assert decision.allowed is False
    assert decision.reason_code == "MAX_RISK_PER_TRADE"
    assert decision.final_action.action_type is ActionType.ENTER_LONG
    assert decision.final_action.size_fraction == pytest.approx(0.5)
    assert decision.adjusted_size == pytest.approx(0.5)


def test_explicit_stop_return_drives_risk_per_trade_limit() -> None:
    pred = _pred().model_copy(update={"stop_return": -0.04, "target_return": 0.04})
    decision = RiskShield(RiskShieldConfig()).validate(LARGE_LONG, _state(), RISK, pred)
    assert decision.reason_code == "MAX_RISK_PER_TRADE"
    assert decision.final_action.size_fraction == pytest.approx(0.25)


def test_short_risk_per_trade_uses_upper_barrier_move_as_adverse_risk() -> None:
    pred = _pred().model_copy(update={"stop_return": -0.02, "target_return": 0.04})
    decision = RiskShield(RiskShieldConfig()).validate(LARGE_SHORT, _state(), RISK, pred)
    assert decision.reason_code == "MAX_RISK_PER_TRADE"
    assert decision.final_action.size_fraction == pytest.approx(0.25)


def test_event_blackout_blocks_entry_but_allows_exit() -> None:
    shield = RiskShield(RiskShieldConfig(blackout_block=True))
    event_ts = datetime(2024, 2, 8, 10, 0)  # RBI event day
    pred = _pred().model_copy(update={"ts": event_ts})
    assert shield.validate(LONG, _state(), RISK, pred).reason_code == "EVENT_BLACKOUT"
    assert shield.validate(EXIT, _state(), RISK, pred).allowed


def test_strategy_risk_overlay_is_enforced_by_shield() -> None:
    tighter = RISK.model_copy(
        update={
            "max_daily_loss": 0.01,
            "max_drawdown": 0.05,
            "max_exposure": 0.35,
            "max_trades_per_day": 3,
            "consecutive_loss_cooldown": 2,
            "max_risk_per_trade": 0.004,
        }
    )
    shield = RiskShield(RiskShieldConfig())
    assert shield.validate(LONG, _state(exposure=0.35), tighter, _pred()).reason_code == "EXPOSURE_LIMIT"
    assert shield.validate(LONG, _state(daily_pnl=-6_000.0), tighter, _pred()).reason_code == "DAILY_LOSS_LIMIT"
    assert shield.validate(LONG, _state(drawdown=0.05), tighter, _pred()).reason_code == "MAX_DRAWDOWN_BREACHED"
    assert shield.validate(LONG, _state(trades_today=3), tighter, _pred()).reason_code == "MAX_TRADES_PER_DAY"
    assert shield.validate(LONG, _state(consecutive_losses=2), tighter, _pred()).reason_code == "CONSECUTIVE_LOSS_COOLDOWN"
    resized = shield.validate(LARGE_LONG, _state(), tighter, _pred())
    assert resized.reason_code == "MAX_RISK_PER_TRADE"
    assert resized.final_action.size_fraction == pytest.approx(0.2)


def test_risk_helpers_cover_exposure_margin_drawdown_and_events() -> None:
    tighter = RISK.model_copy(update={"max_daily_loss": 0.01, "max_drawdown": 0.05, "max_exposure": 0.35})
    state = _state(drawdown=0.05, daily_pnl=-6_000.0, exposure=0.35, free_margin=90_000.0)
    assert DrawdownGuard().daily_loss_breached(state, tighter)
    assert DrawdownGuard().max_drawdown_breached(state, tighter)
    assert ExposureManager().exceeds_limit(state, tighter)
    assert MarginSimulator().free_margin_below_floor(state, tighter, floor=0.20)

    event = EventContext(
        symbol="BANKNIFTY",
        ts=TS,
        available_at=TS,
        expiry_flag=False,
        event_day_flag=True,
        blackout_active=False,
        event_type=EventType.RBI,
    )
    assert EventBlackout().is_active(event)
