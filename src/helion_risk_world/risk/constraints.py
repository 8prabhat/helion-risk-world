"""Risk rules (SPEC.md §19, §26 OCP, Day 6).

Each rule is a small, deterministic, independently-testable unit implementing ``RiskRuleProtocol``.
New rules are pluggable without touching the shield (OCP). De-risking actions (NO_TRADE / EXIT /
REDUCE) are never blocked — guards only veto *risk-increasing* actions (ENTER_*/INCREASE). The
exception is the max-drawdown guard, which forces a flatten regardless.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Protocol, runtime_checkable

from helion_risk_world.config.risk_config import RiskShieldConfig
from helion_risk_world.data.event_calendar import event_type_for, is_event_day
from quanthelion.calendars.expiry_calendar import monthly_expiry
from helion_risk_world.risk.drawdown_guard import DrawdownGuard
from helion_risk_world.risk.event_blackout import EventBlackout
from helion_risk_world.risk.exposure_manager import ExposureManager
from helion_risk_world.risk.margin_simulator import MarginSimulator
from helion_risk_world.schemas.action_schema import ActionType, CandidateAction
from helion_risk_world.schemas.market_schema import EventContext, EventType
from helion_risk_world.schemas.portfolio_schema import PortfolioState, RiskProfile
from helion_risk_world.schemas.prediction_schema import ModelPrediction

_RISK_INCREASING = {ActionType.ENTER_LONG, ActionType.ENTER_SHORT, ActionType.INCREASE}
_EPS = 1e-9


def _is_unsafe_reading(value: float) -> bool:
    """True for NaN/Inf uncertainty readings (review finding H11).

    A non-finite epistemic/OOD score never satisfies a ``> threshold`` comparison,
    so a broken uncertainty estimate would otherwise silently pass every safety
    gate instead of blocking. Treat non-finite as maximally unsafe.
    """
    return not math.isfinite(value)


class RuleOutcome:
    """Result of a risk rule check."""

    def __init__(
        self,
        allowed: bool,
        reason_code: str = "OK",
        fallback_action: CandidateAction | None = None,
        fallback_size: float = 0.0,
    ) -> None:
        self.allowed = allowed
        self.reason_code = reason_code
        self.fallback_action = fallback_action
        self.fallback_size = fallback_size


@runtime_checkable
class RiskRuleProtocol(Protocol):
    """A single deterministic risk rule (SPEC.md §19, §26 OCP — rules are pluggable)."""

    reason_code: str

    def check(
        self,
        action: CandidateAction,
        state: PortfolioState,
        risk: RiskProfile,
        prediction: ModelPrediction,
    ) -> RuleOutcome: ...


def _ok() -> RuleOutcome:
    return RuleOutcome(allowed=True)


def _no_trade(reason: str) -> RuleOutcome:
    return RuleOutcome(
        allowed=False,
        reason_code=reason,
        fallback_action=CandidateAction(action_type=ActionType.NO_TRADE, size_fraction=0.0),
        fallback_size=0.0,
    )


def _exit(reason: str) -> RuleOutcome:
    return RuleOutcome(
        allowed=False,
        reason_code=reason,
        fallback_action=CandidateAction(action_type=ActionType.EXIT, size_fraction=0.0),
        fallback_size=0.0,
    )


def _is_risk_increasing(action: CandidateAction) -> bool:
    return action.action_type in _RISK_INCREASING


class MaxDrawdownRule:
    """Hard stop: at/over max drawdown -> force EXIT regardless of the proposed action."""

    reason_code = "MAX_DRAWDOWN_BREACHED"

    def __init__(self, cfg: RiskShieldConfig) -> None:
        self._cfg = cfg
        self._guard = DrawdownGuard()

    def check(self, action: CandidateAction, state: PortfolioState, risk: RiskProfile,
              prediction: ModelPrediction) -> RuleOutcome:
        effective_risk = risk.model_copy(
            update={"max_drawdown": min(risk.max_drawdown, self._cfg.max_drawdown)}
        )
        if self._guard.max_drawdown_breached(state, effective_risk):
            return _exit(self.reason_code)
        return _ok()


class DailyLossRule:
    """Block new risk once the daily loss limit is breached."""

    reason_code = "DAILY_LOSS_LIMIT"

    def __init__(self, cfg: RiskShieldConfig) -> None:
        self._cfg = cfg
        self._guard = DrawdownGuard()

    def check(self, action: CandidateAction, state: PortfolioState, risk: RiskProfile,
              prediction: ModelPrediction) -> RuleOutcome:
        effective_risk = risk.model_copy(
            update={"max_daily_loss": min(risk.max_daily_loss, self._cfg.daily_loss_limit)}
        )
        if _is_risk_increasing(action) and self._guard.daily_loss_breached(state, effective_risk):
            return _no_trade(self.reason_code)
        return _ok()


class FreeMarginRule:
    """Block new risk when free margin is below the floor."""

    reason_code = "FREE_MARGIN_FLOOR"

    def __init__(self, cfg: RiskShieldConfig) -> None:
        self._cfg = cfg
        self._margin = MarginSimulator()

    def check(self, action: CandidateAction, state: PortfolioState, risk: RiskProfile,
              prediction: ModelPrediction) -> RuleOutcome:
        if _is_risk_increasing(action) and self._margin.free_margin_below_floor(
            state, risk, self._cfg.free_margin_floor
        ):
            return _no_trade(self.reason_code)
        return _ok()


class ExposureRule:
    """Block new risk when gross exposure is already at/over the limit."""

    reason_code = "EXPOSURE_LIMIT"

    def __init__(self, cfg: RiskShieldConfig) -> None:
        self._cfg = cfg
        self._exposure = ExposureManager()

    def check(self, action: CandidateAction, state: PortfolioState, risk: RiskProfile,
              prediction: ModelPrediction) -> RuleOutcome:
        effective_risk = risk.model_copy(
            update={"max_exposure": min(risk.max_exposure, self._cfg.max_exposure)}
        )
        if _is_risk_increasing(action) and self._exposure.exceeds_limit(state, effective_risk):
            return _no_trade(self.reason_code)
        return _ok()


class MaxTradesRule:
    """Block new risk once the per-day trade count is reached."""

    reason_code = "MAX_TRADES_PER_DAY"

    def __init__(self, cfg: RiskShieldConfig) -> None:
        self._cfg = cfg

    def check(self, action: CandidateAction, state: PortfolioState, risk: RiskProfile,
              prediction: ModelPrediction) -> RuleOutcome:
        max_trades = min(risk.max_trades_per_day, self._cfg.max_trades_per_day)
        if _is_risk_increasing(action) and state.trades_today >= max_trades:
            return _no_trade(self.reason_code)
        return _ok()


class ConsecutiveLossRule:
    """Cool-down: block new risk after a run of consecutive losses."""

    reason_code = "CONSECUTIVE_LOSS_COOLDOWN"

    def __init__(self, cfg: RiskShieldConfig) -> None:
        self._cfg = cfg

    def check(self, action: CandidateAction, state: PortfolioState, risk: RiskProfile,
              prediction: ModelPrediction) -> RuleOutcome:
        cooled = state.consecutive_losses >= min(
            risk.consecutive_loss_cooldown,
            self._cfg.consecutive_loss_cooldown,
        )
        if _is_risk_increasing(action) and cooled:
            return _no_trade(self.reason_code)
        return _ok()


class UncertaintyRule:
    """Block new risk when epistemic uncertainty is too high.

    NOTE (review finding H9): this gate is inert whenever
    ``prediction.epistemic_calibrated`` is False — ``ForecasterPredictor`` (the
    default, non-world-model predictor) always emits ``epistemic=0.0`` with no
    real ensemble behind it, so this rule can never block a trade on that path.
    Use ``model_kind='world_model'`` for this gate to be meaningful.
    """

    reason_code = "UNCERTAINTY_TOO_HIGH"

    def __init__(self, cfg: RiskShieldConfig) -> None:
        self._cfg = cfg

    def check(self, action: CandidateAction, state: PortfolioState, risk: RiskProfile,
              prediction: ModelPrediction) -> RuleOutcome:
        uncertain = (
            _is_unsafe_reading(prediction.epistemic)
            or prediction.epistemic > self._cfg.uncertainty_block_threshold
        )
        if _is_risk_increasing(action) and uncertain:
            return _no_trade(self.reason_code)
        return _ok()


class OODRule:
    """Block new risk when the market state is out-of-distribution."""

    reason_code = "OOD_QUARANTINE"

    def __init__(self, cfg: RiskShieldConfig) -> None:
        self._cfg = cfg

    def check(self, action: CandidateAction, state: PortfolioState, risk: RiskProfile,
              prediction: ModelPrediction) -> RuleOutcome:
        unsafe = (
            _is_unsafe_reading(prediction.ood_score)
            or prediction.ood_score > self._cfg.ood_block_threshold
        )
        if _is_risk_increasing(action) and unsafe:
            return _no_trade(self.reason_code)
        return _ok()


class MaxRiskPerTradeRule:
    """Cap risk-increasing action size so predicted volatility risk stays within account limits."""

    reason_code = "MAX_RISK_PER_TRADE"

    def check(
        self,
        action: CandidateAction,
        state: PortfolioState,
        risk: RiskProfile,
        prediction: ModelPrediction,
    ) -> RuleOutcome:
        if not _is_risk_increasing(action):
            return _ok()
        side = "short" if action.action_type is ActionType.ENTER_SHORT else "long"
        per_unit_risk = max(
            abs(float(prediction.resolved_stop_return_for_side(side))),
            _EPS,
        )
        max_size = min(1.0, max(0.0, float(risk.max_risk_per_trade) / per_unit_risk))
        if action.size_fraction <= max_size + _EPS:
            return _ok()
        if max_size <= _EPS:
            return _no_trade(self.reason_code)
        adjusted = CandidateAction(action_type=action.action_type, size_fraction=max_size)
        return RuleOutcome(
            allowed=False,
            reason_code=self.reason_code,
            fallback_action=adjusted,
            fallback_size=max_size,
        )


class EventBlackoutRule:
    """Block new risk on configured event/expiry blackout bars."""

    reason_code = "EVENT_BLACKOUT"

    def __init__(self, cfg: RiskShieldConfig) -> None:
        self._cfg = cfg
        self._blackout = EventBlackout()

    def check(
        self,
        action: CandidateAction,
        state: PortfolioState,
        risk: RiskProfile,
        prediction: ModelPrediction,
    ) -> RuleOutcome:
        if not self._cfg.blackout_block or not _is_risk_increasing(action):
            return _ok()
        ts = prediction.ts
        event = _event_context(prediction.symbol, ts)
        if self._blackout.is_active(event):
            return _no_trade(self.reason_code)
        return _ok()


def _event_context(symbol: str, ts: datetime) -> EventContext:
    trade_date = ts.date()
    expiry = monthly_expiry(trade_date.year, trade_date.month)
    event_type = event_type_for(trade_date)
    if event_type is EventType.NONE and trade_date == expiry:
        event_type = EventType.EXPIRY
    return EventContext(
        symbol=symbol,
        ts=ts,
        available_at=ts,
        expiry_flag=trade_date == expiry,
        event_day_flag=is_event_day(trade_date),
        blackout_active=False,
        event_type=event_type,
    )


def default_rules(cfg: RiskShieldConfig) -> list[RiskRuleProtocol]:
    """The default ordered rule set. Max-drawdown first (it forces a flatten)."""
    return [
        MaxDrawdownRule(cfg),
        DailyLossRule(cfg),
        FreeMarginRule(cfg),
        ExposureRule(cfg),
        MaxTradesRule(cfg),
        ConsecutiveLossRule(cfg),
        UncertaintyRule(cfg),
        OODRule(cfg),
        EventBlackoutRule(cfg),
        MaxRiskPerTradeRule(),
    ]
