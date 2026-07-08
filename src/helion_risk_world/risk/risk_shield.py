"""Deterministic hard Risk Shield (SPEC.md §19, §27, Day 6).

Evaluates ordered ``RiskRuleProtocol`` rules; the FIRST breach wins and replaces the proposed action
with that rule's safe fallback (NO_TRADE / REDUCE / EXIT). The ML model can NEVER override this: the
shield runs *after* the planner and its decision is final. SRP: validate/override only.
"""

from __future__ import annotations

from collections.abc import Sequence

from helion_risk_world.config.risk_config import RiskShieldConfig
from helion_risk_world.risk.constraints import RiskRuleProtocol, default_rules
from helion_risk_world.schemas.action_schema import ActionType, CandidateAction, RiskDecision
from helion_risk_world.schemas.portfolio_schema import PortfolioState, RiskProfile
from helion_risk_world.schemas.prediction_schema import ModelPrediction


class RiskShield:
    """Hard, deterministic shield above the ML planner (SPEC.md §19)."""

    def __init__(
        self, cfg: RiskShieldConfig, rules: Sequence[RiskRuleProtocol] | None = None
    ) -> None:
        self._cfg = cfg
        self._rules = tuple(rules) if rules is not None else tuple(default_rules(cfg))

    def validate(
        self,
        action: CandidateAction,
        state: PortfolioState,
        risk: RiskProfile,
        prediction: ModelPrediction,
    ) -> RiskDecision:
        """Allow the action, or override it with the first breached rule's safe fallback."""
        for rule in self._rules:
            outcome = rule.check(action, state, risk, prediction)
            if not outcome.allowed:
                fallback = outcome.fallback_action or CandidateAction(
                    action_type=ActionType.NO_TRADE, size_fraction=0.0
                )
                return RiskDecision(
                    allowed=False,
                    reason_code=outcome.reason_code,
                    final_action=fallback,
                    adjusted_size=outcome.fallback_size,
                )
        return RiskDecision(
            allowed=True,
            reason_code="OK",
            final_action=action,
            adjusted_size=action.size_fraction,
        )

    @staticmethod
    def _no_trade(reason: str) -> RiskDecision:
        nt = CandidateAction(action_type=ActionType.NO_TRADE, size_fraction=0.0)
        return RiskDecision(allowed=False, reason_code=reason, final_action=nt, adjusted_size=0.0)
