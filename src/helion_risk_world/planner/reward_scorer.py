"""Mean–CVaR planner utility U(a) (SPEC.md §19, Appendix A).

  U(a) = E[ΔW(a)] − λ · CVaR_α[ΔW(a)] − Cost(a)

where CVaR_α[X] = −E[X | X ≤ q_α(X)] ≥ 0  (POSITIVE shortfall, SPEC.md §19 sign convention).

One interpretable risk-aversion parameter λ, not eight magic weights that require
ad-hoc rescaling.  NO_TRADE baseline = 0; a trade is taken only if U(a*) > 0 so
risk-adjusted edge beats cost.

SRP: scoring only — action enumeration and portfolio simulation live elsewhere.
"""

from __future__ import annotations

from helion_risk_world.config.risk_config import RiskShieldConfig
from helion_risk_world.config.planner_config import PlannerConfig
from helion_risk_world.schemas.action_schema import ActionType, CandidateAction
from helion_risk_world.schemas.execution_schema import ExecutionRealism
from helion_risk_world.schemas.execution_schema import CostEstimate
from helion_risk_world.schemas.portfolio_schema import Consequence, PortfolioState, RiskProfile
from helion_risk_world.schemas.prediction_schema import ModelPrediction

_EPS = 1e-9


class RewardScorer:
    """Compute U(a) = E[ΔW] − λ · CVaR_α[ΔW] − Cost(a).  NO_TRADE baseline = 0."""

    def __init__(self, cfg: PlannerConfig, risk_cfg: RiskShieldConfig | None = None) -> None:
        self._lam = cfg.risk_aversion_lambda
        self._alpha = cfg.cvar_alpha
        self._risk_cfg = risk_cfg or RiskShieldConfig()

    def score(
        self,
        action: CandidateAction,
        prediction: ModelPrediction,
        consequence: Consequence,
        cost: CostEstimate,
        risk: RiskProfile,
        state: PortfolioState,
    ) -> tuple[float, dict[str, float]]:
        """Return (U, component_dict) for audit logging.

        All values are in fraction-of-capital units (matching Consequence.exp_dW).
        CVaR convention: consequence.cvar_dW is already a POSITIVE shortfall ≥ 0.
        """
        is_trade = action.action_type is not ActionType.NO_TRADE

        if not is_trade:
            # NO_TRADE is the reference action: U = 0
            return 0.0, {"exp_dW": 0.0, "cvar_term": 0.0, "cost_term": 0.0, "utility": 0.0}

        cap = max(state.capital, _EPS)
        exp_dW = consequence.exp_dW                           # E[ΔW] / capital
        cvar_term = self._lam * consequence.cvar_dW           # λ · CVaR_α (positive shortfall)
        cost_total = cost.total_cost / cap
        filled_edge = exp_dW * cost.fill_prob
        raw_utility = filled_edge - cvar_term - cost_total
        utility = raw_utility

        blocked_reason: str | None = None
        if cost.realism is ExecutionRealism.LOW:
            blocked_reason = "execution_realism_low"
        elif self._risk_cfg.require_edge_over_cost and filled_edge <= cost_total:
            blocked_reason = "edge_below_cost"
        elif filled_edge > 0:
            slippage_burden = (cost.slippage / cap) / max(filled_edge, _EPS)
            if slippage_burden >= self._risk_cfg.slippage_block_threshold:
                blocked_reason = "slippage_burden"

        if blocked_reason is not None:
            utility = -1e9

        components = {
            "exp_dW": filled_edge,
            "cvar_term": -cvar_term,    # negative (penalty)
            "cost_term": -cost_total,   # negative (penalty)
            "raw_utility": raw_utility,
            "utility": utility,
        }
        if blocked_reason is not None:
            components["blocked"] = 1.0
            components[f"blocked_{blocked_reason}"] = 1.0
        return float(utility), components
