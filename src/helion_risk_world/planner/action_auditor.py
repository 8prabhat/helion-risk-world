"""Decision audit record construction (SPEC.md §29, Day 6).

Builds the ``FinalDecision`` record for every decision step. SRP: record construction only.
"""

from __future__ import annotations

from collections.abc import Sequence

from helion_risk_world.schemas.action_schema import (
    CandidateAction,
    FinalDecision,
    RiskDecision,
    ScoredCandidate,
)
from helion_risk_world.schemas.execution_schema import CostEstimate
from helion_risk_world.schemas.portfolio_schema import Consequence, PortfolioState
from helion_risk_world.schemas.prediction_schema import ModelPrediction


class ActionAuditor:
    """Build the FinalDecision audit record (SPEC.md §29)."""

    def record(
        self,
        risk_decision: RiskDecision,
        scored: Sequence[ScoredCandidate],
        *,
        prediction: ModelPrediction,
        state: PortfolioState,
        chosen: ScoredCandidate,
        final_action: CandidateAction,
        cost: CostEstimate,
        consequence: Consequence,
        strategy_name: str | None = None,
    ) -> FinalDecision:
        # Summary from longest-horizon prediction (the management-horizon H)
        hp = prediction.longest_horizon
        market_summary = {
            "return_p50": hp.return_quantiles.get(0.5, 0.0),
            "volatility": hp.volatility,
            "p_stop": prediction.barrier.stop,
            "p_target": prediction.barrier.target,
            "sigma_H": prediction.sigma_H,
            "stop_return": prediction.resolved_stop_return(),
            "target_return": prediction.resolved_target_return(),
        }

        # Regime from ModelPrediction (optional, state-derived)
        if prediction.regime_probs:
            latent_regime = max(prediction.regime_probs, key=lambda k: prediction.regime_probs[k]).value  # type: ignore[attr-defined]
        else:
            latent_regime = "unknown"

        rejected = [s for s in scored if s.action != chosen.action]
        return FinalDecision(
            ts=prediction.ts,
            symbol=prediction.symbol,
            strategy_name=strategy_name,
            market_summary=market_summary,
            latent_regime=latent_regime,
            uncertainty=prediction.epistemic,
            ood_score=prediction.ood_score,
            portfolio_summary={
                "capital": state.capital,
                "drawdown": state.drawdown,
                "exposure": state.exposure,
                "daily_pnl": state.daily_pnl,
            },
            candidates=list(scored),
            chosen_before_shield=chosen.action,
            execution_realism=cost.realism.value,
            risk_shield=risk_decision,
            final_action=final_action,
            reason_code=risk_decision.reason_code,
            expected_cost=cost.total_cost,
            expected_risk=consequence.cvar_dW,    # positive shortfall (SPEC.md §19)
            expected_reward=consequence.exp_dW,   # fraction-of-capital E[ΔW]
            rejected_actions=rejected,
        )
