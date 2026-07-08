"""Strategy-aware planner wrapper.

This keeps strategy selection out of the backtest/paper engines while still reusing the
shared MPC planner, execution reality, and risk shield.
"""

from __future__ import annotations

from helion_risk_world.config.execution_config import CostModelConfig
from helion_risk_world.config.risk_config import RiskShieldConfig
from helion_risk_world.planner.management_loop import ManagementLoop
from helion_risk_world.planner.mpc_planner import MPCPlanner
from helion_risk_world.planner.protocols import PlannerProtocol
from helion_risk_world.schemas.action_schema import FinalDecision
from helion_risk_world.schemas.execution_schema import ExecutionState
from helion_risk_world.schemas.portfolio_schema import PortfolioState, RiskProfile
from helion_risk_world.schemas.prediction_schema import ModelPrediction
from helion_risk_world.strategy.profiles import StrategyProfile


class StrategyPlanner(PlannerProtocol):
    """Adapt one base planner to a specific strategy profile."""

    def __init__(self, profile: StrategyProfile, planner: PlannerProtocol) -> None:
        self._profile = profile
        self._planner = planner

    @property
    def profile(self) -> StrategyProfile:
        return self._profile

    @classmethod
    def default(
        cls,
        profile: StrategyProfile,
        *,
        risk_cfg: RiskShieldConfig | None = None,
        cost_cfg: CostModelConfig | None = None,
    ) -> StrategyPlanner:
        planner = MPCPlanner.default(
            planner_cfg=profile.planner_config,
            risk_cfg=risk_cfg,
            cost_cfg=cost_cfg,
            confidence_scale=profile.confidence_scale,
            management_loop=ManagementLoop(max_hold_bars=profile.max_hold_bars),
            strategy_name=profile.name.value,
            decision_horizon_bars=profile.decision_horizon_bars,
        )
        return cls(profile, planner)

    def plan(
        self,
        prediction: ModelPrediction,
        state: PortfolioState,
        risk: RiskProfile,
        market: ExecutionState | None = None,
    ) -> FinalDecision:
        effective_risk = self.adapt_risk(risk)
        return self._planner.plan(
            prediction,
            state,
            effective_risk,
            market,
        )

    def adapt_risk(self, risk: RiskProfile) -> RiskProfile:
        """Apply the strategy's account-level risk overlay."""
        return self._profile.apply_risk(risk)

    def set_confidence_scale(self, confidence_scale: float) -> None:
        setter = getattr(self._planner, "set_confidence_scale", None)
        if callable(setter):
            setter(confidence_scale)


__all__ = ["StrategyPlanner"]
