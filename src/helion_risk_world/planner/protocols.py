"""Planner interfaces shared by runtime surfaces.

Backtesting and paper trading depend on this protocol instead of a concrete planner so
strategy-aware wrappers can reuse the same execution surfaces without duplication.
"""

from __future__ import annotations

from typing import Protocol

from helion_risk_world.schemas.action_schema import FinalDecision
from helion_risk_world.schemas.execution_schema import ExecutionState
from helion_risk_world.schemas.portfolio_schema import PortfolioState, RiskProfile
from helion_risk_world.schemas.prediction_schema import ModelPrediction


class PlannerProtocol(Protocol):
    """Decision policy contract for backtest and paper-trading loops."""

    def adapt_risk(self, risk: RiskProfile) -> RiskProfile: ...

    def plan(
        self,
        prediction: ModelPrediction,
        state: PortfolioState,
        risk: RiskProfile,
        market: ExecutionState | None = None,
    ) -> FinalDecision: ...


__all__ = ["PlannerProtocol"]
