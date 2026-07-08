"""Build execution-plane orders from planner actions without duplicating sizing logic."""

from __future__ import annotations

from helion_risk_world.config.execution_config import CostModelConfig
from helion_risk_world.schemas.action_schema import ActionType, CandidateAction
from helion_risk_world.schemas.execution_schema import CandidateOrder, ExecutionState
from helion_risk_world.schemas.portfolio_schema import PortfolioState
from helion_risk_world.worlds.position_math import resolve_executable_position

_EPS = 1e-9


def build_candidate_order(
    action: CandidateAction,
    state: PortfolioState,
    market: ExecutionState,
    *,
    max_exposure: float,
    cost_cfg: CostModelConfig | None = None,
) -> CandidateOrder | None:
    """Translate a planner action into a concrete order for execution/cost estimation."""
    resolved = resolve_executable_position(
        state,
        action,
        max_exposure,
        market=market,
        execution_cfg=cost_cfg,
    )
    if action.action_type is ActionType.NO_TRADE or resolved.traded_notional <= _EPS:
        return None

    side = "buy" if resolved.delta_signed_notional > 0 else "sell"
    qty = resolved.order_qty
    if qty <= _EPS:
        return None
    return CandidateOrder(
        symbol=market.symbol,
        side=side,
        qty=qty,
        notional=resolved.traded_notional,
    )


__all__ = ["build_candidate_order"]
