from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from helion_risk_world.config.execution_config import CostModelConfig
from helion_risk_world.execution.execution_reality import ExecutionReality
from helion_risk_world.execution.order_builder import build_candidate_order
from helion_risk_world.schemas.action_schema import ActionType, CandidateAction
from helion_risk_world.schemas.execution_schema import ExecutionState
from helion_risk_world.schemas.portfolio_schema import PortfolioState

_EPS = 1e-9


@runtime_checkable
class BrokerAdapterProtocol(Protocol):
    """Broker contract the paper engine depends on.

    DIP — never a concrete broker (SPEC.md §24, §26).
    """

    def place(
        self,
        action: CandidateAction,
        *,
        market: ExecutionState | None = None,
        portfolio_state: PortfolioState | None = None,
        max_exposure: float = 1.0,
        expected_edge: float | None = None,
    ) -> Any: ...

    def positions(self) -> Any: ...


@dataclass(frozen=True)
class PaperFill:
    requested_action: CandidateAction
    executed_action: CandidateAction
    status: str
    cost: float = 0.0
    spread_cost: float = 0.0
    statutory_fees: float = 0.0
    slippage: float = 0.0
    fill_fraction: float = 1.0
    requested_notional: float = 0.0
    executed_notional: float = 0.0
    fill_prob: float = 1.0
    partial_fill_prob: float = 0.0
    reject_prob: float = 0.0
    latency_ms: float = 0.0
    execution_realism: str = "high"
    note: str = "dry_run"


class DryRunBrokerAdapter:
    """Dry-run adapter: simulates fills, never sends real orders (SPEC.md §24)."""

    def __init__(self) -> None:
        self._fills: list[PaperFill] = []

    def place(
        self,
        action: CandidateAction,
        *,
        market: ExecutionState | None = None,
        portfolio_state: PortfolioState | None = None,
        max_exposure: float = 1.0,
        expected_edge: float | None = None,
    ) -> Any:
        requested_notional = 0.0
        executed_notional = 0.0
        if market is not None and portfolio_state is not None:
            order = build_candidate_order(
                action,
                portfolio_state,
                market,
                max_exposure=max_exposure,
            )
            if order is not None:
                requested_notional = float(order.notional)
                executed_notional = float(order.notional)
        fill = PaperFill(
            requested_action=action,
            executed_action=action,
            status="accepted",
            requested_notional=requested_notional,
            executed_notional=executed_notional,
        )
        self._fills.append(fill)
        return fill

    def positions(self) -> Any:
        return list(self._fills)


class ExecutionRealityBrokerAdapter:
    """Cost-aware dry-run broker that samples fills from the Execution Reality Layer."""

    def __init__(
        self,
        *,
        execution_reality: ExecutionReality | None = None,
        cost_cfg: CostModelConfig | None = None,
        partial_fill_fraction: float = 0.5,
        seed: int = 7,
    ) -> None:
        if not 0.0 < partial_fill_fraction < 1.0:
            raise ValueError("partial_fill_fraction must be in (0, 1)")
        self._execution = execution_reality or ExecutionReality(cost_cfg or CostModelConfig())
        self._partial_fill_fraction = partial_fill_fraction
        self._rng = np.random.default_rng(seed)
        self._fills: list[PaperFill] = []

    def place(
        self,
        action: CandidateAction,
        *,
        market: ExecutionState | None = None,
        portfolio_state: PortfolioState | None = None,
        max_exposure: float = 1.0,
        expected_edge: float | None = None,
    ) -> Any:
        if market is None or portfolio_state is None:
            fill = PaperFill(
                requested_action=action,
                executed_action=action,
                status="accepted",
                note="missing_context",
            )
            self._fills.append(fill)
            return fill

        order = build_candidate_order(
            action,
            portfolio_state,
            market,
            max_exposure=max_exposure,
        )
        if order is None:
            fill = PaperFill(
                requested_action=action,
                executed_action=action,
                status="no_trade",
                execution_realism="high",
                note="zero_notional",
            )
            self._fills.append(fill)
            return fill

        estimate = self._execution.estimate(order, market, expected_edge=expected_edge)
        draw = float(self._rng.uniform())
        if draw < estimate.fill_prob:
            status = "accepted"
            fill_fraction = 1.0
        elif draw < estimate.fill_prob + estimate.partial_fill_prob:
            status = "partial"
            fill_fraction = self._partial_fill_fraction
        else:
            status = "rejected"
            fill_fraction = 0.0

        executed_action = _scaled_action(action, fill_fraction)
        executed_notional = float(order.notional * fill_fraction)
        scale = 0.0 if fill_fraction <= _EPS else fill_fraction
        fill = PaperFill(
            requested_action=action,
            executed_action=executed_action,
            status=status,
            cost=float(estimate.total_cost * scale),
            spread_cost=float(estimate.spread_cost * scale),
            statutory_fees=float(estimate.statutory_fees * scale),
            slippage=float(estimate.slippage * scale),
            fill_fraction=fill_fraction,
            requested_notional=float(order.notional),
            executed_notional=executed_notional,
            fill_prob=float(estimate.fill_prob),
            partial_fill_prob=float(estimate.partial_fill_prob),
            reject_prob=float(estimate.reject_prob),
            latency_ms=float(estimate.latency_ms),
            execution_realism=estimate.realism.value,
            note="execution_reality",
        )
        self._fills.append(fill)
        return fill

    def positions(self) -> Any:
        return list(self._fills)


def _scaled_action(action: CandidateAction, fill_fraction: float) -> CandidateAction:
    fill_fraction = float(max(0.0, min(1.0, fill_fraction)))
    if fill_fraction <= _EPS:
        return CandidateAction(action_type=ActionType.NO_TRADE, size_fraction=0.0)
    if fill_fraction >= 1.0 - _EPS:
        return action
    if action.action_type is ActionType.EXIT:
        return CandidateAction(action_type=ActionType.REDUCE, size_fraction=fill_fraction)
    return CandidateAction(
        action_type=action.action_type,
        size_fraction=float(action.size_fraction * fill_fraction),
    )
