"""Executable futures sizing/regression tests."""

from __future__ import annotations

from datetime import datetime

from helion_risk_world.execution.order_builder import build_candidate_order
from helion_risk_world.schemas import ActionType, CandidateAction, PortfolioState
from helion_risk_world.schemas.execution_schema import ExecutionState
from helion_risk_world.worlds.portfolio_world import PortfolioWorld

TS = datetime(2026, 7, 1, 10, 0)


def _state(capital: float = 500_000.0) -> PortfolioState:
    return PortfolioState(ts=TS, capital0=capital, capital=capital, cash=capital, free_margin=capital)


def _fut_market(price: float = 50_000.0) -> ExecutionState:
    return ExecutionState(
        symbol="BANKNIFTY_FUT_continuous",
        ts=TS,
        available_at=TS,
        bid=price - 10.0,
        ask=price + 10.0,
        spread=20.0,
    )


def test_order_builder_quantizes_to_integer_futures_contracts() -> None:
    order = build_candidate_order(
        CandidateAction(action_type=ActionType.ENTER_LONG, size_fraction=1.0),
        _state(),
        _fut_market(),
        max_exposure=1.0,
    )

    assert order is not None
    assert order.qty == 1.0
    assert order.side == "buy"
    assert order.notional == 1_500_000.0


def test_apply_fill_uses_margin_budget_for_futures_and_realizes_contract_pnl() -> None:
    next_state = PortfolioWorld.apply_fill(
        _state(),
        CandidateAction(action_type=ActionType.ENTER_LONG, size_fraction=1.0),
        realized_return=0.01,
        cost=0.0,
        max_exposure=1.0,
        market=_fut_market(),
    )

    assert next_state.position_qty == 1.0
    assert next_state.exposure == 0.75
    assert next_state.margin_used == 375_000.0
    assert next_state.free_margin == 140_000.0
    assert next_state.capital == 515_000.0
