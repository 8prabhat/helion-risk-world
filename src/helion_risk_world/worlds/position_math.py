"""Shared portfolio/execution position math.

This module owns the translation from abstract planner actions to executable positions so the
planner, execution layer, paper trading, and backtest all use identical sizing semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from helion_risk_world.config.execution_config import CostModelConfig
from helion_risk_world.execution.instrument_specs import resolve_instrument_spec
from helion_risk_world.schemas.action_schema import ActionType, CandidateAction
from helion_risk_world.schemas.execution_schema import ExecutionState
from helion_risk_world.schemas.portfolio_schema import PortfolioState, PositionSide

_EPS = 1e-9
_SIGN = {PositionSide.LONG: 1.0, PositionSide.SHORT: -1.0, PositionSide.FLAT: 0.0}


def signed_fraction(state: PortfolioState) -> float:
    return _SIGN[state.position] * state.exposure


def target_fraction(action: CandidateAction, current: float, max_exposure: float) -> float:
    """Signed target exposure fraction implied by ``action``."""
    s = action.size_fraction
    match action.action_type:
        case ActionType.NO_TRADE:
            target = current
        case ActionType.ENTER_LONG:
            target = s
        case ActionType.ENTER_SHORT:
            target = -s
        case ActionType.EXIT:
            target = 0.0
        case ActionType.REDUCE:
            target = current * (1.0 - s)
        case ActionType.INCREASE:
            direction = 1.0 if current >= 0 else -1.0
            target = current + direction * s if abs(current) > _EPS else 0.0
        case _:
            target = current
    return float(np.clip(target, -max_exposure, max_exposure))


def resolve_position(
    state: PortfolioState,
    action: CandidateAction,
    max_exposure: float,
) -> tuple[float, float, float]:
    """Continuous fallback position math.

    Returns ``(new_signed_fraction, new_signed_notional, traded_notional)`` using the original
    V1 semantics where exposure is a fraction of account capital.
    """
    old = signed_fraction(state)
    new = target_fraction(action, old, max_exposure)
    return new, new * state.capital, abs(new - old) * state.capital


@dataclass(frozen=True)
class ResolvedPosition:
    """Executable position state shared by planner, execution, and settlement."""

    new_signed_fraction: float
    new_signed_notional: float
    delta_signed_notional: float
    traded_notional: float
    new_margin_used: float
    traded_margin: float
    position_qty: float
    order_qty: float
    used_contract_spec: bool = False


def resolve_executable_position(
    state: PortfolioState,
    action: CandidateAction,
    max_exposure: float,
    *,
    market: ExecutionState | None = None,
    execution_cfg: CostModelConfig | None = None,
) -> ResolvedPosition:
    """Resolve ``action`` into an executable position and order size.

    When a contract spec exists for the symbol and a mid price is available, exposure is interpreted
    as committed margin fraction and position quantity becomes an integer contract count. Otherwise
    the system falls back to the legacy continuous-notional semantics.
    """
    cap = max(float(state.capital), _EPS)
    cfg = execution_cfg or CostModelConfig()
    current = signed_fraction(state)
    target = target_fraction(action, current, max_exposure)
    mid = _mid_price(market)
    spec = resolve_instrument_spec(market.symbol, cfg) if market is not None else None
    if spec is None or mid is None or mid <= _EPS:
        new_frac, new_notional, traded_notional = resolve_position(state, action, max_exposure)
        old_notional = current * cap
        delta_notional = new_notional - old_notional
        order_qty = abs(delta_notional / mid) if mid is not None and mid > _EPS else abs(delta_notional)
        return ResolvedPosition(
            new_signed_fraction=new_frac,
            new_signed_notional=new_notional,
            delta_signed_notional=delta_notional,
            traded_notional=traded_notional,
            new_margin_used=abs(new_notional),
            traded_margin=traded_notional,
            position_qty=abs(new_notional),
            order_qty=order_qty,
            used_contract_spec=False,
        )

    contract_notional = mid * spec.lot_size
    margin_per_contract = contract_notional * spec.margin_fraction
    if contract_notional <= _EPS or margin_per_contract <= _EPS:
        return resolve_executable_position(
            state,
            action,
            max_exposure,
            market=None,
            execution_cfg=cfg,
        )

    current_contracts = _current_contracts(state, cap, margin_per_contract)
    target_contracts = _target_contracts(target, cap, margin_per_contract)
    delta_contracts = target_contracts - current_contracts
    new_notional = target_contracts * contract_notional
    delta_notional = delta_contracts * contract_notional
    new_margin_used = abs(target_contracts) * margin_per_contract
    new_frac = 0.0
    if abs(target_contracts) > _EPS:
        new_frac = (new_margin_used / cap) * (1.0 if target_contracts > 0 else -1.0)
    return ResolvedPosition(
        new_signed_fraction=float(new_frac),
        new_signed_notional=float(new_notional),
        delta_signed_notional=float(delta_notional),
        traded_notional=float(abs(delta_notional)),
        new_margin_used=float(new_margin_used),
        traded_margin=float(abs(delta_contracts) * margin_per_contract),
        position_qty=float(abs(target_contracts)),
        order_qty=float(abs(delta_contracts)),
        used_contract_spec=True,
    )


def minimum_contract_margin_fraction(
    capital: float,
    market: ExecutionState | None,
    execution_cfg: CostModelConfig | None = None,
) -> float | None:
    """Return the minimum capital fraction needed to open one contract at ``market``.

    ``None`` means the symbol is not governed by a discrete contract spec or the price is
    unavailable, so no discrete-contract feasibility check can be made.
    """
    cap = max(float(capital), _EPS)
    cfg = execution_cfg or CostModelConfig()
    if market is None:
        return None
    spec = resolve_instrument_spec(market.symbol, cfg)
    mid = _mid_price(market)
    if spec is None or mid is None or mid <= _EPS:
        return None
    margin_per_contract = mid * spec.lot_size * spec.margin_fraction
    if margin_per_contract <= _EPS:
        return None
    return float(margin_per_contract / cap)


def _mid_price(market: ExecutionState | None) -> float | None:
    if market is None:
        return None
    bid = market.bid
    ask = market.ask
    if bid is not None and ask is not None and bid > 0.0 and ask > 0.0:
        return float((bid + ask) / 2.0)
    if bid is not None and bid > 0.0:
        return float(bid)
    if ask is not None and ask > 0.0:
        return float(ask)
    return None


def _current_contracts(
    state: PortfolioState,
    capital: float,
    margin_per_contract: float,
) -> int:
    sign = int(_SIGN[state.position])
    if sign == 0:
        return 0
    if state.margin_used > _EPS:
        return sign * int(round(state.margin_used / margin_per_contract))
    if state.exposure > _EPS:
        return sign * int(round((state.exposure * capital) / margin_per_contract))
    if state.position_qty > _EPS:
        return sign * int(round(state.position_qty))
    return 0


def _target_contracts(target_fraction_value: float, capital: float, margin_per_contract: float) -> int:
    if abs(target_fraction_value) <= _EPS:
        return 0
    budget = abs(target_fraction_value) * capital
    contracts = int(np.floor((budget + _EPS) / margin_per_contract))
    if contracts <= 0:
        return 0
    return contracts if target_fraction_value > 0 else -contracts


__all__ = [
    "ResolvedPosition",
    "minimum_contract_margin_fraction",
    "resolve_executable_position",
    "resolve_position",
    "signed_fraction",
    "target_fraction",
]
