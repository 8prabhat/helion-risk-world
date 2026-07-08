"""NO_TRADE must always be a first-class candidate (SPEC.md §4, §21, §27)."""

from __future__ import annotations

from datetime import datetime

import pytest

from helion_risk_world.planner.action_sampler import ActionSampler
from helion_risk_world.schemas import ActionType, CandidateAction, PortfolioState, RiskProfile
from helion_risk_world.schemas.portfolio_schema import PositionSide

TS = datetime(2026, 6, 25, 10, 0)
RISK = RiskProfile(
    name="balanced", max_risk_per_trade=0.01, max_daily_loss=0.02, max_weekly_loss=0.05,
    max_drawdown=0.10, max_exposure=1.0, max_trades_per_day=10, consecutive_loss_cooldown=4,
)


def _state(**kw: object) -> PortfolioState:
    base = dict(ts=TS, capital0=500_000.0, capital=500_000.0, cash=500_000.0)
    base.update(kw)
    return PortfolioState(**base)  # type: ignore[arg-type]


def test_no_trade_is_a_defined_action_type() -> None:
    assert ActionType.NO_TRADE.value == "no_trade"


def test_size_grid_must_include_zero_for_no_trade() -> None:
    ActionSampler(sizes=(0.0, 0.1, 0.5, 1.0))  # ok
    with pytest.raises(ValueError):
        ActionSampler(sizes=(0.1, 0.5, 1.0))   # missing NO_TRADE size


def test_no_trade_candidate_is_constructible_at_zero_size() -> None:
    a = CandidateAction(action_type=ActionType.NO_TRADE, size_fraction=0.0)
    assert a.size_fraction == 0.0


def test_no_trade_is_always_first_candidate() -> None:
    sampler = ActionSampler(sizes=(0.0, 0.25, 1.0))
    flat = sampler.enumerate(_state(), RISK)
    held = sampler.enumerate(_state(position=PositionSide.LONG, exposure=0.5), RISK)
    assert flat[0].action_type is ActionType.NO_TRADE
    assert held[0].action_type is ActionType.NO_TRADE


def test_flat_account_can_enter_long_or_short_only() -> None:
    kinds = {a.action_type for a in ActionSampler(sizes=(0.0, 0.5)).enumerate(_state(), RISK)}
    assert kinds == {ActionType.NO_TRADE, ActionType.ENTER_LONG, ActionType.ENTER_SHORT}


def test_held_account_can_exit_and_reduce() -> None:
    """V1: in-position candidates are EXIT and REDUCE only (no pyramiding / INCREASE)."""
    held = _state(position=PositionSide.LONG, exposure=0.5)
    kinds = {a.action_type for a in ActionSampler(sizes=(0.0, 0.5)).enumerate(held, RISK)}
    assert ActionType.EXIT in kinds and ActionType.REDUCE in kinds
    assert ActionType.INCREASE not in kinds  # pyramiding disabled in V1


def test_max_size_caps_positive_candidates() -> None:
    capped = ActionSampler(sizes=(0.0, 0.25, 0.5, 1.0)).enumerate(_state(), RISK, max_size=0.25)
    assert all(a.size_fraction <= 0.25 for a in capped)
