"""ManagementLoop.should_exit_early (SPEC.md §19), including the non-finite-reading guard."""

from __future__ import annotations

from datetime import datetime

import pytest

from helion_risk_world.planner.management_loop import ManagementLoop
from helion_risk_world.schemas.portfolio_schema import PortfolioState, PositionSide
from helion_risk_world.schemas.prediction_schema import (
    BarrierProbabilities,
    HorizonPrediction,
    ModelPrediction,
)

TS = datetime(2026, 6, 25, 10, 0)


def _prediction(*, ood_score: float = 0.0, epistemic: float = 0.0) -> ModelPrediction:
    hp = HorizonPrediction(
        horizon_bars=12,
        return_quantiles={0.1: -0.01, 0.25: 0.0, 0.5: 0.01, 0.75: 0.02, 0.9: 0.03},
        volatility=0.02,
    )
    barrier = BarrierProbabilities(stop=0.2, target=0.6, timeout=0.2)
    return ModelPrediction.model_construct(
        symbol="BANKNIFTY", ts=TS, horizon_preds=[hp], barrier=barrier, mae=0.02, mfe=0.0,
        sigma_H=0.02, stop_return=None, target_return=None, regime_probs=None,
        epistemic=epistemic, aleatoric=0.1, ood_score=ood_score, epistemic_calibrated=True,
    )


def _state(position: PositionSide) -> PortfolioState:
    return PortfolioState(
        ts=TS, capital0=1e6, capital=1e6, cash=1e6, free_margin=1e6, position=position,
    )


def test_holds_when_all_signals_are_calm() -> None:
    loop = ManagementLoop(ood_threshold=0.9, epistemic_threshold=0.5, max_hold_bars=12)
    exit_signal, reason = loop.should_exit_early(_state(PositionSide.LONG), _prediction(), bars_in_position=1)
    assert not exit_signal
    assert reason == ""


def test_exits_on_ood_above_threshold() -> None:
    loop = ManagementLoop(ood_threshold=0.9, epistemic_threshold=0.5, max_hold_bars=12)
    exit_signal, reason = loop.should_exit_early(
        _state(PositionSide.LONG), _prediction(ood_score=0.95), bars_in_position=1
    )
    assert exit_signal and reason == "ood_above_threshold"


def test_exits_on_non_finite_ood_reading() -> None:
    """Review finding H11: a NaN reading must trigger the exit, not silently pass."""
    loop = ManagementLoop(ood_threshold=0.9, epistemic_threshold=0.5, max_hold_bars=12)
    exit_signal, reason = loop.should_exit_early(
        _state(PositionSide.LONG), _prediction(ood_score=float("nan")), bars_in_position=1
    )
    assert exit_signal and reason == "ood_above_threshold"


def test_exits_on_non_finite_epistemic_reading() -> None:
    loop = ManagementLoop(ood_threshold=0.9, epistemic_threshold=0.5, max_hold_bars=12)
    exit_signal, reason = loop.should_exit_early(
        _state(PositionSide.LONG), _prediction(epistemic=float("nan")), bars_in_position=1
    )
    assert exit_signal and reason == "epistemic_above_threshold"
