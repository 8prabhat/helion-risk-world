"""quantile_stop_return/quantile_target_return (2026-07-16) -- sizing stop/target from
the model's own predicted return-quantile distribution instead of a fixed symmetric
BarrierContext multiplier. See ModelPrediction.quantile_stop_return's docstring for the
diagnosis this responds to (CVaR-dominated zero-trade backtest, traced to a frozen
±1.0 symmetric barrier not matching regime-non-stationary real touch frequency).
"""

from __future__ import annotations

from datetime import datetime

from helion_risk_world.schemas.prediction_schema import (
    BarrierProbabilities,
    HorizonPrediction,
    ModelPrediction,
)

TS = datetime(2026, 6, 25, 10, 0)


def _prediction(quantiles: dict[float, float], *, stop_return=None, target_return=None) -> ModelPrediction:
    hp = HorizonPrediction(horizon_bars=48, return_quantiles=quantiles, volatility=0.01)
    barrier = BarrierProbabilities(stop=0.3, target=0.3, timeout=0.4)
    return ModelPrediction(
        symbol="BANKNIFTY", ts=TS,
        horizon_preds=[hp], barrier=barrier, mae=0.02, sigma_H=0.01,
        stop_return=stop_return, target_return=target_return,
        epistemic=0.0, aleatoric=0.1, ood_score=0.0,
    )


_SKEWED = {0.1: -0.005, 0.25: -0.002, 0.5: 0.001, 0.75: 0.008, 0.9: 0.012}
_SYMMETRIC = {0.1: -0.01, 0.25: -0.005, 0.5: 0.0, 0.75: 0.005, 0.9: 0.01}


def test_quantile_stop_and_target_are_asymmetric_when_quantiles_are_skewed() -> None:
    pred = _prediction(_SKEWED)
    stop = pred.quantile_stop_return()
    target = pred.quantile_target_return()
    assert stop == -0.005
    assert target == 0.012
    assert abs(stop) != abs(target)  # genuinely asymmetric, unlike the fixed-mult path


def test_quantile_stop_and_target_symmetric_case_matches_magnitude() -> None:
    pred = _prediction(_SYMMETRIC)
    assert pred.quantile_stop_return() == -0.01
    assert pred.quantile_target_return() == 0.01


def test_quantile_stop_return_sign_clamped_even_if_low_quantile_is_positive() -> None:
    """If the whole predicted distribution is shifted positive (even q10 > 0), the stop
    return must still come back non-positive -- never silently flip sign."""
    shifted = {0.1: 0.001, 0.25: 0.003, 0.5: 0.006, 0.75: 0.009, 0.9: 0.014}
    pred = _prediction(shifted)
    assert pred.quantile_stop_return() <= 0.0
    assert pred.quantile_target_return() >= 0.0


def test_quantile_returns_respect_min_abs_return_floor() -> None:
    tiny = {0.1: -0.0001, 0.25: -0.00005, 0.5: 0.0, 0.75: 0.00005, 0.9: 0.0001}
    pred = _prediction(tiny)
    stop = pred.quantile_stop_return(min_abs_return=0.003)
    target = pred.quantile_target_return(min_abs_return=0.003)
    assert stop == -0.003
    assert target == 0.003


def test_quantile_resolvers_fall_back_when_level_missing() -> None:
    partial = {0.25: -0.002, 0.5: 0.001, 0.75: 0.008}  # no 0.1 or 0.9 keys
    pred = _prediction(partial, stop_return=-0.02, target_return=0.02)
    # falls back to resolved_stop_return()/resolved_target_return(), which use the
    # explicit BarrierContext-derived value since it's present here
    assert pred.quantile_stop_return() == -0.02
    assert pred.quantile_target_return() == 0.02


def test_quantile_for_side_swaps_correctly_for_short() -> None:
    pred = _prediction(_SKEWED)
    long_stop = pred.quantile_stop_return_for_side("long")
    long_target = pred.quantile_target_return_for_side("long")
    short_stop = pred.quantile_stop_return_for_side("short")
    short_target = pred.quantile_target_return_for_side("short")
    assert long_stop == pred.quantile_stop_return()
    assert long_target == pred.quantile_target_return()
    # short's adverse move is the magnitude of long's favourable move, and vice versa
    assert short_stop == abs(long_target)
    assert short_target == -abs(long_stop)
