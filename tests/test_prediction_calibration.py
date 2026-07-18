from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from helion_risk_world.prediction_calibration import (
    HorizonPredictionCalibration,
    PredictionCalibration,
    fit_prediction_calibration,
)
from helion_risk_world.schemas.market_schema import Regime
from helion_risk_world.schemas.prediction_schema import (
    BarrierProbabilities,
    HorizonPrediction,
    ModelPrediction,
)


def test_fit_prediction_calibration_learns_positive_shift_and_temperature() -> None:
    realized = np.array([-0.02, -0.01, 0.0, 0.01, 0.02], dtype=float)
    pred_quantiles = np.stack(
        [realized - 0.02, realized - 0.015, realized - 0.01, realized - 0.005, realized],
        axis=1,
    )
    predicted_volatility = np.full_like(realized, 0.01)
    realized_volatility = np.full_like(realized, 0.02)
    # Barrier temperature fitting now requires >= 200 samples split into a chronological
    # fit/check holdout (2026-07-13 safeguard against overfitting a small calibration
    # sample -- see prediction_calibration.py's module docstring). Repeat the same
    # systematically-overconfident-and-often-wrong pattern many times so both the fit half
    # and the held-out check half see the same strong, consistent miscalibration signal.
    n = 240
    barrier_probs = np.tile(np.array([0.99, 0.005, 0.005], dtype=float), (n, 1))
    barrier_labels = np.tile(np.array([0, 1], dtype=int), n // 2)

    calibration = fit_prediction_calibration(
        quantile_levels=(0.1, 0.25, 0.5, 0.75, 0.9),
        horizon_payloads={
            12: {
                "pred_quantiles": pred_quantiles,
                "realized": realized,
                "predicted_volatility": predicted_volatility,
                "realized_volatility": realized_volatility,
            }
        },
        barrier_probs=barrier_probs,
        barrier_labels=barrier_labels,
        source="unit_test",
    )

    assert calibration is not None
    horizon = calibration.horizons[12]
    assert horizon.quantile_offsets[0.5] > 0.009
    assert horizon.volatility_scale >= 1.5
    assert calibration.barrier_temperature > 1.0


def test_fit_prediction_calibration_keeps_identity_barrier_temperature_when_already_perfect() -> None:
    realized = np.array([-0.01, 0.0, 0.01], dtype=float)
    pred_quantiles = np.stack(
        [realized - 0.01, realized - 0.005, realized, realized + 0.005, realized + 0.01],
        axis=1,
    )
    perfect_barrier_probs = np.eye(3, dtype=float)
    calibration = fit_prediction_calibration(
        quantile_levels=(0.1, 0.25, 0.5, 0.75, 0.9),
        horizon_payloads={
            12: {
                "pred_quantiles": pred_quantiles,
                "realized": realized,
                "predicted_volatility": np.full_like(realized, 0.01),
                "realized_volatility": np.full_like(realized, 0.01),
            }
        },
        barrier_probs=perfect_barrier_probs,
        barrier_labels=np.array([0, 1, 2], dtype=int),
        source="unit_test",
    )

    assert calibration is not None
    assert calibration.barrier_temperature == 1.0


def test_fit_prediction_calibration_widens_too_narrow_interval() -> None:
    """Tier 2 (2026-07-05): a location-only shift cannot fix a systematically too-narrow
    interval -- CQR must WIDEN the outer/inner pairs, not just re-center them. Synthetic
    realized outcomes are much more spread out than the (tightly clustered) predicted
    quantiles, mirroring the real diagnostic finding (every level had ~40-58% empirical
    coverage regardless of nominal target)."""
    rng = np.random.default_rng(0)
    n = 2000
    realized = rng.normal(loc=0.0, scale=0.02, size=n)
    # Predicted quantiles are all clustered tightly around 0 -- far narrower than the
    # true spread of realized outcomes.
    narrow_quantiles = np.tile(np.array([-0.002, -0.001, 0.0, 0.001, 0.002]), (n, 1))

    calibration = fit_prediction_calibration(
        quantile_levels=(0.1, 0.25, 0.5, 0.75, 0.9),
        horizon_payloads={
            12: {
                "pred_quantiles": narrow_quantiles,
                "realized": realized,
                "predicted_volatility": np.full(n, 0.01),
                "realized_volatility": np.full(n, 0.02),
            }
        },
        source="unit_test",
    )

    assert calibration is not None
    offsets = calibration.horizons[12].quantile_offsets
    # The outer pair must widen: 0.1 pushed down (negative), 0.9 pushed up (positive).
    assert offsets[0.1] < -0.01
    assert offsets[0.9] > 0.01
    # Inner pair also widens, by less than the outer pair (tighter nominal interval).
    assert offsets[0.25] < 0.0
    assert offsets[0.75] > 0.0
    assert abs(offsets[0.1]) > abs(offsets[0.25])
    assert abs(offsets[0.9]) > abs(offsets[0.75])

    # Applying the fix should bring empirical coverage close to nominal, unlike the raw
    # (too-narrow) predictions which would cluster around ~50% coverage at every level.
    adjusted = narrow_quantiles + np.array([offsets[level] for level in sorted(offsets)])
    adjusted = np.maximum.accumulate(adjusted, axis=1)
    empirical_coverage = (realized[:, None] <= adjusted).mean(axis=0)
    nominal = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    assert np.abs(empirical_coverage - nominal).mean() < 0.05


def test_prediction_calibration_apply_preserves_schema_and_updates_management_fields() -> None:
    calibration = PredictionCalibration(
        quantile_levels=(0.1, 0.25, 0.5, 0.75, 0.9),
        horizons={
            12: HorizonPredictionCalibration(
                horizon_bars=12,
                quantile_offsets={0.1: -0.01, 0.25: 0.0, 0.5: 0.02, 0.75: 0.03, 0.9: 0.04},
                volatility_scale=1.5,
                volatility_bias=0.005,
                sample_count=32,
            )
        },
        barrier_temperature=2.0,
        regime_temperature=2.0,
        source="unit_test",
        sample_count=32,
    )
    raw = ModelPrediction(
        symbol="BANKNIFTY",
        ts=datetime(2026, 6, 30, 10, 0),
        horizon_preds=[
            HorizonPrediction(
                horizon_bars=12,
                return_quantiles={0.1: -0.03, 0.25: -0.02, 0.5: -0.01, 0.75: 0.0, 0.9: 0.01},
                volatility=0.02,
            )
        ],
        barrier=BarrierProbabilities(stop=0.1, target=0.8, timeout=0.1),
        mae=0.02,
        mfe=0.03,
        sigma_H=0.02,
        regime_probs={Regime.TREND: 0.9, Regime.RANGE: 0.1},
        epistemic=0.05,
        aleatoric=0.02,
        ood_score=0.1,
    )

    calibrated = calibration.apply(raw)

    assert calibrated.horizon_preds[0].return_quantiles[0.5] == 0.01
    assert calibrated.horizon_preds[0].volatility == pytest.approx(0.035)
    assert calibrated.mae > raw.mae
    assert calibrated.mfe > raw.mfe
    assert calibrated.sigma_H == pytest.approx(0.035)
    assert calibrated.barrier.target < raw.barrier.target
    assert abs(
        calibrated.barrier.stop + calibrated.barrier.target + calibrated.barrier.timeout - 1.0
    ) < 1e-6
    assert calibrated.regime_probs is not None
    assert calibrated.regime_probs[Regime.TREND] < raw.regime_probs[Regime.TREND]


def _horizon_payload_identity(n: int = 8) -> dict:
    realized = np.linspace(-0.02, 0.02, n)
    pred_quantiles = np.stack(
        [realized - 0.02, realized - 0.01, realized, realized + 0.01, realized + 0.02],
        axis=1,
    )
    return {
        12: {
            "pred_quantiles": pred_quantiles,
            "realized": realized,
            "predicted_volatility": np.full_like(realized, 0.01),
            "realized_volatility": np.full_like(realized, 0.01),
        }
    }


def test_barrier_prior_offsets_correct_class_weight_induced_prior_shift() -> None:
    """Class-weighted CE inflates minority-class probabilities relative to true base
    rates -- a prior shift temperature scaling cannot express. The fitted per-class
    log-prior offsets must pull the mean predicted distribution back toward the true
    base rates (2026-07-18)."""
    rng = np.random.default_rng(7)
    n = 600
    # True base rates ~ (27%, 3%, 70%); model systematically predicts ~(38%, 27%, 35%)
    # (the empirically observed class-weight distortion pattern from 2026-07-11).
    labels = rng.choice([0, 1, 2], size=n, p=[0.27, 0.03, 0.70])
    base = np.array([0.38, 0.27, 0.35], dtype=float)
    noise = rng.normal(0.0, 0.03, size=(n, 3))
    probs = np.clip(base.reshape(1, -1) + noise, 0.01, None)
    # Give predictions a little genuine signal so argmax isn't pure noise.
    probs[np.arange(n), labels] += 0.05
    probs = probs / probs.sum(axis=1, keepdims=True)

    calibration = fit_prediction_calibration(
        quantile_levels=(0.1, 0.25, 0.5, 0.75, 0.9),
        horizon_payloads=_horizon_payload_identity(),
        barrier_probs=probs,
        barrier_labels=labels,
        source="unit_test",
    )
    assert calibration is not None
    assert calibration.barrier_prior_offsets is not None
    offsets = np.asarray(calibration.barrier_prior_offsets)
    # The over-predicted minority class (target, index 1) must be pushed DOWN relative
    # to the under-predicted majority class (timeout, index 2).
    assert offsets[1] < offsets[2]

    # Applying the calibration must move the mean predicted distribution toward the
    # true base rates.
    pred = ModelPrediction(
        symbol="X", ts=datetime(2026, 7, 18, 10, 0),
        horizon_preds=[HorizonPrediction(
            horizon_bars=12,
            return_quantiles={0.1: -0.02, 0.25: -0.01, 0.5: 0.0, 0.75: 0.01, 0.9: 0.02},
            volatility=0.01,
        )],
        barrier=BarrierProbabilities(stop=0.38, target=0.27, timeout=0.35),
        mae=0.01, sigma_H=0.01, epistemic=0.0, aleatoric=0.01, ood_score=0.0,
    )
    calibrated = calibration.apply(pred)
    assert calibrated.barrier.target < pred.barrier.target
    assert calibrated.barrier.timeout > pred.barrier.timeout


def test_barrier_prior_offsets_rejected_when_probs_already_match_base_rates() -> None:
    """No prior shift -> the holdout check must reject the (near-zero) offsets rather
    than shipping a correction fit to noise."""
    rng = np.random.default_rng(11)
    n = 600
    p_true = [0.3, 0.3, 0.4]
    labels = rng.choice([0, 1, 2], size=n, p=p_true)
    probs = np.clip(
        np.array(p_true).reshape(1, -1) + rng.normal(0.0, 0.02, size=(n, 3)), 0.01, None
    )
    probs = probs / probs.sum(axis=1, keepdims=True)
    calibration = fit_prediction_calibration(
        quantile_levels=(0.1, 0.25, 0.5, 0.75, 0.9),
        horizon_payloads=_horizon_payload_identity(),
        barrier_probs=probs,
        barrier_labels=labels,
        source="unit_test",
    )
    assert calibration is not None
    if calibration.barrier_prior_offsets is not None:
        # If accepted at all, the shrunk offsets must be tiny.
        assert float(np.abs(np.asarray(calibration.barrier_prior_offsets)).max()) < 0.1


def test_barrier_prior_offsets_round_trip_metadata() -> None:
    calibration = PredictionCalibration(
        quantile_levels=(0.1, 0.9),
        horizons={},
        barrier_temperature=1.2,
        barrier_prior_offsets=(0.3, -0.8, 0.5),
        source="unit_test",
        sample_count=10,
    )
    restored = PredictionCalibration.from_metadata(calibration.to_metadata())
    assert restored is not None
    assert restored.barrier_prior_offsets == pytest.approx((0.3, -0.8, 0.5))
    # Absent from metadata -> None, not zeros.
    payload = calibration.to_metadata()
    payload.pop("barrier_prior_offsets")
    restored_none = PredictionCalibration.from_metadata(payload)
    assert restored_none is not None
    assert restored_none.barrier_prior_offsets is None


def test_calibration_apply_preserves_epistemic_calibrated_flag() -> None:
    """Regression (2026-07-18): apply() used to rebuild ModelPrediction without the
    epistemic_calibrated field, silently defaulting the placeholder flag back to True."""
    calibration = PredictionCalibration(
        quantile_levels=(0.1, 0.25, 0.5, 0.75, 0.9), horizons={},
    )
    pred = ModelPrediction(
        symbol="X", ts=datetime(2026, 7, 18, 10, 0),
        horizon_preds=[HorizonPrediction(
            horizon_bars=12,
            return_quantiles={0.1: -0.02, 0.25: -0.01, 0.5: 0.0, 0.75: 0.01, 0.9: 0.02},
            volatility=0.01,
        )],
        barrier=BarrierProbabilities(stop=0.3, target=0.3, timeout=0.4),
        mae=0.01, sigma_H=0.01, epistemic=0.0, aleatoric=0.01, ood_score=0.0,
        epistemic_calibrated=False,
    )
    assert calibration.apply(pred).epistemic_calibrated is False
