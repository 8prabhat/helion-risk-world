"""Forecaster -> ModelPrediction bridge (SPEC.md §17, §18, §27).

Direction head removed from spec. Barrier probabilities and regime_probs are at the
top level of ModelPrediction (not per HorizonPrediction).
"""

from __future__ import annotations

from datetime import datetime

import pytest

torch = pytest.importorskip("torch")

from helion_risk_world.config.model_config import LossWeights, ModelConfig  # noqa: E402
from helion_risk_world.config.training_config import TrainingConfig  # noqa: E402
from helion_risk_world.inference import ForecasterPredictor  # noqa: E402
from helion_risk_world.losses.composite_loss import ForecasterLoss  # noqa: E402
from helion_risk_world.model import HRWForecaster  # noqa: E402
from helion_risk_world.planner.mpc_planner import MPCPlanner  # noqa: E402
from helion_risk_world.schemas import ActionType, PortfolioState, RiskProfile  # noqa: E402
from helion_risk_world.schemas.prediction_schema import QUANTILE_LEVELS  # noqa: E402
from helion_risk_world.training.trainer import ForecastBatch, HRWTrainer  # noqa: E402

A, L, FEAT = 2, 12, 9
TS = datetime(2026, 6, 25, 10, 0)
RISK = RiskProfile(
    name="balanced", max_risk_per_trade=0.01, max_daily_loss=0.02, max_weekly_loss=0.05,
    max_drawdown=0.10, max_exposure=1.0, max_trades_per_day=100, consecutive_loss_cooldown=99,
    cvar_alpha=0.05, n_paths=256,
)


def _model() -> HRWForecaster:
    return HRWForecaster(n_features=FEAT, cfg=ModelConfig(latent_dim=16, temporal_layers=1,
                                                         dropout=0.0))


def _account() -> PortfolioState:
    cap = 500_000.0
    return PortfolioState(ts=TS, capital0=cap, capital=cap, cash=cap, free_margin=cap)


def test_predict_one_produces_valid_model_prediction() -> None:
    torch.manual_seed(0)
    pred = ForecasterPredictor(_model()).predict_one(torch.randn(A, L, FEAT), "BANKNIFTY", TS)
    assert pred.symbol == "BANKNIFTY" and len(pred.horizon_preds) == 1
    hp = pred.horizon_preds[0]
    # Quantile keys cover canonical levels; values are non-decreasing (schema-validated).
    assert set(hp.return_quantiles) == set(QUANTILE_LEVELS)
    vals = [hp.return_quantiles[q] for q in sorted(hp.return_quantiles)]
    assert all(b >= a - 1e-6 for a, b in zip(vals, vals[1:], strict=False))
    assert hp.volatility > 0
    assert pred.epistemic >= 0 and pred.aleatoric >= 0
    # Barrier at top level; sums to 1
    total = pred.barrier.stop + pred.barrier.target + pred.barrier.timeout
    assert total == pytest.approx(1.0, abs=1e-5)


def test_predict_batch_matches_batch_size() -> None:
    preds = ForecasterPredictor(_model()).predict_batch(
        torch.randn(3, A, L, FEAT), "NIFTY", [TS, TS, TS]
    )
    assert len(preds) == 3 and all(p.symbol == "NIFTY" for p in preds)


def test_predict_batch_rejects_mismatched_timestamps() -> None:
    with pytest.raises(ValueError):
        ForecasterPredictor(_model()).predict_batch(torch.randn(2, A, L, FEAT), "X", [TS])


def test_planner_consumes_forecaster_prediction() -> None:
    """The bridge output is directly usable by the MPC planner (end-to-end shape contract)."""
    pred = ForecasterPredictor(_model()).predict_one(torch.randn(A, L, FEAT), "BANKNIFTY", TS)
    decision = MPCPlanner.default().plan(pred, _account(), RISK)
    assert decision.final_action.action_type in set(ActionType)
    assert decision.candidates


def test_trained_model_produces_positive_edge() -> None:
    """After overfitting a positive-return slice, the bridged prediction has positive median."""
    torch.manual_seed(0)
    features = torch.randn(16, A, L, FEAT)
    signal = features.mean(dim=(1, 2, 3))
    forward_return = signal.abs() * 0.05 + 0.02            # strictly positive returns
    direction = torch.full((16,), 2, dtype=torch.long)     # all "up"
    batch = ForecastBatch(features=features, forward_return=forward_return, direction=direction)

    loss = ForecasterLoss(LossWeights(uncertainty=0.0, calibration=0.0))
    model = _model()
    HRWTrainer(model, loss, TrainingConfig(device="cpu", lr=5e-3, max_epochs=300,
                                           embargo_bars=12)).fit([batch])
    pred = ForecasterPredictor(model).predict_one(features[0], "BANKNIFTY", TS)
    assert pred.horizon_preds[0].return_quantiles[0.5] > 0   # learned positive median
    # Barrier head requires barrier labels to train; only the return signal is trained here.
