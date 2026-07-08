"""Regime head + OOD detector, and the OOD-quarantine Risk Shield rule (SPEC.md §17, §19, §27)."""

from __future__ import annotations

from datetime import datetime

import pytest

torch = pytest.importorskip("torch")

from helion_risk_world.config.model_config import ModelConfig  # noqa: E402
from helion_risk_world.config.risk_config import RiskShieldConfig  # noqa: E402
from helion_risk_world.data.primitives import regime_label  # noqa: E402
from helion_risk_world.heads.ood_head import OODHead  # noqa: E402
from helion_risk_world.heads.regime_head import REGIME_CLASSES, RegimeHead  # noqa: E402
from helion_risk_world.inference import ForecasterPredictor  # noqa: E402
from helion_risk_world.model import HRWForecaster  # noqa: E402
from helion_risk_world.risk.constraints import OODRule  # noqa: E402
from helion_risk_world.schemas import (  # noqa: E402
    ActionType,
    CandidateAction,
    PortfolioState,
    RiskProfile,
)
from helion_risk_world.schemas.market_schema import Regime  # noqa: E402

A, L, FEAT = 2, 12, 9
TS = datetime(2026, 6, 25, 10, 0)


def test_regime_label_heuristic() -> None:
    assert regime_label(0.0, realized_vol=0.05) is Regime.HIGH_VOL     # high vol dominates
    assert regime_label(0.03, realized_vol=0.008) is Regime.TREND      # strong move -> trend
    assert regime_label(0.0, realized_vol=0.001) is Regime.LOW_VOL     # quiet, tiny move
    assert regime_label(0.005, realized_vol=0.012) is Regime.CHOP      # mid vol, middling move


def test_regime_head_shape_and_class_order() -> None:
    out = RegimeHead(latent_dim=16).forward(torch.zeros(4, 16))
    assert out.shape == (4, len(REGIME_CLASSES))
    assert REGIME_CLASSES[0] is Regime.TREND and len(set(REGIME_CLASSES)) == len(Regime)


def test_ood_detector_low_in_distribution_high_outside() -> None:
    torch.manual_seed(0)
    det = OODHead(latent_dim=8)
    train = torch.randn(500, 8)
    det.fit(train)
    in_dist = float(det.score(torch.zeros(1, 8)).item())          # at the mean -> low
    out_dist = float(det.score(torch.full((1, 8), 20.0)).item())  # far away -> high
    assert in_dist < 0.5 < out_dist
    assert out_dist > 0.9


def test_unfitted_ood_scores_zero() -> None:
    assert float(OODHead(latent_dim=8).score(torch.randn(3, 8)).mean()) == 0.0


def test_bridge_emits_real_regime_probs_and_ood() -> None:
    torch.manual_seed(0)
    model = HRWForecaster(n_features=FEAT, cfg=ModelConfig(latent_dim=16, temporal_layers=1,
                                                          dropout=0.0))
    model.fit_ood(torch.randn(200, A, L, FEAT))
    pred = ForecasterPredictor(model).predict_one(torch.randn(A, L, FEAT), "BANKNIFTY", TS)
    # regime_probs is at the top level of ModelPrediction (not per HorizonPrediction)
    rp = pred.regime_probs
    assert rp is not None and set(rp) == set(Regime) and sum(rp.values()) == pytest.approx(1.0, abs=1e-5)
    assert 0.0 <= pred.ood_score


def test_ood_rule_quarantines_entry_on_high_ood() -> None:
    rule = OODRule(RiskShieldConfig(ood_block_threshold=0.9))
    from helion_risk_world.schemas.prediction_schema import (
        BarrierProbabilities,
        HorizonPrediction,
        ModelPrediction,
    )

    hp = HorizonPrediction(
        horizon_bars=12,
        return_quantiles={0.1: -0.01, 0.25: 0.0, 0.5: 0.01, 0.75: 0.02, 0.9: 0.03},
        volatility=0.02,
    )
    barrier = BarrierProbabilities(stop=0.4, target=0.4, timeout=0.2)
    weird = ModelPrediction(
        symbol="X", ts=TS,
        horizon_preds=[hp], barrier=barrier, mae=0.02, sigma_H=0.02,
        epistemic=0.0, aleatoric=0.1, ood_score=0.99,
    )
    state = PortfolioState(ts=TS, capital0=1e6, capital=1e6, cash=1e6, free_margin=1e6)
    risk = RiskProfile(name="b", max_risk_per_trade=0.01, max_daily_loss=0.02, max_weekly_loss=0.05,
                       max_drawdown=0.1, max_exposure=1.0, max_trades_per_day=10,
                       consecutive_loss_cooldown=4)
    long = CandidateAction(action_type=ActionType.ENTER_LONG, size_fraction=0.5)
    exit_ = CandidateAction(action_type=ActionType.EXIT, size_fraction=0.0)
    assert rule.check(long, state, risk, weird).reason_code == "OOD_QUARANTINE"
    assert rule.check(exit_, state, risk, weird).allowed  # de-risking still allowed


def test_ood_and_uncertainty_rules_block_on_non_finite_reading() -> None:
    """Review finding H11: a NaN reading (e.g. from a broken epistemic estimate)
    must never silently pass a `> threshold` check — treat it as maximally unsafe.
    Pydantic validation normally rejects NaN in ModelPrediction (ge=0.0 constraint),
    so simulate a prediction that reached this rule with a non-finite value anyway
    (e.g. constructed via a path that bypasses validation) using model_construct."""
    from helion_risk_world.risk.constraints import UncertaintyRule
    from helion_risk_world.schemas.prediction_schema import (
        BarrierProbabilities,
        HorizonPrediction,
        ModelPrediction,
    )

    hp = HorizonPrediction(
        horizon_bars=12,
        return_quantiles={0.1: -0.01, 0.25: 0.0, 0.5: 0.01, 0.75: 0.02, 0.9: 0.03},
        volatility=0.02,
    )
    barrier = BarrierProbabilities(stop=0.4, target=0.4, timeout=0.2)
    nan_ood = ModelPrediction.model_construct(
        symbol="X", ts=TS, horizon_preds=[hp], barrier=barrier, mae=0.02, mfe=0.0,
        sigma_H=0.02, stop_return=None, target_return=None, regime_probs=None,
        epistemic=0.0, aleatoric=0.1, ood_score=float("nan"), epistemic_calibrated=True,
    )
    nan_epistemic = ModelPrediction.model_construct(
        symbol="X", ts=TS, horizon_preds=[hp], barrier=barrier, mae=0.02, mfe=0.0,
        sigma_H=0.02, stop_return=None, target_return=None, regime_probs=None,
        epistemic=float("nan"), aleatoric=0.1, ood_score=0.0, epistemic_calibrated=True,
    )
    state = PortfolioState(ts=TS, capital0=1e6, capital=1e6, cash=1e6, free_margin=1e6)
    risk = RiskProfile(name="b", max_risk_per_trade=0.01, max_daily_loss=0.02, max_weekly_loss=0.05,
                       max_drawdown=0.1, max_exposure=1.0, max_trades_per_day=10,
                       consecutive_loss_cooldown=4)
    long = CandidateAction(action_type=ActionType.ENTER_LONG, size_fraction=0.5)

    ood_rule = OODRule(RiskShieldConfig(ood_block_threshold=0.9))
    assert ood_rule.check(long, state, risk, nan_ood).reason_code == "OOD_QUARANTINE"

    uncertainty_rule = UncertaintyRule(RiskShieldConfig(uncertainty_block_threshold=0.5))
    assert uncertainty_rule.check(long, state, risk, nan_epistemic).reason_code == "UNCERTAINTY_TOO_HIGH"
