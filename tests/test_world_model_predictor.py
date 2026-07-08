"""HRWWorldModel + WorldModelPredictor: RSSM → multi-horizon prediction (SPEC.md §13, §18)."""

from __future__ import annotations

from datetime import datetime

import pytest

torch = pytest.importorskip("torch")

from helion_risk_world.config.model_config import ModelConfig  # noqa: E402
from helion_risk_world.encoders.futures_encoder import FUTURES_FEATURE_DIM  # noqa: E402
from helion_risk_world.inference import WorldModelPredictor  # noqa: E402
from helion_risk_world.model import HRWWorldModel  # noqa: E402
from helion_risk_world.planner.mpc_planner import MPCPlanner  # noqa: E402
from helion_risk_world.schemas import ActionType, PortfolioState, RiskProfile  # noqa: E402
from helion_risk_world.schemas.prediction_schema import QUANTILE_LEVELS  # noqa: E402

A, L, FEAT = 2, 12, 9
TS = datetime(2026, 6, 25, 10, 0)
RISK = RiskProfile(
    name="balanced", max_risk_per_trade=0.01, max_daily_loss=0.02, max_weekly_loss=0.05,
    max_drawdown=0.10, max_exposure=1.0, max_trades_per_day=100, consecutive_loss_cooldown=99,
    cvar_alpha=0.05, n_paths=256,
)


def _model(horizons: tuple[int, ...] = (1, 3, 6)) -> HRWWorldModel:
    return HRWWorldModel(n_features=FEAT, cfg=ModelConfig(latent_dim=16, temporal_layers=1,
                                                          dropout=0.0), horizons=horizons,
                         n_samples=12)


def _account() -> PortfolioState:
    cap = 500_000.0
    return PortfolioState(ts=TS, capital0=cap, capital=cap, cash=cap, free_margin=cap)


_T_FUTURES, _F_FUTURES = 24, FUTURES_FEATURE_DIM   # lookback bars × futures feature dim


def test_world_model_forward_shapes() -> None:
    out = _model()(torch.randn(4, A, L, FEAT))
    assert out["return_quantiles"].shape == (4, 3, 5)   # [B, |H|, Q]
    # direction_logits removed in new spec
    assert "direction_logits" not in out
    assert out["epistemic"].shape == (4, 3)             # [B, |H|] from RSSM ensemble
    assert out["barrier_probs"].shape == (4, 3)         # [B, 3] at H=max
    assert out["mae"].shape == (4, 3)
    assert out["mfe"].shape == (4, 3)
    assert out["ood_score"].shape == (4, 1)


def test_predictor_emits_multi_horizon_prediction() -> None:
    torch.manual_seed(0)
    wmp = WorldModelPredictor(_model((1, 3, 6)))
    pred = wmp.predict_one(torch.randn(A, L, FEAT), "BANKNIFTY", TS)
    assert [hp.horizon_bars for hp in pred.horizon_preds] == [1, 3, 6]  # sorted
    for hp in pred.horizon_preds:
        assert set(hp.return_quantiles) == set(QUANTILE_LEVELS)
        vals = [hp.return_quantiles[q] for q in sorted(hp.return_quantiles)]
        assert all(b >= a - 1e-6 for a, b in zip(vals, vals[1:], strict=False))  # non-crossing
    # Barrier at top level (not per horizon)
    total = pred.barrier.stop + pred.barrier.target + pred.barrier.timeout
    assert total == pytest.approx(1.0, abs=1e-4)
    assert pred.mae >= 0
    assert pred.epistemic >= 0 and pred.aleatoric >= 0


def test_planner_consumes_world_model_prediction() -> None:
    pred = WorldModelPredictor(_model()).predict_one(torch.randn(A, L, FEAT), "BANKNIFTY", TS)
    decision = MPCPlanner.default().plan(pred, _account(), RISK)
    assert decision.final_action.action_type in set(ActionType)
    assert decision.candidates


def test_predictor_accepts_futures_and_fits_ood() -> None:
    model = _model()
    model.fit_ood(torch.randn(50, A, L, FEAT))
    futures = torch.randn(_T_FUTURES, _F_FUTURES)   # [T, F] for predict_one
    pred = WorldModelPredictor(model).predict_one(
        torch.randn(A, L, FEAT), "BANKNIFTY", TS, futures=futures
    )
    assert 0.0 <= pred.ood_score and len(pred.horizon_preds) == 3


def _spy_on_forward(model: HRWWorldModel) -> list:
    """Record the ``state`` kwarg passed to each forward() call while delegating
    to the real implementation, so tests can assert on wiring without depending
    on the RSSM's stochastic outputs."""
    recorded: list = []
    original_forward = model.forward

    def spy_forward(*args, **kwargs):
        recorded.append(kwargs.get("state"))
        return original_forward(*args, **kwargs)

    model.forward = spy_forward  # type: ignore[method-assign]
    return recorded


def test_predict_one_threads_state_across_calls_same_day() -> None:
    """Review finding H1: successive calls on the same trading day must reuse the
    previous call's RSSM state instead of resetting to zero every time."""
    model = _model()
    recorded = _spy_on_forward(model)
    wmp = WorldModelPredictor(model)
    wmp.predict_one(torch.randn(A, L, FEAT), "BANKNIFTY", TS)
    wmp.predict_one(torch.randn(A, L, FEAT), "BANKNIFTY", TS.replace(minute=5))
    assert recorded[0] is None
    assert recorded[1] is not None


def test_predict_one_resets_state_on_new_trading_day() -> None:
    model = _model()
    recorded = _spy_on_forward(model)
    wmp = WorldModelPredictor(model)
    wmp.predict_one(torch.randn(A, L, FEAT), "BANKNIFTY", TS)
    next_day = TS.replace(day=TS.day + 1)
    wmp.predict_one(torch.randn(A, L, FEAT), "BANKNIFTY", next_day)
    assert recorded[0] is None
    assert recorded[1] is None  # crossing a day boundary resets, not carried


def test_persist_state_false_always_resets() -> None:
    model = _model()
    recorded = _spy_on_forward(model)
    wmp = WorldModelPredictor(model, persist_state=False)
    wmp.predict_one(torch.randn(A, L, FEAT), "BANKNIFTY", TS)
    wmp.predict_one(torch.randn(A, L, FEAT), "BANKNIFTY", TS.replace(minute=5))
    assert recorded == [None, None]


def test_reset_state_clears_persisted_state() -> None:
    wmp = WorldModelPredictor(_model())
    assert wmp.state is None
    wmp.predict_one(torch.randn(A, L, FEAT), "BANKNIFTY", TS)
    assert wmp.state is not None
    wmp.reset_state()
    assert wmp.state is None


def test_normalize_ood_unfitted_fallback_is_batch_independent() -> None:
    """Review finding M1: before fit_ood() is called, the unfitted-fallback OOD
    normalization used to divide by the CURRENT BATCH's own mean |raw| score,
    making a sample's OOD reading depend on whatever else happened to be in the
    same batch (a wild outlier batchmate would dampen everyone else's score by
    inflating the shared denominator). The fallback must be a fixed,
    architecture-derived scale so the same raw score always normalizes the same
    way regardless of batch composition."""
    model = _model()
    raw_alone = torch.tensor([2.0])
    raw_with_outlier = torch.tensor([2.0, 500.0])  # same first entry, wild second entry

    out_alone = model._normalize_ood(raw_alone)
    out_with_outlier = model._normalize_ood(raw_with_outlier)

    assert out_alone[0].item() == pytest.approx(out_with_outlier[0].item(), abs=1e-6)


def test_world_model_predictor_deterministic_gives_reproducible_predictions() -> None:
    """Review finding M3: WorldModelPredictor(deterministic=True) must give
    identical predictions across repeated calls with the same input, unlike the
    default stochastic behavior."""
    model = _model()
    features = torch.randn(A, L, FEAT)

    wmp = WorldModelPredictor(model, deterministic=True, persist_state=False)
    pred_a = wmp.predict_one(features, "BANKNIFTY", TS)
    pred_b = wmp.predict_one(features, "BANKNIFTY", TS)
    assert pred_a.horizon_preds[0].return_quantiles == pytest.approx(
        pred_b.horizon_preds[0].return_quantiles
    )
    assert pred_a.sigma_H == pytest.approx(pred_b.sigma_H)
