"""Stage 2 — self-supervised future-latent pretraining (SPEC.md §20.2, §27)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from helion_risk_world.config.model_config import ModelConfig  # noqa: E402
from helion_risk_world.config.training_config import TrainingConfig  # noqa: E402
from helion_risk_world.data.regime_builder import REGIME_CONTEXT_FEATURES  # noqa: E402
from helion_risk_world.encoders.futures_encoder import FUTURES_FEATURE_DIM  # noqa: E402
from helion_risk_world.losses.latent_consistency_loss import LatentPredictionLoss  # noqa: E402
from helion_risk_world.model import HRWForecaster  # noqa: E402
from helion_risk_world.training.pretrain_market_state import (  # noqa: E402
    LatentPair,
    MarketStatePretrainer,
)

A, L, FEAT = 2, 12, 9


def test_latent_loss_penalises_collapse() -> None:
    loss = LatentPredictionLoss()
    collapsed = torch.zeros(16, 8)            # every embedding identical -> max variance penalty
    spread = torch.randn(16, 8) * 2.0
    loss(collapsed, collapsed)
    var_collapsed = loss.last_components["variance"]
    loss(spread, spread)
    var_spread = loss.last_components["variance"]
    assert var_collapsed > var_spread          # collapse is penalised, spread is not
    assert var_spread == pytest.approx(0.0, abs=1e-3)


def test_latent_loss_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        LatentPredictionLoss()(torch.randn(4, 8), torch.randn(4, 7))


def _model() -> HRWForecaster:
    return HRWForecaster(n_features=FEAT, cfg=ModelConfig(latent_dim=16, temporal_layers=1,
                                                         dropout=0.0))


def _pairs(n_batches: int = 4, b: int = 12) -> list[LatentPair]:
    torch.manual_seed(0)
    pairs = []
    for _ in range(n_batches):
        ctx = torch.randn(b, A, L, FEAT)
        # The future window is a temporal shift of the context -> a genuine, learnable dynamic.
        future = torch.roll(ctx, shifts=-1, dims=2) + 0.01 * torch.randn_like(ctx)
        pairs.append(LatentPair(context=ctx, future=future))
    return pairs


def test_pretraining_reduces_loss_without_collapse() -> None:
    pairs = _pairs()
    pre = MarketStatePretrainer(_model(), TrainingConfig(device="cpu", lr=3e-3, max_epochs=60,
                                                         embargo_bars=12))
    pre.fit(pairs)
    assert pre.history[-1] < pre.history[0]                 # the world-model objective improves
    assert pre.latent_collapse_std(pairs) > 0.1            # representation did NOT collapse


def test_pretrainer_requires_pairs() -> None:
    with pytest.raises(ValueError):
        MarketStatePretrainer(_model(), TrainingConfig(embargo_bars=12)).fit([])


def test_pretrainer_accepts_optional_futures_and_regime_inputs() -> None:
    ctx = torch.randn(8, A, L, FEAT)
    future = torch.roll(ctx, shifts=-1, dims=2)
    pair = LatentPair(
        context=ctx,
        future=future,
        context_futures=torch.randn(8, L, FUTURES_FEATURE_DIM),
        future_futures=torch.randn(8, L, FUTURES_FEATURE_DIM),
        context_regime=torch.randn(8, len(REGIME_CONTEXT_FEATURES)),
        future_regime=torch.randn(8, len(REGIME_CONTEXT_FEATURES)),
    )
    pre = MarketStatePretrainer(
        _model(),
        TrainingConfig(device="cpu", lr=3e-3, max_epochs=2, embargo_bars=12),
    )
    pre.fit([pair], epochs=1)
    assert pre.history
