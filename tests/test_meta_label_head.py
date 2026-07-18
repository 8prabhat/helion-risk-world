"""MetaLabelHead: model wiring, causal primary-side computation, and the composite
loss's NaN-masked BCE term (2026-07-18)."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from helion_risk_world.config.model_config import LossWeights, ModelConfig  # noqa: E402
from helion_risk_world.heads.meta_label_head import (  # noqa: E402
    MetaLabelHead,
    primary_side_from_candle_features,
)
from helion_risk_world.losses.composite_loss import ForecasterLoss  # noqa: E402
from helion_risk_world.model import HRWForecaster  # noqa: E402

B, A, L, F = 8, 2, 24, 9


def _model(latent: int = 32) -> HRWForecaster:
    cfg = ModelConfig(size="small", latent_dim=latent, temporal_layers=1, dropout=0.0)
    return HRWForecaster(n_features=F, cfg=cfg, n_quantiles=5, meta_label_lookback=12)


def test_meta_label_head_output_shape() -> None:
    head = MetaLabelHead(latent_dim=16)
    z = torch.randn(5, 16)
    side = torch.tensor([1.0, -1.0, 0.0, 1.0, -1.0])
    out = head(z, side)
    assert out.shape == (5,)


def test_forecaster_forward_emits_meta_label_and_primary_side() -> None:
    out = _model()(torch.randn(B, A, L, F))
    assert out["meta_label_logit"].shape == (B,)
    assert out["primary_side"].shape == (B,)
    assert set(out["primary_side"].unique().tolist()).issubset({-1.0, 0.0, 1.0})


def test_forecaster_forward_respects_explicit_primary_side() -> None:
    """Passing primary_side explicitly (as the trainer does, from the label file) must
    be echoed back unchanged, not recomputed from features."""
    model = _model()
    features = torch.randn(B, A, L, F)
    explicit_side = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    out = model(features, primary_side=explicit_side)
    assert torch.equal(out["primary_side"], explicit_side)


def test_primary_side_from_candle_features_matches_close_based_signal() -> None:
    """The model's torch-native primary-side computation must reproduce
    labeling/meta_labels.py::primary_side_from_close on the same underlying price path
    -- verified via reconstructing a log_return channel from a known close series."""
    close = torch.tensor([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 108.0, 109.0, 111.0, 112.0, 113.0])
    log_returns = torch.log(close[1:] / close[:-1])  # 11 values -> spans 12 close points
    features = torch.zeros(1, 1, 11, 1)
    features[0, 0, :, 0] = log_returns
    side = primary_side_from_candle_features(
        features, primary_asset_idx=0, log_return_channel=0, lookback=12
    )
    assert side.item() == 1.0  # sustained uptrend

    close_down = torch.tensor([113.0, 112.0, 111.0, 109.0, 108.0, 106.0, 105.0, 104.0, 103.0, 102.0, 101.0, 100.0])
    log_returns_down = torch.log(close_down[1:] / close_down[:-1])
    features_down = torch.zeros(1, 1, 11, 1)
    features_down[0, 0, :, 0] = log_returns_down
    side_down = primary_side_from_candle_features(
        features_down, primary_asset_idx=0, log_return_channel=0, lookback=12
    )
    assert side_down.item() == -1.0


def test_primary_side_is_gradient_free() -> None:
    features = torch.randn(4, 1, 12, 1, requires_grad=True)
    side = primary_side_from_candle_features(features, lookback=12)
    assert side.requires_grad is False


def test_composite_loss_meta_label_term_masks_nan_rows() -> None:
    """Rows with meta_label=NaN (primary_side==0, no trade proposed) must contribute
    exactly zero to the meta-label loss term -- not a NaN that poisons the whole loss."""
    model = _model()
    loss_fn = ForecasterLoss(weights=LossWeights(
        return_=0.0, direction=0.0, volatility=0.0, mae=0.0, mfe=0.0, barrier=0.0,
        regime=0.0, calibration=0.0, uncertainty=0.0, ood=0.0, repr_var=0.0, repr_cov=0.0,
        meta_label=1.0,
    ))
    features = torch.randn(B, A, L, F)
    prediction = model(features)
    target = {
        "forward_return": torch.zeros(B),
        "meta_label": torch.tensor([1.0, 0.0, float("nan"), 1.0, float("nan"), 0.0, 1.0, 0.0]),
    }
    loss = loss_fn(prediction, target)
    assert math.isfinite(float(loss.detach()))
    assert "meta_label" in loss_fn.last_components
    assert math.isfinite(loss_fn.last_components["meta_label"])


def test_composite_loss_meta_label_term_all_nan_batch_is_safe() -> None:
    """Degenerate case: every row in the batch has primary_side==0. Must not NaN/blow up."""
    model = _model()
    loss_fn = ForecasterLoss(weights=LossWeights(
        return_=0.0, direction=0.0, volatility=0.0, mae=0.0, mfe=0.0, barrier=0.0,
        regime=0.0, calibration=0.0, uncertainty=0.0, ood=0.0, repr_var=0.0, repr_cov=0.0,
        meta_label=1.0,
    ))
    features = torch.randn(B, A, L, F)
    prediction = model(features)
    target = {
        "forward_return": torch.zeros(B),
        "meta_label": torch.full((B,), float("nan")),
    }
    loss = loss_fn(prediction, target)
    assert math.isfinite(float(loss.detach()))
    assert loss_fn.last_components["meta_label"] == pytest.approx(0.0)


def test_composite_loss_inert_when_meta_label_absent_from_target() -> None:
    model = _model()
    loss_fn = ForecasterLoss()
    features = torch.randn(B, A, L, F)
    prediction = model(features)
    target = {"forward_return": torch.zeros(B)}
    loss_fn(prediction, target)
    assert "meta_label" not in loss_fn.last_components
