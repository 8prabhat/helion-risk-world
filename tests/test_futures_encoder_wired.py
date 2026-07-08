"""Verify FuturesEncoder is wired into HRWForecaster and HRWWorldModel (SPEC.md §9.2, §13)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from helion_risk_world.config.model_config import ModelConfig  # noqa: E402
from helion_risk_world.encoders.futures_encoder import FUTURES_FEATURE_DIM  # noqa: E402
from helion_risk_world.model import HRWForecaster, HRWWorldModel  # noqa: E402

B, A, L, FEAT = 2, 1, 24, 7
T_FUT = 24   # futures lookback bars


def _fut(b: int = B) -> "torch.Tensor":
    return torch.randn(b, T_FUT, FUTURES_FEATURE_DIM)


def test_forecaster_with_futures_output_shape() -> None:
    model = HRWForecaster(n_features=FEAT, cfg=ModelConfig(latent_dim=32, temporal_layers=1)).eval()
    with torch.no_grad():
        out = model(torch.randn(B, A, L, FEAT), futures=_fut())
    assert out["return_quantiles"].shape == (B, 5)
    assert out["z"].shape == (B, 32)


def test_forecaster_without_futures_backward_compat() -> None:
    model = HRWForecaster(n_features=FEAT, cfg=ModelConfig(latent_dim=32, temporal_layers=1)).eval()
    with torch.no_grad():
        out = model(torch.randn(B, A, L, FEAT))   # futures=None (default)
    assert out["return_quantiles"].shape == (B, 5)


def test_futures_changes_latent() -> None:
    """FuturesEncoder output must influence z_t (fusion uses the slot)."""
    torch.manual_seed(42)
    model = HRWForecaster(n_features=FEAT, cfg=ModelConfig(latent_dim=32, temporal_layers=1)).eval()
    feats = torch.randn(B, A, L, FEAT)
    with torch.no_grad():
        z_no = model(feats)["z"]
        z_ft = model(feats, _fut())["z"]
    assert not torch.allclose(z_no, z_ft, atol=1e-5)


def test_world_model_with_futures_output_shape() -> None:
    model = HRWWorldModel(n_features=FEAT, cfg=ModelConfig(latent_dim=32, temporal_layers=1),
                          horizons=(1, 3), n_samples=4).eval()
    with torch.no_grad():
        out = model(torch.randn(B, A, L, FEAT), futures=_fut())
    assert out["return_quantiles"].shape == (B, 2, 5)   # [B, |H|, Q]
    assert out["epistemic"].shape == (B, 2)


def test_forecaster_futures_encoder_trains() -> None:
    """Gradient flows into futures_encoder when futures tensor is supplied."""
    model = HRWForecaster(n_features=FEAT, cfg=ModelConfig(latent_dim=32, temporal_layers=1))
    feats = torch.randn(B, A, L, FEAT)
    fut = _fut()
    out = model(feats, futures=fut)
    out["return_quantiles"].sum().backward()
    assert model.futures_encoder.conv[0].weight.grad is not None
