"""Cross-asset relation encoder (SPEC.md §16, §27)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from helion_risk_world.config.model_config import ModelConfig  # noqa: E402
from helion_risk_world.encoders.cross_asset_encoder import CrossAssetEncoder  # noqa: E402
from helion_risk_world.model import HRWForecaster  # noqa: E402

A, L, FEAT = 4, 10, 8


def test_output_shape() -> None:
    enc = CrossAssetEncoder(n_features=FEAT, latent_dim=16, n_heads=4)
    assert enc(torch.randn(3, A, L, FEAT)).shape == (3, 16)


def test_rejects_wrong_feature_dim() -> None:
    with pytest.raises(ValueError):
        CrossAssetEncoder(n_features=FEAT, latent_dim=16).forward(torch.randn(2, A, L, FEAT + 1))


def test_invariant_to_asset_order() -> None:
    """Attention + symmetric mean-pool => the embedding does not depend on asset ordering."""
    enc = CrossAssetEncoder(n_features=FEAT, latent_dim=16, n_heads=4).eval()
    x = torch.randn(2, A, L, FEAT)
    perm = x[:, torch.tensor([3, 1, 0, 2]), :, :]
    with torch.no_grad():
        assert torch.allclose(enc(x), enc(perm), atol=1e-5)


def test_depends_on_asset_content() -> None:
    enc = CrossAssetEncoder(n_features=FEAT, latent_dim=16, n_heads=4).eval()
    x = torch.randn(2, A, L, FEAT)
    y = x.clone()
    y[:, 0] += 5.0  # change one asset's data -> a different (concentrated) cross-asset picture
    with torch.no_grad():
        assert not torch.allclose(enc(x), enc(y))


def test_forecaster_integrates_cross_asset() -> None:
    """The forecaster now fuses a cross-asset embedding; forward stays valid and uses it."""
    model = HRWForecaster(n_features=FEAT, cfg=ModelConfig(latent_dim=16, temporal_layers=1,
                                                          dropout=0.0)).eval()
    out = model(torch.randn(2, A, L, FEAT))
    assert out["z"].shape == (2, 16)
    # The cross-asset encoder is in the graph -> its params receive gradients.
    model.train()
    model(torch.randn(2, A, L, FEAT))["z"].pow(2).mean().backward()
    grad = model.cross_asset.proj.weight.grad
    assert grad is not None and float(grad.abs().sum()) > 0
