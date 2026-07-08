"""Tensor shape tests (SPEC.md §27). Encoders require torch; skipped if torch is absent."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_temporal_encoder_output_shape() -> None:
    from helion_risk_world.encoders.temporal_encoder import TemporalEncoder

    enc = TemporalEncoder(n_features=8, latent_dim=128, layers=2)
    out = enc.forward(torch.zeros(4, 6, 96, 8))  # [B, A, L, F]
    assert out.shape == (4, 128)


def test_temporal_encoder_rejects_wrong_feature_dim() -> None:
    from helion_risk_world.encoders.temporal_encoder import TemporalEncoder

    enc = TemporalEncoder(n_features=8, latent_dim=32, layers=1)
    with pytest.raises(ValueError):
        enc.forward(torch.zeros(2, 3, 10, 9))  # F=9 != 8


def test_fusion_encoder_temporal_only() -> None:
    from helion_risk_world.encoders.fusion_encoder import FusionEncoder

    z = FusionEncoder(latent_dim=16).forward(torch.zeros(5, 16))
    assert z.shape == (5, 16)
