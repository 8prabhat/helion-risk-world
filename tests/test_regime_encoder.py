"""Regime/event encoder + featuriser + four-plane fusion integration (SPEC.md §16, §27)."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from helion_risk_world.data.regime_builder import REGIME_CONTEXT_FEATURES, featurize_regime
from helion_risk_world.schemas.market_schema import EventContext, EventType, RegimeContext

torch = pytest.importorskip("torch")

from helion_risk_world.config.model_config import ModelConfig  # noqa: E402
from helion_risk_world.config.training_config import TrainingConfig  # noqa: E402
from helion_risk_world.encoders.regime_encoder import RegimeEncoder  # noqa: E402
from helion_risk_world.losses.composite_loss import ForecasterLoss  # noqa: E402
from helion_risk_world.model import HRWForecaster  # noqa: E402
from helion_risk_world.training.trainer import ForecastBatch, HRWTrainer  # noqa: E402

TS = datetime(2026, 6, 25, 10, 0)
A, L, FEAT = 2, 12, 9
K = len(REGIME_CONTEXT_FEATURES)


def _ctx(vix: float = 14.0, event: EventType = EventType.NONE, blackout: bool = False):
    reg = RegimeContext(symbol="BANKNIFTY", ts=TS, available_at=TS, vix=vix, vix_pct=0.4,
                        atm_iv=0.18, iv_skew=0.01)
    evt = EventContext(symbol="BANKNIFTY", ts=TS, available_at=TS, expiry_flag=False,
                       event_day_flag=event is not EventType.NONE, blackout_active=blackout,
                       event_type=event, fii_dii_net=1200.0, usdinr=83.0, crude=80.0)
    return reg, evt


def test_featurize_shape_and_event_onehot() -> None:
    vec = featurize_regime(*_ctx(event=EventType.RBI))
    assert vec.shape == (K,) and vec.dtype == np.float32
    # exactly one event-type one-hot is set (RBI), and it is in the tail block.
    onehot = vec[len(REGIME_CONTEXT_FEATURES) - len(EventType):]
    assert onehot.sum() == pytest.approx(1.0)
    assert onehot[list(EventType).index(EventType.RBI)] == 1.0


def test_featurize_regime_missing_mask_flags_when_all_four_unavailable() -> None:
    """Review Idea #5: regime_missing_mask should be 1.0 only when ATM IV, IV
    skew, PCR, and basis are ALL unavailable — not conflated with the 0.0 the
    numeric features themselves fall back to, so the model can tell "no signal"
    apart from "genuinely neutral/zero"."""
    mask_idx = REGIME_CONTEXT_FEATURES.index("regime_missing_mask")

    reg_missing = RegimeContext(
        symbol="BANKNIFTY", ts=TS, available_at=TS, vix=14.0, vix_pct=0.4,
        atm_iv=None, iv_skew=None,
    )
    evt_missing = EventContext(
        symbol="BANKNIFTY", ts=TS, available_at=TS, expiry_flag=False,
        event_day_flag=False, blackout_active=False, event_type=EventType.NONE,
        fii_dii_net=1200.0, usdinr=83.0, crude=80.0, pc_oi_ratio=None, basis_daily=None,
    )
    vec_missing = featurize_regime(reg_missing, evt_missing)
    assert vec_missing[mask_idx] == pytest.approx(1.0)

    # Present by default via _ctx() (atm_iv/iv_skew set) -> mask must be 0.0.
    vec_present = featurize_regime(*_ctx())
    assert vec_present[mask_idx] == pytest.approx(0.0)

    # Only ONE of the four present -> still 0.0 (not "all missing").
    reg_partial = RegimeContext(
        symbol="BANKNIFTY", ts=TS, available_at=TS, vix=14.0, vix_pct=0.4,
        atm_iv=0.15, iv_skew=None,
    )
    vec_partial = featurize_regime(reg_partial, evt_missing)
    assert vec_partial[mask_idx] == pytest.approx(0.0)


def test_featurize_regime_includes_stabilized_macro_slots() -> None:
    """Feature/label overhaul Phase 0/2: fii_dii_net_z/pc_oi_ratio_z/usdinr_ret_5d/
    crude_ret_5d/usdinr_vol/crude_vol slots are populated from EventContext, not the
    old fixed-divisor-rescaled raw fields."""
    evt = EventContext(
        symbol="BANKNIFTY", ts=TS, available_at=TS, expiry_flag=False,
        event_day_flag=False, blackout_active=False, event_type=EventType.NONE,
        fii_dii_net_z=1.5, pc_oi_ratio_z=-0.75, usdinr_ret_5d=0.01, crude_ret_5d=-0.02,
        usdinr_vol=0.008, crude_vol=0.015,
    )
    reg = RegimeContext(symbol="BANKNIFTY", ts=TS, available_at=TS, vix=14.0, vix_pct=0.4)
    vec = featurize_regime(reg, evt)
    assert vec[REGIME_CONTEXT_FEATURES.index("fii_dii_net_z")] == pytest.approx(1.5)
    assert vec[REGIME_CONTEXT_FEATURES.index("pc_oi_ratio_z")] == pytest.approx(-0.75)
    assert vec[REGIME_CONTEXT_FEATURES.index("usdinr_ret_5d")] == pytest.approx(0.01)
    assert vec[REGIME_CONTEXT_FEATURES.index("crude_ret_5d")] == pytest.approx(-0.02)
    assert vec[REGIME_CONTEXT_FEATURES.index("usdinr_vol")] == pytest.approx(0.008)
    assert vec[REGIME_CONTEXT_FEATURES.index("crude_vol")] == pytest.approx(0.015)


def test_encoder_shape_and_validation() -> None:
    enc = RegimeEncoder(K, latent_dim=16)
    assert enc(torch.randn(4, K)).shape == (4, 16)
    with pytest.raises(ValueError):
        enc(torch.randn(4, K + 1))


def _model() -> HRWForecaster:
    return HRWForecaster(n_features=FEAT, cfg=ModelConfig(latent_dim=16, temporal_layers=1,
                                                         dropout=0.0))


def test_forecaster_uses_regime_when_provided() -> None:
    torch.manual_seed(0)
    model = _model().eval()
    feats = torch.randn(1, A, L, FEAT)
    reg = torch.tensor(featurize_regime(*_ctx())).unsqueeze(0)
    with torch.no_grad():
        z_no = model(feats)["z"]
        z_reg = model(feats, None, reg)["z"]
    assert not torch.allclose(z_no, z_reg)  # the regime context changes the latent state


def test_training_with_regime_trains_the_regime_encoder() -> None:
    torch.manual_seed(0)
    bsz = 8
    feats = torch.randn(bsz, A, L, FEAT)
    ret = torch.rand(bsz) * 0.04
    direction = torch.randint(0, 3, (bsz,))
    reg = torch.tensor(featurize_regime(*_ctx())).unsqueeze(0).repeat(bsz, 1)
    cfg = TrainingConfig(device="cpu", lr=1e-2, max_epochs=20, embargo_bars=12)

    m_no = _model()
    w_no = m_no.regime_encoder.mlp[0].weight.detach().clone()
    HRWTrainer(m_no, ForecasterLoss(), cfg).fit([ForecastBatch(feats, ret, direction)])
    assert torch.allclose(w_no, m_no.regime_encoder.mlp[0].weight)  # inert without regime input

    m_reg = _model()
    w_reg = m_reg.regime_encoder.mlp[0].weight.detach().clone()
    batch = ForecastBatch(feats, ret, direction, regime_context=reg)
    HRWTrainer(m_reg, ForecasterLoss(), cfg).fit([batch])
    assert not torch.allclose(w_reg, m_reg.regime_encoder.mlp[0].weight)  # trained with regime
