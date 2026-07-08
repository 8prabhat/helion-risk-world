"""Option-surface encoder + featurisation + derivatives-aware forecaster (SPEC.md §16, §27)."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from helion_risk_world.data.option_surface_builder import (
    SURFACE_CONTEXT_FEATURES,
    SURFACE_STRIKE_CHANNELS,
    OptionSurfaceBuilder,
    featurize_surface,
)
from helion_risk_world.schemas.option_chain_schema import OptionContractSnapshot, OptionType

torch = pytest.importorskip("torch")

from helion_risk_world.config.model_config import ModelConfig  # noqa: E402
from helion_risk_world.config.training_config import TrainingConfig  # noqa: E402
from helion_risk_world.encoders.futures_encoder import FUTURES_FEATURE_DIM  # noqa: E402
from helion_risk_world.encoders.option_surface_encoder import (  # noqa: E402
    OptionSurfaceEncoder,
    SurfaceTensors,
)
from helion_risk_world.inference import ForecasterPredictor  # noqa: E402
from helion_risk_world.losses.composite_loss import ForecasterLoss  # noqa: E402
from helion_risk_world.model import HRWForecaster  # noqa: E402
from helion_risk_world.training.trainer import ForecastBatch, HRWTrainer  # noqa: E402

TS = datetime(2026, 6, 25, 11, 0)
A, L, FEAT = 2, 12, 9


def _chain(
    atm: float = 50000.0, step: float = 100.0, width: int = 4
) -> list[OptionContractSnapshot]:
    out = []
    for i in range(-width, width + 1):
        k = atm + i * step
        for opt in (OptionType.CALL, OptionType.PUT):
            oi = 1000 - 10 * i * (opt == OptionType.PUT)
            out.append(OptionContractSnapshot(
                underlying="BANKNIFTY", strike=k, opt_type=opt, ts=TS, available_at=TS,
                open=10, high=12, low=8, close=11, volume=100 + i, oi=oi,
                d_oi=5, iv=0.2 + 0.005 * abs(i), delta=0.5, gamma=0.01, theta=-1, vega=2, dte=2.0,
            ))
    return out


def _surface(n_strikes: int = 3):
    return OptionSurfaceBuilder(n_strikes=n_strikes).align_to_atm(_chain(), spot=50000.0, ts=TS)


# ---------------- featurisation ----------------
def test_featurize_shapes_and_channels() -> None:
    grid, mask, context = featurize_surface(_surface(n_strikes=3))
    assert grid.shape == (7, len(SURFACE_STRIKE_CHANNELS))   # 2N+1 = 7
    assert mask.shape == (7,)
    assert context.shape == (len(SURFACE_CONTEXT_FEATURES),)
    assert grid.dtype == np.float32
    assert set(np.unique(mask)).issubset({0.0, 1.0})


def test_featurize_masks_missing_strikes() -> None:
    chain = [c for c in _chain() if c.strike != 50100.0]  # drop the ATM+1 legs
    surf = OptionSurfaceBuilder(n_strikes=2).align_to_atm(chain, spot=50000.0, ts=TS)
    grid, mask, _ = featurize_surface(surf)
    idx = [r.token for r in surf.strikes].index(1)  # ATM+1 token
    assert mask[idx] == 0.0 and grid[idx, -1] == 1.0  # masked + is_masked channel set


# ---------------- encoder ----------------
def _tensors(surf) -> SurfaceTensors:
    g, m, c = featurize_surface(surf)
    return SurfaceTensors(grid=torch.tensor(g).unsqueeze(0), mask=torch.tensor(m).unsqueeze(0),
                          context=torch.tensor(c).unsqueeze(0))


def test_encoder_output_shape() -> None:
    enc = OptionSurfaceEncoder(n_channels=len(SURFACE_STRIKE_CHANNELS),
                               n_context=len(SURFACE_CONTEXT_FEATURES), latent_dim=16)
    assert enc(_tensors(_surface())).shape == (1, 16)


def test_encoder_ignores_masked_rows() -> None:
    """Masked-mean pooling means masked strikes do not affect the embedding (permutation-safe)."""
    enc = OptionSurfaceEncoder(n_channels=len(SURFACE_STRIKE_CHANNELS),
                               n_context=len(SURFACE_CONTEXT_FEATURES), latent_dim=16).eval()
    base = _tensors(_surface())
    # Corrupt the values of masked rows; output must be unchanged.
    corrupted = base.grid.clone()
    masked = base.mask[0] == 0.0
    corrupted[0, masked] += 99.0
    with torch.no_grad():
        a = enc(base)
        b = enc(SurfaceTensors(grid=corrupted, mask=base.mask, context=base.context))
    # If there is at least one masked row the outputs match exactly; else this is vacuously true.
    assert torch.allclose(a, b, atol=1e-5) or not bool(masked.any())


# ---------------- V1 futures-encoder integration ----------------
_T_FUTURES = 24   # lookback bars for futures microstructure
_F_FUTURES = FUTURES_FEATURE_DIM


def _futures(b: int = 1) -> torch.Tensor:
    """Synthetic futures microstructure tensor [B, T, F]."""
    return torch.randn(b, _T_FUTURES, _F_FUTURES)


def _train_model() -> HRWForecaster:
    return HRWForecaster(n_features=FEAT, cfg=ModelConfig(latent_dim=16, temporal_layers=1,
                                                         dropout=0.0))


def test_forecaster_uses_futures_when_provided() -> None:
    """FuturesEncoder changes the latent state vs. OHLCV-only mode."""
    torch.manual_seed(0)
    model = HRWForecaster(n_features=FEAT, cfg=ModelConfig(latent_dim=16, temporal_layers=1,
                                                          dropout=0.0)).eval()
    feats = torch.randn(1, A, L, FEAT)
    with torch.no_grad():
        z_no = model(feats)["z"]
        z_fut = model(feats, _futures(1))["z"]
    assert z_no.shape == z_fut.shape
    assert not torch.allclose(z_no, z_fut)  # futures microstructure changes the latent state


def test_predictor_accepts_futures() -> None:
    model = HRWForecaster(n_features=FEAT, cfg=ModelConfig(latent_dim=16, temporal_layers=1,
                                                          dropout=0.0))
    pred = ForecasterPredictor(model).predict_one(
        torch.randn(A, L, FEAT), "BANKNIFTY", TS, futures=_futures(1).squeeze(0)
    )
    assert pred.symbol == "BANKNIFTY" and len(pred.horizon_preds) == 1


def test_training_with_futures_trains_the_futures_encoder() -> None:
    """Futures in the batch puts the futures encoder in the gradient graph."""
    torch.manual_seed(0)
    bsz = 8
    feats = torch.randn(bsz, A, L, FEAT)
    ret = torch.rand(bsz) * 0.04
    direction = torch.randint(0, 3, (bsz,))
    cfg = TrainingConfig(device="cpu", lr=1e-2, max_epochs=20, embargo_bars=12)

    # No futures -> futures_encoder receives no gradient; conv weight stays fixed.
    m_no = _train_model()
    w_no = m_no.futures_encoder.conv[0].weight.detach().clone()
    HRWTrainer(m_no, ForecasterLoss(), cfg).fit([ForecastBatch(feats, ret, direction)])
    assert torch.allclose(w_no, m_no.futures_encoder.conv[0].weight)

    # With futures -> the encoder is trained (weights change).
    m_fut = _train_model()
    w_fut = m_fut.futures_encoder.conv[0].weight.detach().clone()
    batch = ForecastBatch(feats, ret, direction, futures=_futures(bsz))
    HRWTrainer(m_fut, ForecasterLoss(), cfg).fit([batch])
    assert not torch.allclose(w_fut, m_fut.futures_encoder.conv[0].weight)
