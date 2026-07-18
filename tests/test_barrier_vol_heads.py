"""Volatility + barrier heads, and the bridge using them (SPEC.md §15, §17, §27).

Direction head is REMOVED from the spec (§15). Barrier probabilities are at the
top-level ModelPrediction.barrier, not per HorizonPrediction.
"""

from __future__ import annotations

from datetime import datetime

import pytest

torch = pytest.importorskip("torch")

from helion_risk_world.config.model_config import LossWeights, ModelConfig  # noqa: E402
from helion_risk_world.config.training_config import TrainingConfig  # noqa: E402
from helion_risk_world.data.option_surface_builder import (  # noqa: E402
    SURFACE_CONTEXT_FEATURES,
    SURFACE_STRIKE_CHANNELS,
)
from helion_risk_world.encoders.option_surface_encoder import SurfaceTensors  # noqa: E402
from helion_risk_world.heads.barrier_head import BARRIER_CLASSES, BarrierHead  # noqa: E402
from helion_risk_world.heads.direction_head import DirectionHead  # noqa: E402
from helion_risk_world.heads.excursion_barrier_head import ExcursionBarrierHead  # noqa: E402
from helion_risk_world.heads.excursion_head import ExcursionHead  # noqa: E402
from helion_risk_world.heads.touch_head import TouchHead  # noqa: E402
from helion_risk_world.heads.uncertainty_head import UncertaintyHead  # noqa: E402
from helion_risk_world.heads.volatility_head import VolatilityHead  # noqa: E402
from helion_risk_world.inference import ForecasterPredictor  # noqa: E402
from helion_risk_world.losses.composite_loss import ForecasterLoss  # noqa: E402
from helion_risk_world.model import HRWForecaster  # noqa: E402
from helion_risk_world.training.trainer import ForecastBatch, HRWTrainer  # noqa: E402

A, L, FEAT = 2, 12, 9
TS = datetime(2026, 6, 25, 10, 0)


def _model() -> HRWForecaster:
    return HRWForecaster(n_features=FEAT, cfg=ModelConfig(latent_dim=16, temporal_layers=1,
                                                         dropout=0.0))


def test_volatility_head_positive() -> None:
    out = VolatilityHead(latent_dim=16)(torch.randn(4, 16))
    assert out.shape == (4,) and bool((out >= 0).all())   # [B] not [B, 1]
    assert float(out.detach().mean()) == pytest.approx(0.001, abs=5e-4)


def test_excursion_head_positive() -> None:
    out = ExcursionHead(latent_dim=16)(torch.randn(4, 16))
    assert out.shape == (4,) and bool((out >= 0).all())
    assert float(out.detach().mean()) == pytest.approx(0.002, abs=5e-4)


def test_uncertainty_head_initial_scale_is_small() -> None:
    out = UncertaintyHead(latent_dim=16)(torch.randn(4, 16))
    assert out.shape == (4,) and bool((out >= 1e-3).all())
    assert float(out.detach().mean()) == pytest.approx(0.005, abs=1e-3)


def test_barrier_head_shape_and_classes() -> None:
    assert BARRIER_CLASSES == ("stop_first", "target_first", "neither")
    assert BarrierHead(latent_dim=16)(torch.randn(4, 16)).shape == (4, 3)


def test_barrier_head_accepts_auxiliary_context() -> None:
    head = BarrierHead(latent_dim=16, context_dim=6)
    z = torch.randn(4, 16)
    ctx = torch.randn(4, 6)
    assert head(z, context=ctx).shape == (4, 3)


def test_excursion_barrier_head_prefers_timeout_below_thresholds() -> None:
    head = ExcursionBarrierHead()
    logits = head(torch.tensor([[0.4, 0.7, 0.5]], dtype=torch.float32))
    assert int(logits.argmax(dim=-1)[0]) == 2


def test_excursion_barrier_head_volatility_ratio_is_connected_but_neutral_at_init() -> None:
    """Review finding M2: volatility_ratio used to be validated as a required
    input but never read in forward(). It must now actually influence the
    computation graph (gradients reach its weight column) while leaving the
    identity-mapped cold-start behavior for stop/target/timeout unchanged."""
    head = ExcursionBarrierHead()
    ratios_a = torch.tensor([[0.4, 0.7, 0.5]], dtype=torch.float32)
    ratios_b = torch.tensor([[0.4, 0.7, 5.0]], dtype=torch.float32)  # only vol_ratio differs

    # At initialization, volatility_ratio has zero effect (weight column is 0).
    assert torch.allclose(head(ratios_a), head(ratios_b))

    # But it IS wired into the graph: a gradient w.r.t. the 4th weight column exists.
    out = head(ratios_a)
    out.sum().backward()
    assert head.linear.weight.grad is not None
    assert float(head.linear.weight.grad[:, 3].abs().sum()) > 0.0


def test_forecaster_emits_volatility_and_barrier() -> None:
    out = _model()(torch.randn(3, A, L, FEAT))
    assert out["volatility"].shape == (3,)         # [B] since squeeze(-1) in VolatilityHead
    assert out["barrier_logits"].shape == (3, 3)


def test_bridge_uses_model_volatility_and_barrier() -> None:
    """Prediction's volatility + barrier probs come from the heads, not a derived fallback."""
    torch.manual_seed(0)
    model = _model().eval()
    feats = torch.randn(A, L, FEAT)
    with torch.no_grad():
        out = model(feats.unsqueeze(0))
    vol = float(out["volatility"][0])          # [B] → scalar
    bprobs = torch.softmax(out["barrier_logits"][0], dim=-1)

    pred = ForecasterPredictor(model).predict_one(feats, "BANKNIFTY", TS)
    hp = pred.horizon_preds[0]
    assert hp.volatility == pytest.approx(vol, abs=1e-5)
    # Barrier is at ModelPrediction level (management horizon)
    assert pred.barrier.stop == pytest.approx(float(bprobs[0]), abs=1e-5)
    assert pred.barrier.target == pytest.approx(float(bprobs[1]), abs=1e-5)


def test_training_with_vol_barrier_trains_those_heads() -> None:
    torch.manual_seed(0)
    bsz = 8
    feats = torch.randn(bsz, A, L, FEAT)
    ret = torch.rand(bsz) * 0.04
    direction = torch.randint(0, 3, (bsz,))
    realized_vol = torch.rand(bsz) * 0.02 + 0.01
    mae = torch.rand(bsz) * 0.02
    mfe = torch.rand(bsz) * 0.02
    barrier = torch.randint(0, 3, (bsz,))
    barrier_context = torch.tensor(
        [[0.01, -0.02, 0.02]] * bsz,
        dtype=torch.float32,
    )
    cfg = TrainingConfig(device="cpu", lr=1e-2, max_epochs=20, embargo_bars=12)

    m_no = _model()
    m_no.set_barrier_mode("derived")
    wv = m_no.volatility_head.linear.weight.detach().clone()
    wmae = m_no.mae_head.linear.weight.detach().clone()
    wmfe = m_no.mfe_head.linear.weight.detach().clone()
    wb = m_no.excursion_barrier_head.linear.weight.detach().clone()
    HRWTrainer(m_no, ForecasterLoss(), cfg).fit([ForecastBatch(feats, ret, direction)])
    assert torch.allclose(wv, m_no.volatility_head.linear.weight)   # inert without vol target
    assert torch.allclose(wmae, m_no.mae_head.linear.weight)
    assert torch.allclose(wmfe, m_no.mfe_head.linear.weight)
    assert torch.allclose(wb, m_no.excursion_barrier_head.linear.weight)

    m_yes = _model()
    m_yes.set_barrier_mode("derived")
    wv2 = m_yes.volatility_head.linear.weight.detach().clone()
    wmae2 = m_yes.mae_head.linear.weight.detach().clone()
    wmfe2 = m_yes.mfe_head.linear.weight.detach().clone()
    wb2 = m_yes.excursion_barrier_head.linear.weight.detach().clone()
    batch = ForecastBatch(
        feats,
        ret,
        direction,
        realized_vol=realized_vol,
        mae=mae,
        mfe=mfe,
        barrier=barrier,
        barrier_context=barrier_context,
    )
    HRWTrainer(m_yes, ForecasterLoss(), cfg).fit([batch])
    assert not torch.allclose(wv2, m_yes.volatility_head.linear.weight)
    assert not torch.allclose(wmae2, m_yes.mae_head.linear.weight)
    assert not torch.allclose(wmfe2, m_yes.mfe_head.linear.weight)
    assert not torch.allclose(wb2, m_yes.excursion_barrier_head.linear.weight)


def test_barrier_supervision_flows_into_excursion_heads() -> None:
    torch.manual_seed(0)
    bsz = 8
    feats = torch.randn(bsz, A, L, FEAT)
    ret = torch.zeros(bsz)
    direction = torch.zeros(bsz, dtype=torch.long)
    barrier = torch.randint(0, 3, (bsz,))
    barrier_context = torch.tensor(
        [[0.01, -0.02, 0.02]] * bsz,
        dtype=torch.float32,
    )
    cfg = TrainingConfig(device="cpu", lr=1e-2, max_epochs=20, embargo_bars=12, weight_decay=0.0)

    model = _model()
    model.set_barrier_mode("derived")
    initial_mae = model.mae_head.linear.weight.detach().clone()
    initial_mfe = model.mfe_head.linear.weight.detach().clone()
    batch = ForecastBatch(
        feats,
        ret,
        direction,
        barrier=barrier,
        barrier_context=barrier_context,
    )
    loss = ForecasterLoss(
        weights=LossWeights(
            return_=0.0,
            direction=0.0,
            volatility=0.0,
            mae=0.0,
            mfe=0.0,
            barrier=1.0,
            regime=0.0,
            calibration=0.0,
            uncertainty=0.0,
            ood=0.0,
        ),
    )
    HRWTrainer(model, loss, cfg).fit([batch])

    assert not torch.allclose(initial_mae, model.mae_head.linear.weight)
    assert not torch.allclose(initial_mfe, model.mfe_head.linear.weight)


def test_barrier_weight_zero_keeps_derived_barrier_path_inert() -> None:
    torch.manual_seed(0)
    bsz = 8
    feats = torch.randn(bsz, A, L, FEAT)
    ret = torch.rand(bsz) * 0.04
    direction = torch.randint(0, 3, (bsz,))
    barrier = torch.randint(0, 3, (bsz,))
    barrier_context = torch.tensor(
        [[0.01, -0.02, 0.02]] * bsz,
        dtype=torch.float32,
    )
    cfg = TrainingConfig(device="cpu", lr=1e-2, max_epochs=20, embargo_bars=12, weight_decay=0.0)

    model = _model()
    model.set_barrier_mode("derived")
    initial = model.excursion_barrier_head.linear.weight.detach().clone()
    batch = ForecastBatch(
        feats,
        ret,
        direction,
        barrier=barrier,
        barrier_weight=torch.zeros(bsz),
        barrier_context=barrier_context,
    )
    HRWTrainer(model, ForecasterLoss(), cfg).fit([batch])

    assert torch.allclose(initial, model.excursion_barrier_head.linear.weight)


# ---------------- decomposed barrier architecture (2026-07-13) ----------------
def _surface_tensors(bsz: int, n_strikes: int = 5) -> SurfaceTensors:
    s = 2 * n_strikes + 1
    return SurfaceTensors(
        grid=torch.randn(bsz, s, len(SURFACE_STRIKE_CHANNELS)),
        mask=torch.ones(bsz, s),
        context=torch.randn(bsz, len(SURFACE_CONTEXT_FEATURES)),
    )


def test_touch_head_shape() -> None:
    assert TouchHead(latent_dim=16)(torch.randn(4, 16)).shape == (4,)


def test_direction_head_shape_and_skip_connection_matters() -> None:
    head = DirectionHead(latent_dim=16, surface_dim=16)
    z = torch.randn(4, 16)
    surface_a = torch.randn(4, 16)
    surface_b = torch.randn(4, 16)
    out_a = head(z, surface_a)
    out_b = head(z, surface_b)
    assert out_a.shape == (4,)
    # Different surface embeddings (same z) must change the output -- otherwise the skip
    # connection isn't actually wired into the computation graph.
    assert not torch.allclose(out_a, out_b)


def test_forecaster_decomposed_mode_emits_touch_and_direction_logits() -> None:
    model = _model()
    model.set_barrier_mode("decomposed")
    bsz = 3
    surf = _surface_tensors(bsz)
    out = model(torch.randn(bsz, A, L, FEAT), surface=surf)
    assert out["touch_logit"].shape == (bsz,)
    assert out["direction_logit"].shape == (bsz,)
    assert out["barrier_logits"].shape == (bsz, 3)
    # barrier_logits is log(probs) reconstructed from touch/direction -- softmax must recover
    # a valid, normalized 3-way distribution consistent with the two binary heads.
    probs = torch.softmax(out["barrier_logits"], dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(bsz), atol=1e-5)
    p_touch = torch.sigmoid(out["touch_logit"])
    assert torch.allclose(probs[:, 2], 1.0 - p_touch, atol=1e-5)  # timeout = 1 - touch


def test_forecaster_decomposed_mode_rejects_invalid_barrier_mode() -> None:
    model = _model()
    with pytest.raises(ValueError):
        model.set_barrier_mode("not_a_real_mode")


def test_training_decomposed_mode_trains_touch_and_direction_heads_not_old_heads() -> None:
    """Gradients must reach the new heads (and their skip-connected option-surface encoder)
    while the OLD 3-way barrier heads (unused in this mode) stay completely inert -- proving
    the double-supervision guard in composite_loss.py actually works."""
    torch.manual_seed(0)
    bsz = 8
    feats = torch.randn(bsz, A, L, FEAT)
    ret = torch.rand(bsz) * 0.04
    direction = torch.randint(0, 3, (bsz,))
    barrier = torch.randint(0, 3, (bsz,))  # 0=stop, 1=target, 2=timeout
    surf = _surface_tensors(bsz)
    cfg = TrainingConfig(device="cpu", lr=1e-2, max_epochs=20, embargo_bars=12, weight_decay=0.0)

    model = _model()
    model.set_barrier_mode("decomposed")
    w_touch = model.touch_head.mlp[0].weight.detach().clone()
    w_direction = model.direction_head.mlp[0].weight.detach().clone()
    w_surface = model.option_surface_encoder.phi[0].weight.detach().clone()
    w_barrier_head = model.barrier_head.linear.weight.detach().clone()
    w_excursion_barrier = model.excursion_barrier_head.linear.weight.detach().clone()

    batch = ForecastBatch(
        feats, ret, direction, barrier=barrier,
        surface_grid=surf.grid, surface_mask=surf.mask, surface_context=surf.context,
    )
    HRWTrainer(model, ForecasterLoss(), cfg).fit([batch])

    assert not torch.allclose(w_touch, model.touch_head.mlp[0].weight)
    assert not torch.allclose(w_direction, model.direction_head.mlp[0].weight)
    assert not torch.allclose(w_surface, model.option_surface_encoder.phi[0].weight)
    # The old 3-way heads are unused in decomposed mode -- must receive zero gradient.
    assert torch.allclose(w_barrier_head, model.barrier_head.linear.weight)
    assert torch.allclose(w_excursion_barrier, model.excursion_barrier_head.linear.weight)


def test_direction_loss_ignores_timeout_rows() -> None:
    """All-timeout labels must train touch_head (learn to always predict "no touch") but
    leave direction_head untouched (there's no direction to supervise when nothing touched)."""
    torch.manual_seed(0)
    bsz = 8
    feats = torch.randn(bsz, A, L, FEAT)
    ret = torch.rand(bsz) * 0.04
    direction = torch.randint(0, 3, (bsz,))
    barrier = torch.full((bsz,), 2, dtype=torch.long)  # all timeout
    surf = _surface_tensors(bsz)
    cfg = TrainingConfig(device="cpu", lr=1e-2, max_epochs=20, embargo_bars=12, weight_decay=0.0)

    model = _model()
    model.set_barrier_mode("decomposed")
    w_direction = model.direction_head.mlp[0].weight.detach().clone()

    batch = ForecastBatch(
        feats, ret, direction, barrier=barrier,
        surface_grid=surf.grid, surface_mask=surf.mask, surface_context=surf.context,
    )
    HRWTrainer(model, ForecasterLoss(), cfg).fit([batch])

    assert torch.allclose(w_direction, model.direction_head.mlp[0].weight)
