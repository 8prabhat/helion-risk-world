from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from helion_risk_world.config.model_config import LossWeights, ModelConfig  # noqa: E402
from helion_risk_world.config.training_config import TrainingConfig  # noqa: E402
from helion_risk_world.data.regime_builder import REGIME_CONTEXT_FEATURES  # noqa: E402
from helion_risk_world.encoders.futures_encoder import FUTURES_FEATURE_DIM  # noqa: E402
from helion_risk_world.losses.world_model_loss import WorldModelLoss  # noqa: E402
from helion_risk_world.model import HRWWorldModel  # noqa: E402
from helion_risk_world.training.trainer import ForecastBatch, HRWTrainer  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
_SPEC = importlib.util.spec_from_file_location("train_script", _ROOT / "scripts" / "train.py")
assert _SPEC is not None and _SPEC.loader is not None
train_script = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = train_script
_SPEC.loader.exec_module(train_script)


def test_demo_batches_include_multi_horizon_targets() -> None:
    batches = train_script.build_demo_batches(
        ("BANKNIFTY", "NIFTY"),
        12,
        12,
        batch_size=16,
        target_horizons=(3, 6, 12),
    )
    batch = batches[0]
    assert batch.horizon_returns is not None
    assert batch.horizon_volatility is not None
    assert batch.target_horizons == (3, 6, 12)
    assert batch.horizon_returns.shape[1] == 3
    assert batch.horizon_volatility.shape[1] == 3


def test_world_model_supervised_head_training_smoke() -> None:
    batches = train_script.build_demo_batches(
        ("BANKNIFTY", "NIFTY"),
        12,
        12,
        batch_size=16,
        target_horizons=(3, 6, 12),
    )
    model = HRWWorldModel(
        n_features=batches[0].features.shape[-1],
        cfg=ModelConfig(latent_dim=16, temporal_layers=1, dropout=0.0, rollout_samples=4),
        horizons=(3, 6, 12),
        n_samples=4,
    )
    trainer = HRWTrainer(
        model,
        WorldModelLoss(),
        TrainingConfig(device="cpu", batch_size=16, max_epochs=1, lr=1e-3),
    )
    trainer.fit(batches[:2], epochs=1)
    out = model(batches[0].features[:2])
    assert out["return_quantiles"].shape == (2, 3, 5)
    assert out["mae"].shape == (2, 3)
    assert out["mfe"].shape == (2, 3)


def test_world_model_supervised_loss_backprops_into_encoders_and_rssm() -> None:
    torch.manual_seed(0)
    batch = ForecastBatch(
        features=torch.randn(4, 2, 12, 9),
        forward_return=torch.randn(4),
        direction=torch.zeros(4, dtype=torch.long),
        regime=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        futures=torch.randn(4, 12, FUTURES_FEATURE_DIM),
        regime_context=torch.randn(4, len(REGIME_CONTEXT_FEATURES)),
        realized_vol=torch.rand(4),
        barrier=torch.tensor([0, 1, 2, 1], dtype=torch.long),
        sample_weight=torch.ones(4),
        horizon_returns=torch.randn(4, 3),
        horizon_volatility=torch.rand(4, 3),
        target_horizons=(3, 6, 12),
    )
    model = HRWWorldModel(
        n_features=9,
        cfg=ModelConfig(latent_dim=16, temporal_layers=1, dropout=0.0, rollout_samples=4),
        horizons=(3, 6, 12),
        n_samples=4,
    )
    out = model(batch.features, batch.futures, batch.regime_context, n_samples=4)
    loss = WorldModelLoss()(out, batch)
    loss.backward()

    assert model.temporal.proj.weight.grad is not None
    assert model.cross_asset.proj.weight.grad is not None
    assert model.futures_encoder.conv[0].weight.grad is not None
    assert model.regime_encoder.mlp[0].weight.grad is not None
    assert model.fusion.gate.weight.grad is not None
    assert model.market_world.rssm.gru.weight_hh.grad is not None


def test_world_model_loss_includes_calibration_component() -> None:
    batch = ForecastBatch(
        features=torch.randn(2, 2, 12, 9),
        forward_return=torch.zeros(2),
        direction=torch.zeros(2, dtype=torch.long),
        horizon_returns=torch.full((2, 2), 0.02),
        horizon_volatility=torch.full((2, 2), 0.01),
        target_horizons=(3, 6),
    )
    model = HRWWorldModel(
        n_features=9,
        cfg=ModelConfig(latent_dim=16, temporal_layers=1, dropout=0.0, rollout_samples=4),
        horizons=(3, 6),
        n_samples=4,
    )
    prediction = model(batch.features)
    loss = WorldModelLoss(
        weights=LossWeights(return_=0.0, volatility=0.0, calibration=1.0),
    )

    total = loss(prediction, batch)

    assert float(total.detach()) > 0.0
    assert "calibration" in loss.last_components
    assert loss.last_components["calibration"] > 0.0


def test_world_model_direction_surrogate_uses_horizon_returns_not_single_step_direction() -> None:
    batch = ForecastBatch(
        features=torch.randn(1, 2, 12, 9),
        forward_return=torch.tensor([-0.02], dtype=torch.float32),
        direction=torch.tensor([0], dtype=torch.long),
        horizon_returns=torch.tensor([[0.02, 0.025]], dtype=torch.float32),
        horizon_volatility=torch.full((1, 2), 0.01, dtype=torch.float32),
        target_horizons=(3, 6),
    )
    loss = WorldModelLoss(
        weights=LossWeights(
            return_=0.0,
            volatility=0.0,
            calibration=0.0,
            barrier=0.0,
            regime=0.0,
            mae=0.0,
            mfe=0.0,
            direction=1.0,
        ),
    )
    aligned = {
        "horizons": (3, 6),
        "return_quantiles": torch.tensor(
            [[[-0.002, -0.001, 0.004, 0.006, 0.008], [-0.002, -0.001, 0.005, 0.007, 0.009]]],
            dtype=torch.float32,
        ),
        "volatility": torch.full((1, 2), 0.01, dtype=torch.float32),
    }
    misaligned = {
        "horizons": (3, 6),
        "return_quantiles": torch.tensor(
            [[[-0.009, -0.007, -0.004, -0.002, 0.001], [-0.010, -0.008, -0.005, -0.002, 0.001]]],
            dtype=torch.float32,
        ),
        "volatility": torch.full((1, 2), 0.01, dtype=torch.float32),
    }

    aligned_loss = float(loss(aligned, batch))
    misaligned_loss = float(loss(misaligned, batch))

    assert misaligned_loss > aligned_loss
    assert loss.last_components["direction"] > 0.0


def test_world_model_loss_uses_barrier_geometry_for_intermediate_barrier_aux() -> None:
    """Phase 5b: the excursion-reconstructed aux loss moved from the (now real-label-only)
    management horizon to the intermediate horizons, using horizon_mae/horizon_mfe[:, :-1]
    and barrier_logits_intermediate instead of the final-step barrier_logits."""
    # horizon index 0 is the intermediate horizon (3 bars); index 1 (12 bars, management) is
    # unused here since barrier=0.0. mae/mfe at index 0 give edge = 0.001/0.001 - 0.006/0.001
    # = -5, well past the neutral band -> a clear "stop" (class 0) label at the intermediate
    # horizon, scaled by sqrt(3/12)=0.5 from the raw barrier_context stop/target widths.
    batch = ForecastBatch(
        features=torch.randn(1, 2, 12, 9),
        forward_return=torch.tensor([0.0], dtype=torch.float32),
        direction=torch.tensor([1], dtype=torch.long),
        horizon_returns=torch.zeros((1, 2), dtype=torch.float32),
        horizon_volatility=torch.full((1, 2), 0.01, dtype=torch.float32),
        horizon_mae=torch.tensor([[0.006, 0.001]], dtype=torch.float32),
        horizon_mfe=torch.tensor([[0.001, 0.001]], dtype=torch.float32),
        barrier_context=torch.tensor([[0.001, -0.002, 0.002]], dtype=torch.float32),
        target_horizons=(3, 12),
    )
    loss = WorldModelLoss(
        weights=LossWeights(
            return_=0.0,
            volatility=0.0,
            calibration=0.0,
            barrier=0.0,
            barrier_intermediate=1.0,
            regime=0.0,
            mae=0.0,
            mfe=0.0,
            direction=0.0,
        ),
    )
    aligned = {
        "horizons": (3, 12),
        "return_quantiles": torch.zeros((1, 2, 5), dtype=torch.float32),
        "volatility": torch.full((1, 2), 0.01, dtype=torch.float32),
        "barrier_logits_intermediate": torch.tensor([[[3.0, -3.0, -3.0]]], dtype=torch.float32),
    }
    misaligned = {
        "horizons": (3, 12),
        "return_quantiles": torch.zeros((1, 2, 5), dtype=torch.float32),
        "volatility": torch.full((1, 2), 0.01, dtype=torch.float32),
        "barrier_logits_intermediate": torch.tensor([[[-3.0, 3.0, -3.0]]], dtype=torch.float32),
    }

    aligned_loss = float(loss(aligned, batch))
    aligned_aux = loss.last_components["barrier_intermediate"]
    misaligned_loss = float(loss(misaligned, batch))
    misaligned_aux = loss.last_components["barrier_intermediate"]

    assert misaligned_loss > aligned_loss
    assert misaligned_aux > aligned_aux


def test_world_model_forward_includes_barrier_logits() -> None:
    """Regression guard for the Phase 5a bug: HRWWorldModel.forward() was silently
    dropping "barrier_logits" from its output dict, so both barrier loss terms in
    WorldModelLoss were no-ops for every batch of real training."""
    model = HRWWorldModel(
        n_features=9,
        cfg=ModelConfig(latent_dim=16, temporal_layers=1, dropout=0.0, rollout_samples=4),
        horizons=(3, 6, 12),
        n_samples=4,
    )
    out = model(torch.randn(2, 2, 12, 9))
    assert isinstance(out["barrier_logits"], torch.Tensor)
    assert out["barrier_logits"].shape == (2, 3)
    assert isinstance(out["barrier_logits_intermediate"], torch.Tensor)
    assert out["barrier_logits_intermediate"].shape == (2, 2, 3)


def _world_model_barrier_training_batch(bsz: int = 8) -> ForecastBatch:
    torch.manual_seed(0)
    return ForecastBatch(
        features=torch.randn(bsz, 2, 12, 9),
        forward_return=torch.zeros(bsz),
        direction=torch.zeros(bsz, dtype=torch.long),
        barrier=torch.randint(0, 3, (bsz,)),
        barrier_context=torch.tensor([[0.01, -0.02, 0.02]] * bsz, dtype=torch.float32),
        horizon_returns=torch.randn(bsz, 3),
        horizon_volatility=torch.rand(bsz, 3) * 0.02 + 0.01,
        horizon_mae=torch.rand(bsz, 3) * 0.02,
        horizon_mfe=torch.rand(bsz, 3) * 0.02,
        target_horizons=(3, 6, 12),
    )


def test_world_model_barrier_supervision_trains_derived_head() -> None:
    """The load-bearing regression test for the Phase 5a bug: a real HRWTrainer.fit()
    call must actually move barrier_head/excursion_barrier_head weights. This test
    (or its legacy-mode equivalent below) would have caught the dropped-key bug on the
    first CI run — the equivalent already existed for HRWForecaster
    (test_barrier_vol_heads.py) but was never written for HRWWorldModel."""
    batch = _world_model_barrier_training_batch()
    cfg = TrainingConfig(device="cpu", lr=1e-2, max_epochs=5, embargo_bars=12, weight_decay=0.0)

    model = HRWWorldModel(
        n_features=9,
        cfg=ModelConfig(latent_dim=16, temporal_layers=1, dropout=0.0, rollout_samples=4),
        horizons=(3, 6, 12),
        n_samples=4,
    )
    model.set_barrier_mode("derived")
    initial = model.market_world.excursion_barrier_head.linear.weight.detach().clone()

    HRWTrainer(model, WorldModelLoss(), cfg).fit([batch])

    assert not torch.allclose(initial, model.market_world.excursion_barrier_head.linear.weight)


def test_world_model_barrier_supervision_trains_legacy_head() -> None:
    batch = _world_model_barrier_training_batch()
    cfg = TrainingConfig(device="cpu", lr=1e-2, max_epochs=5, embargo_bars=12, weight_decay=0.0)

    model = HRWWorldModel(
        n_features=9,
        cfg=ModelConfig(latent_dim=16, temporal_layers=1, dropout=0.0, rollout_samples=4),
        horizons=(3, 6, 12),
        n_samples=4,
    )
    model.set_barrier_mode("legacy")
    initial = model.market_world.barrier_head.linear.weight.detach().clone()

    HRWTrainer(model, WorldModelLoss(), cfg).fit([batch])

    assert not torch.allclose(initial, model.market_world.barrier_head.linear.weight)


def test_world_model_barrier_intermediate_loss_reaches_rssm_via_short_path() -> None:
    """Phase 5b's actual claim: with ONLY the intermediate-horizon barrier loss active
    (management-horizon barrier weight zeroed out), gradient still reaches an early RSSM
    parameter — proving the short 3/6-bar paths carry real gradient independent of the
    full 12-bar (or, in production, 192-bar) management-horizon path."""
    torch.manual_seed(0)
    batch = ForecastBatch(
        features=torch.randn(4, 2, 12, 9),
        forward_return=torch.randn(4),
        direction=torch.zeros(4, dtype=torch.long),
        barrier_context=torch.tensor([[0.01, -0.02, 0.02]] * 4, dtype=torch.float32),
        horizon_returns=torch.randn(4, 3),
        horizon_volatility=torch.rand(4, 3) * 0.02 + 0.01,
        horizon_mae=torch.rand(4, 3) * 0.02,
        horizon_mfe=torch.rand(4, 3) * 0.02,
        target_horizons=(3, 6, 12),
    )
    model = HRWWorldModel(
        n_features=9,
        cfg=ModelConfig(latent_dim=16, temporal_layers=1, dropout=0.0, rollout_samples=4),
        horizons=(3, 6, 12),
        n_samples=4,
    )
    out = model(batch.features, barrier_context=batch.barrier_context, n_samples=4)
    loss = WorldModelLoss(
        weights=LossWeights(
            return_=0.0,
            volatility=0.0,
            calibration=0.0,
            barrier=0.0,
            barrier_intermediate=1.0,
            regime=0.0,
            mae=0.0,
            mfe=0.0,
            direction=0.0,
            uncertainty=0.0,
            ood=0.0,
        ),
    )(out, batch)
    loss.backward()

    assert model.market_world.rssm.gru.weight_ih.grad is not None
    assert float(model.market_world.rssm.gru.weight_ih.grad.abs().sum()) > 0.0


def test_world_model_loss_penalizes_return_excursion_incoherence() -> None:
    batch = ForecastBatch(
        features=torch.randn(1, 2, 12, 9),
        forward_return=torch.tensor([0.0], dtype=torch.float32),
        direction=torch.tensor([1], dtype=torch.long),
        horizon_returns=torch.zeros((1, 2), dtype=torch.float32),
        horizon_volatility=torch.full((1, 2), 0.01, dtype=torch.float32),
        horizon_mae=torch.full((1, 2), 0.005, dtype=torch.float32),
        horizon_mfe=torch.full((1, 2), 0.005, dtype=torch.float32),
        target_horizons=(3, 12),
    )
    loss = WorldModelLoss(
        weights=LossWeights(
            return_=0.0,
            volatility=0.0,
            calibration=0.0,
            barrier=0.0,
            regime=0.0,
            direction=0.0,
            mae=1.0,
            mfe=1.0,
        ),
    )
    coherent = {
        "horizons": (3, 12),
        "return_quantiles": torch.tensor(
            [[[-0.004, -0.002, 0.0, 0.002, 0.004], [-0.004, -0.002, 0.0, 0.002, 0.004]]],
            dtype=torch.float32,
        ),
        "volatility": torch.full((1, 2), 0.01, dtype=torch.float32),
        "mae": torch.full((1, 2), 0.005, dtype=torch.float32),
        "mfe": torch.full((1, 2), 0.005, dtype=torch.float32),
    }
    incoherent = {
        "horizons": (3, 12),
        "return_quantiles": torch.tensor(
            [[[-0.007, -0.006, 0.0, 0.006, 0.007], [-0.007, -0.006, 0.0, 0.006, 0.007]]],
            dtype=torch.float32,
        ),
        "volatility": torch.full((1, 2), 0.01, dtype=torch.float32),
        "mae": torch.full((1, 2), 0.005, dtype=torch.float32),
        "mfe": torch.full((1, 2), 0.005, dtype=torch.float32),
    }

    coherent_loss = float(loss(coherent, batch))
    coherent_aux = loss.last_components["excursion_coherence"]
    incoherent_loss = float(loss(incoherent, batch))
    incoherent_aux = loss.last_components["excursion_coherence"]

    assert incoherent_loss > coherent_loss
    assert incoherent_aux > coherent_aux


def test_build_world_model_sequences_uses_overlap_and_tail_coverage() -> None:
    batch = ForecastBatch(
        features=torch.randn(10, 1, 4, 3),
        forward_return=torch.zeros(10),
        direction=torch.zeros(10, dtype=torch.long),
    )

    sequences = train_script._build_world_model_sequences([batch], seq_len=4, seq_batch_size=8)

    total_windows = sum(seq[0].shape[1] for seq in sequences)
    assert total_windows == 4  # starts 0, 2, 4, 6 with default half-overlap stride
    last_chunk = sequences[-1][0]
    assert torch.allclose(last_chunk[-1, -1], batch.features[9])
