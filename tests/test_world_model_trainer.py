"""WorldModelTrainer: Stage-3 RSSM dynamics training contract (SPEC.md §14.2, §20)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from helion_risk_world.config.model_config import ModelConfig  # noqa: E402
from helion_risk_world.config.training_config import TrainingConfig  # noqa: E402
from helion_risk_world.model import HRWWorldModel  # noqa: E402
from helion_risk_world.training.train_world_model import WorldModelTrainer  # noqa: E402

B, A, L, FEAT = 2, 1, 12, 7
T_SEQ = 5   # number of consecutive bars in an RSSM training sequence


def _model() -> HRWWorldModel:
    return HRWWorldModel(n_features=FEAT, cfg=ModelConfig(latent_dim=16, temporal_layers=1,
                                                          dropout=0.0), horizons=(1, 3),
                         n_samples=4)


def _cfg() -> TrainingConfig:
    return TrainingConfig(device="cpu", lr=1e-3, max_epochs=2)


def test_encode_sequence_output_shape() -> None:
    """encode_sequence produces [T, B, embed_dim] from raw [T, B, A, L, F]."""
    model = _model()
    raw = torch.randn(T_SEQ, B, A, L, FEAT)
    seq_e = WorldModelTrainer.encode_sequence(raw, model)
    assert seq_e.shape == (T_SEQ, B, model.latent_dim)


def test_encode_sequence_no_grad() -> None:
    """encode_sequence runs in no-grad mode — encoder weights stay frozen during preprocess."""
    model = _model()
    raw = torch.randn(T_SEQ, B, A, L, FEAT, requires_grad=True)
    seq_e = WorldModelTrainer.encode_sequence(raw, model)
    assert not seq_e.requires_grad


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires an MPS device")
def test_encode_sequence_moves_cpu_inputs_to_models_device() -> None:
    """Regression test: Stage-2 pretraining (MarketStatePretrainer.fit()) moves the
    model to the resolved training device internally, before scripts/train.py calls
    encode_sequence() to prep Stage-3 RSSM inputs. The raw sequence tensors built
    from the original batches are still CPU-resident at that point, so
    encode_sequence() must move them to the model's device itself rather than
    assuming both are already co-located — this crashed with a device-mismatch
    RuntimeError end-to-end on a real world_model + pretrain-epochs + MPS run."""
    model = _model().to(torch.device("mps"))
    raw = torch.randn(T_SEQ, B, A, L, FEAT)  # deliberately left on CPU
    seq_e = WorldModelTrainer.encode_sequence(raw, model)
    assert seq_e.device.type == "mps"


def test_fit_reduces_loss() -> None:
    """WorldModelTrainer.fit() must decrease training loss over epochs."""
    model = _model()
    trainer = WorldModelTrainer(model, _cfg())
    # pre-encode a small synthetic sequence
    raw = torch.randn(T_SEQ, B, A, L, FEAT)
    seq_e = WorldModelTrainer.encode_sequence(raw, model)
    trainer.fit([seq_e], epochs=10)
    assert len(trainer.history) == 10
    assert trainer.history[-1] < trainer.history[0], (
        f"loss did not decrease: {trainer.history[0]:.4f} → {trainer.history[-1]:.4f}"
    )


def test_fit_requires_nonempty_sequences() -> None:
    trainer = WorldModelTrainer(_model(), _cfg())
    with pytest.raises(ValueError, match="non-empty"):
        trainer.fit([])


def test_fit_returns_model() -> None:
    model = _model()
    trainer = WorldModelTrainer(model, _cfg())
    raw = torch.randn(T_SEQ, B, A, L, FEAT)
    seq_e = WorldModelTrainer.encode_sequence(raw, model)
    returned = trainer.fit([seq_e], epochs=1)
    assert returned is model


def test_diagnostics_returns_kl_and_prior_coverage_metrics() -> None:
    """Review finding M12: KL-collapse and prior-predictive-coverage diagnostics
    must actually be computable from a real (trained-or-not) RSSM and a batch of
    encoded sequences, not just implemented-but-unreachable in isolation."""
    model = _model()
    trainer = WorldModelTrainer(model, _cfg())
    raw = torch.randn(T_SEQ, B, A, L, FEAT)
    seq_e = WorldModelTrainer.encode_sequence(raw, model)
    trainer.fit([seq_e], epochs=1)

    diagnostics = trainer.diagnostics([seq_e], n_prior_samples=8)

    assert "mean_kl" in diagnostics
    assert "kl_collapse_frac" in diagnostics
    assert "prior_coverage" in diagnostics
    assert 0.0 <= diagnostics["kl_collapse_frac"] <= 1.0
    assert 0.0 <= diagnostics["prior_coverage"] <= 1.0
