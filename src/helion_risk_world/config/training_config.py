"""Typed training configuration (SPEC.md §20, §29)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    """Optimisation and walk-forward settings."""

    seed: int = 7
    device: str = "auto"               # auto -> mps on Mac Studio, else cpu/cuda
    batch_size: int = 256
    grad_accum_steps: int = 1
    max_epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0        # max global grad norm; <= 0 disables clipping
    early_stopping_patience: int = 8
    checkpoint_dir: str = "runs"
    pretrain_epochs: int = 0
    # Review finding H7: Stage 4 head-fine-tuning (HeadTrainer) was documented as a
    # wired pipeline stage but no CLI script ever called it. 0 = off (default,
    # matches the historical behavior of HRWTrainer alone training end-to-end).
    head_finetune_epochs: int = 0
    # Review finding H6: with the default lookback_bars=96, gap_bars=1 made the Stage-2
    # future-latent-prediction task near-identity (context/future windows overlap 95/96
    # bars), defeating its purpose. 12 matches the default management horizon
    # (horizons.horizon_steps=[3,6,12]) so pretraining predicts a genuinely future latent.
    pretrain_gap_bars: int = 12
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    # Walk-forward / purged split.
    n_folds: int = 5
    embargo_bars: int = 12             # >= max horizon to prevent label bleed

    def __post_init__(self) -> None:
        if self.embargo_bars < 1:
            raise ValueError("embargo_bars must be >= max forecast horizon")
        if self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")
        if self.pretrain_epochs < 0:
            raise ValueError("pretrain_epochs must be >= 0")
        if self.head_finetune_epochs < 0:
            raise ValueError("head_finetune_epochs must be >= 0")
        if self.pretrain_gap_bars < 1:
            raise ValueError("pretrain_gap_bars must be >= 1")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be in (0, 1)")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError("val_fraction must be in [0, 1)")
        if self.train_fraction + self.val_fraction >= 1.0:
            raise ValueError("train_fraction + val_fraction must be < 1")
