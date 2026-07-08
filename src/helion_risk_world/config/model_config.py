"""Typed model configuration (dataclasses validated from YAML)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ModelSize = Literal["small", "medium", "large"]

# Reference sizes from SPEC.md §29. Mac Studio 64GB -> small/medium only.
_SIZE_PRESETS: dict[str, dict[str, int]] = {
    "small": {"latent_dim": 128, "temporal_layers": 2, "futures_conv_layers": 2},
    "medium": {"latent_dim": 256, "temporal_layers": 4, "futures_conv_layers": 2},
    "large": {"latent_dim": 512, "temporal_layers": 8, "futures_conv_layers": 2},
}


@dataclass(frozen=True)
class ModelConfig:
    """Architecture hyperparameters for the tri-plane model."""

    size: ModelSize = "small"
    latent_dim: int = 128
    temporal_layers: int = 2
    futures_conv_layers: int = 2   # depth of FuturesEncoder 1-D conv stack
    cross_asset_heads: int = 4
    # Only "gated" is implemented today (FusionEncoder raises NotImplementedError
    # for anything else, see encoders/fusion_encoder.py) — review finding M4/Idea #4:
    # the type used to advertise "attention"/"moe"/"uncertainty" as if they existed.
    # Revisit once a concrete driver exists (e.g. the option-surface plane is wired).
    fusion: Literal["gated"] = "gated"
    rollout_samples: int = 16
    dropout: float = 0.1

    @classmethod
    def from_size(cls, size: ModelSize) -> ModelConfig:
        preset = _SIZE_PRESETS[size]
        return cls(size=size, **preset)  # type: ignore[arg-type]

    def __post_init__(self) -> None:
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.rollout_samples < 1:
            raise ValueError("rollout_samples must be >= 1")


@dataclass(frozen=True)
class HorizonConfig:
    """Forecast horizons (SPEC.md §7)."""

    base_interval: str = "5min"
    horizon_steps: tuple[int, ...] = (3, 6, 12)  # 15/30/60 min at 5-min bars

    def __post_init__(self) -> None:
        if not self.horizon_steps or any(h <= 0 for h in self.horizon_steps):
            raise ValueError("horizon_steps must be a non-empty tuple of positive ints")


@dataclass(frozen=True)
class LossWeights:
    """Composite-loss weights (SPEC.md §21)."""

    return_: float = 1.0
    direction: float = 0.5
    volatility: float = 0.3
    mae: float = 0.15
    mfe: float = 0.15
    barrier: float = 0.5
    barrier_intermediate: float = 0.15  # deep-supervision aux at non-management horizons
                                         # (Phase 5b) — starting-point default, needs tuning
    regime: float = 0.3
    calibration: float = 0.4
    uncertainty: float = 0.2
    ood: float = 0.2
    # Per-class weights [stop, target, timeout] for the barrier cross-entropy terms (main +
    # excursion_barrier proxy + barrier_intermediate). None = unweighted CE. The true label
    # distribution is ~12/9/80 stop/target/timeout (see configs/v1.yaml), which without
    # reweighting lets the optimizer collapse to always predicting "timeout" for ~80% "accuracy"
    # with zero gradient signal on the minority classes — confirmed empirically (2026-07-06
    # full retrain reproduced 0 recall on stop/target, identical to an undertrained smoke run).
    barrier_class_weights: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class ModelSpec:
    """Top-level model spec bundling architecture, horizons, and loss weights."""

    model: ModelConfig = field(default_factory=ModelConfig)
    horizons: HorizonConfig = field(default_factory=HorizonConfig)
    loss_weights: LossWeights = field(default_factory=LossWeights)
