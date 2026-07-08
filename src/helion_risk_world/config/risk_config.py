"""Typed risk-shield configuration (SPEC.md §19, §22)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskShieldConfig:
    """Hard, deterministic thresholds. The ML model can never override these."""

    daily_loss_limit: float = 0.02          # fraction of capital
    max_drawdown: float = 0.10
    free_margin_floor: float = 0.20         # fraction of capital
    uncertainty_block_threshold: float = 0.8
    ood_block_threshold: float = 0.9
    slippage_block_threshold: float = 0.5   # fraction of expected edge
    max_exposure: float = 1.0
    max_trades_per_day: int = 10
    consecutive_loss_cooldown: int = 4
    require_edge_over_cost: bool = True
    blackout_block: bool = True

    def __post_init__(self) -> None:
        for name in ("daily_loss_limit", "max_drawdown", "free_margin_floor"):
            v = getattr(self, name)
            if not 0.0 < v < 1.0:
                raise ValueError(f"{name} must be in (0, 1), got {v}")
