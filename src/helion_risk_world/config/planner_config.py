"""Typed planner configuration (SPEC.md §19).

V1 uses a single-λ mean–CVaR objective:
  U(a) = E[ΔW] − λ · CVaR_α[ΔW] − Cost(a)
with CVaR as a positive shortfall (≥ 0).  One interpretable risk-aversion parameter,
not eight magic weights that require repeated ad-hoc rescaling.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PlannerConfig:
    """Conservative MPC planner configuration — mean–CVaR objective (SPEC.md §19)."""

    risk_aversion_lambda: float = 3.0
    cvar_alpha: float = 0.05
    sizes: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0)
    n_outcome_samples: int = 1000
    # 2026-07-16: "quantile" sizes stop/target from the model's own predicted
    # return-quantile distribution (asymmetric, regime-adaptive) instead of the fixed
    # symmetric BarrierContext multiplier frozen at training time -- see
    # ModelPrediction.quantile_stop_return's docstring for the diagnosis this responds
    # to. Default preserves the original behavior exactly.
    stop_target_mode: str = "barrier_context"

    def __post_init__(self) -> None:
        if not self.sizes or self.sizes[0] != 0.0:
            raise ValueError("sizes must start at 0.0 (NO_TRADE size) and be non-empty")
        if self.risk_aversion_lambda < 0:
            raise ValueError("risk_aversion_lambda must be >= 0")
        if not 0.0 < self.cvar_alpha < 1.0:
            raise ValueError("cvar_alpha must be in (0, 1)")
        if self.stop_target_mode not in ("barrier_context", "quantile"):
            raise ValueError(f"unsupported stop_target_mode: {self.stop_target_mode!r}")
