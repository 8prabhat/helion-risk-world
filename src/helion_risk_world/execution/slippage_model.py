"""Slippage model (SPEC.md §15, Day 5).

Conservative slippage proxy: a multiple of the spread scaled by order size. V1 has no live depth, so
this deliberately over-estimates rather than under-estimates. SRP: slippage only.
"""

from __future__ import annotations

from helion_risk_world.config.execution_config import CostModelConfig
from helion_risk_world.schemas.execution_schema import CandidateOrder, ExecutionState


class SlippageModel:
    """Estimate slippage from spread and order size. SRP: slippage only (SPEC.md §15)."""

    def __init__(self, cfg: CostModelConfig | None = None) -> None:
        self._cfg = cfg or CostModelConfig()

    def estimate(self, order: CandidateOrder, market: ExecutionState) -> float:
        """Slippage cost (INR), notional-relative: slippage_bps * notional (depth-free)."""
        return float(self._cfg.slippage_bps * abs(order.notional))
