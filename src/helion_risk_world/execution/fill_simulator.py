"""Fill simulator (SPEC.md §15, Day 5).

Estimates fill / partial-fill / rejection probabilities. V1 scales a base fill probability by the
liquidity score; partial-fill and rejection absorb the remainder. SRP: fills only.
"""

from __future__ import annotations

from dataclasses import dataclass

from helion_risk_world.config.execution_config import CostModelConfig
from helion_risk_world.schemas.execution_schema import CandidateOrder, ExecutionState


@dataclass(frozen=True)
class FillProbabilities:
    """Fill / partial-fill / rejection probabilities; the three sum to 1.0."""

    fill: float
    partial: float
    reject: float


class FillSimulator:
    """Simulate fill / partial-fill / rejection probabilities. SRP: fills only (SPEC.md §15)."""

    def __init__(self, cfg: CostModelConfig | None = None) -> None:
        self._cfg = cfg or CostModelConfig()

    def probabilities(
        self, order: CandidateOrder, market: ExecutionState, liquidity: float
    ) -> FillProbabilities:
        """Full breakdown. fill scales with liquidity; the rest splits into partial then reject."""
        fill = max(0.0, min(1.0, self._cfg.base_fill_prob * liquidity))
        remainder = 1.0 - fill
        partial = remainder * 0.7   # most non-full fills are partials, not outright rejects
        reject = remainder - partial
        return FillProbabilities(fill=fill, partial=partial, reject=reject)

    def fill_prob(self, order: CandidateOrder, market: ExecutionState, liquidity: float) -> float:
        """Probability of a complete fill."""
        return self.probabilities(order, market, liquidity).fill
