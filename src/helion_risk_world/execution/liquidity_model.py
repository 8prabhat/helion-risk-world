"""Liquidity model (SPEC.md §15, Day 5).

Returns a liquidity score in [0, 1] (1 = ample). V1 derives it from top-of-book depth when
available, otherwise a neutral-conservative default. SRP: liquidity only.
"""

from __future__ import annotations

from helion_risk_world.schemas.execution_schema import CandidateOrder, ExecutionState


class LiquidityModel:
    """Assess liquidity condition (V1: depth when available; else neutral). SRP: liquidity only."""

    def __init__(self, default_score: float = 0.95) -> None:
        # Default high: the V1 universe (index + large-cap banks) is liquid, and absence of depth
        # data should NOT imply poor liquidity (cost is still charged via spread/slippage/
        # statutory).
        # V2 replaces this with measured book depth.
        if not 0.0 <= default_score <= 1.0:
            raise ValueError("default_score must be in [0, 1]")
        self._default = default_score

    def condition(self, order: CandidateOrder, market: ExecutionState) -> float:
        """Liquidity score in [0, 1]. depth/qty when depth is known, else the neutral default."""
        if market.depth is not None and order.qty > 0:
            return float(min(1.0, market.depth / order.qty))
        return self._default
