from __future__ import annotations

from helion_risk_world.schemas.portfolio_schema import PortfolioState, RiskProfile


class MarginSimulator:
    """Simulates margin usage / free-margin floor. SRP: margin only (SPEC.md §19)."""

    def free_margin_below_floor(
        self, state: PortfolioState, risk: RiskProfile, floor: float
    ) -> bool:
        required_free_margin = max(0.0, float(floor)) * float(state.capital0)
        return bool(state.free_margin < required_free_margin)
