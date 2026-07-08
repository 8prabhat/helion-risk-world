from __future__ import annotations

from helion_risk_world.schemas.portfolio_schema import PortfolioState, RiskProfile

_EPS = 1e-9


class ExposureManager:
    """Tracks/validates gross exposure vs limits. SRP: exposure only (SPEC.md §19)."""

    def exceeds_limit(self, state: PortfolioState, risk: RiskProfile) -> bool:
        return bool(state.exposure >= max(0.0, float(risk.max_exposure)) - _EPS)
