from __future__ import annotations

from helion_risk_world.schemas.portfolio_schema import PortfolioState, RiskProfile


class DrawdownGuard:
    """Daily-loss + max-drawdown guard. SRP: drawdown only (SPEC.md §19)."""

    def daily_loss_breached(self, state: PortfolioState, risk: RiskProfile) -> bool:
        limit = -float(risk.max_daily_loss) * float(state.capital0)
        return bool(state.daily_pnl <= limit)

    def max_drawdown_breached(self, state: PortfolioState, risk: RiskProfile) -> bool:
        return bool(state.drawdown >= float(risk.max_drawdown))
