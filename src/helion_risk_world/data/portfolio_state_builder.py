"""Counterfactual account-state builder (SPEC.md §14, Day 5).

Produces the canonical starting account conditions the Portfolio World is simulated against, so the
same market future can be evaluated for many accounts (a fresh account vs one already in drawdown vs
one near its daily-loss limit, etc.). PORTFOLIO plane only — these MUST NOT reach the encoders
(SPEC.md §5). SRP: state construction only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from helion_risk_world.schemas.portfolio_schema import PortfolioState, RiskProfile


@dataclass(frozen=True)
class NamedProfile:
    """A labelled counterfactual account state."""

    label: str
    state: PortfolioState


class PortfolioStateBuilder:
    """Build counterfactual PortfolioStates for Portfolio World / Planner ONLY (SPEC.md §14)."""

    def fresh(self, capital0: float, ts: datetime) -> PortfolioState:
        """A flat, full-cash account at the start of the day."""
        return PortfolioState(
            ts=ts, capital0=capital0, capital=capital0, cash=capital0, free_margin=capital0
        )

    def synthetic_profiles(
        self, capital0: float, risk: RiskProfile, ts: datetime
    ) -> list[NamedProfile]:
        """The canonical counterfactual conditions for one risk profile (SPEC.md §14).

        ``capital0`` sets the account scale (RiskProfile carries limits, not capital). The same
        market future evaluated across these implies different correct actions per account.
        """
        base = self.fresh(capital0, ts)
        in_drawdown = base.model_copy(
            update={
                "capital": capital0 * (1 - 0.8 * risk.max_drawdown),
                "drawdown": 0.8 * risk.max_drawdown,
                "daily_pnl": -0.5 * risk.max_daily_loss * capital0,
            }
        )
        near_daily_loss = base.model_copy(
            update={
                "capital": capital0 * (1 - 0.95 * risk.max_daily_loss),
                "daily_pnl": -0.95 * risk.max_daily_loss * capital0,
            }
        )
        near_exposure = base.model_copy(
            update={
                "position": base.position,
                "exposure": 0.95 * risk.max_exposure,
                "margin_used": 0.95 * risk.max_exposure * capital0,
                "free_margin": capital0 * (1 - 0.95 * risk.max_exposure),
            }
        )
        post_losses = base.model_copy(
            update={"consecutive_losses": risk.consecutive_loss_cooldown, "trades_today": 3}
        )
        return [
            NamedProfile("fresh", base),
            NamedProfile("in_drawdown", in_drawdown),
            NamedProfile("near_daily_loss", near_daily_loss),
            NamedProfile("near_exposure", near_exposure),
            NamedProfile("post_consecutive_losses", post_losses),
        ]
