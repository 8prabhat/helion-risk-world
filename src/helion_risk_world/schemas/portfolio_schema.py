"""Portfolio-plane schemas. These fields MUST NOT reach the Market World encoder (SPEC.md §5).

Consumed only by: Portfolio World, Planner, Position Sizer, Risk Shield, Reward Scorer, Backtest
Simulator. The field names here are registered in ``data/leakage_checks.PORTFOLIO_FEATURE_NAMES``.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class PositionSide(StrEnum):
    FLAT = "flat"
    LONG = "long"
    SHORT = "short"


class RiskProfile(BaseModel):
    """Account risk configuration (resolved from configs/risk_profiles.yaml)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    max_risk_per_trade: float = Field(gt=0, description="Fraction of capital at risk per trade.")
    max_daily_loss: float = Field(gt=0)
    max_weekly_loss: float = Field(gt=0)
    max_drawdown: float = Field(gt=0)
    max_exposure: float = Field(
        gt=0,
        description=(
            "Max committed exposure fraction. For futures-backed execution this is interpreted as "
            "margin budget, not full cash notional."
        ),
    )
    max_trades_per_day: int = Field(ge=0)
    consecutive_loss_cooldown: int = Field(ge=0)
    cvar_alpha: float = Field(default=0.05, gt=0, lt=1)
    n_paths: int = Field(default=256, ge=1, description="Monte-Carlo paths for consequence sims.")


class PortfolioState(BaseModel):
    """Mutable account state at a decision step. Portfolio plane — never a market feature."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ts: datetime
    capital0: float = Field(description="Initial capital.")
    capital: float
    cash: float
    position: PositionSide = PositionSide.FLAT
    position_qty: float = 0.0
    entry_price: float | None = None
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    daily_pnl: float = 0.0
    drawdown: float = Field(default=0.0, ge=0.0, description="Current drawdown fraction.")
    margin_used: float = 0.0
    free_margin: float = 0.0
    exposure: float = Field(
        default=0.0,
        description=(
            "Committed exposure fraction. For futures-backed execution this tracks margin "
            "utilization rather than full contract notional."
        ),
    )
    risk_budget_used: float = 0.0
    trades_today: int = 0
    consecutive_losses: int = 0
    # Aggregated option greeks (for option positions).
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_theta: float = 0.0
    net_vega: float = 0.0
    expiry_concentration: float = 0.0
    strike_concentration: float = 0.0


class Consequence(BaseModel):
    """Output of PortfolioWorld.step — analytic account-level consequence (SPEC.md §17).

    All values in fraction-of-capital units so the planner's U(a) is unit-consistent.

    CVaR sign convention (SPEC.md §17, §19): ``cvar_dW`` is the **positive shortfall** —
    the expected loss in the worst-α tail, reported ≥ 0. A riskier action has a larger
    ``cvar_dW``, so ``− λ · cvar_dW`` correctly reduces U(a).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    exp_dW: float = Field(description="E[ΔW/capital] from analytic head-implied distribution.")
    cvar_dW: float = Field(
        ge=0.0,
        description="CVaR_α[ΔW/capital] as a POSITIVE shortfall (expected worst-α loss ≥ 0).",
    )
    p_drawdown_breach: float = Field(ge=0.0, le=1.0)
    d_margin: float
    d_exposure: float
