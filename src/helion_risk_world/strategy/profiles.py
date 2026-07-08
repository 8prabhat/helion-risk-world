"""Built-in trading strategy profiles.

Each profile reuses the common world/risk/execution stack while tuning horizon, hold
cadence, planner aggressiveness, and account-level risk limits for a distinct style.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from helion_risk_world.config.planner_config import PlannerConfig
from helion_risk_world.schemas.portfolio_schema import RiskProfile


class StrategyName(StrEnum):
    SCALPING = "scalping"
    MEDIUM_FREQUENCY = "medium_frequency"
    LOW_FREQUENCY = "low_frequency"


@dataclass(frozen=True)
class RiskProfileOverride:
    """Partial risk-profile update applied per strategy."""

    max_risk_per_trade: float | None = None
    max_daily_loss: float | None = None
    max_weekly_loss: float | None = None
    max_drawdown: float | None = None
    max_exposure: float | None = None
    max_trades_per_day: int | None = None
    consecutive_loss_cooldown: int | None = None
    cvar_alpha: float | None = None
    n_paths: int | None = None

    def apply(self, risk: RiskProfile) -> RiskProfile:
        """Return a copy of ``risk`` with the configured overrides applied."""
        updates = {
            name: value for name, value in self.__dict__.items() if value is not None
        }
        return risk if not updates else risk.model_copy(update=updates)


@dataclass(frozen=True)
class StrategyProfile:
    """Typed operating profile for one trading style."""

    name: StrategyName
    decision_horizon_bars: int
    max_hold_bars: int
    confidence_scale: float
    planner_config: PlannerConfig
    risk_override: RiskProfileOverride = field(default_factory=RiskProfileOverride)
    description: str = ""

    def apply_risk(self, risk: RiskProfile) -> RiskProfile:
        """Adapt a base account risk profile to this strategy."""
        return self.risk_override.apply(risk)


_BUILTIN_PROFILES: dict[StrategyName, StrategyProfile] = {
    StrategyName.SCALPING: StrategyProfile(
        name=StrategyName.SCALPING,
        decision_horizon_bars=3,
        max_hold_bars=3,
        confidence_scale=0.85,
        planner_config=PlannerConfig(
            risk_aversion_lambda=1.5,
            cvar_alpha=0.10,
            sizes=(0.0, 0.05, 0.10, 0.20, 0.35),
            n_outcome_samples=750,
        ),
        risk_override=RiskProfileOverride(
            max_risk_per_trade=0.004,
            max_daily_loss=0.010,
            max_weekly_loss=0.025,
            max_drawdown=0.050,
            max_exposure=0.35,
            max_trades_per_day=24,
            consecutive_loss_cooldown=2,
            cvar_alpha=0.10,
            n_paths=192,
        ),
        description="Short-horizon, low-exposure trading with tight holding windows.",
    ),
    StrategyName.MEDIUM_FREQUENCY: StrategyProfile(
        name=StrategyName.MEDIUM_FREQUENCY,
        decision_horizon_bars=192,
        max_hold_bars=192,
        confidence_scale=1.0,
        planner_config=PlannerConfig(
            risk_aversion_lambda=3.0,
            cvar_alpha=0.05,
            sizes=(0.0, 0.10, 0.25, 0.50, 0.75),
            n_outcome_samples=1000,
        ),
        risk_override=RiskProfileOverride(
            max_risk_per_trade=0.010,
            max_daily_loss=0.020,
            max_weekly_loss=0.050,
            max_drawdown=0.100,
            max_exposure=0.75,
            max_trades_per_day=10,
            consecutive_loss_cooldown=4,
            cvar_alpha=0.05,
            n_paths=256,
        ),
        description="Barrier-managed BankNIFTY futures profile aligned to the H192 artifact.",
    ),
    StrategyName.LOW_FREQUENCY: StrategyProfile(
        name=StrategyName.LOW_FREQUENCY,
        decision_horizon_bars=12,
        max_hold_bars=12,
        confidence_scale=0.95,
        planner_config=PlannerConfig(
            risk_aversion_lambda=4.5,
            cvar_alpha=0.025,
            sizes=(0.0, 0.10, 0.20, 0.35, 0.50),
            n_outcome_samples=1250,
        ),
        risk_override=RiskProfileOverride(
            max_risk_per_trade=0.006,
            max_daily_loss=0.015,
            max_weekly_loss=0.035,
            max_drawdown=0.080,
            max_exposure=0.60,
            max_trades_per_day=4,
            consecutive_loss_cooldown=5,
            cvar_alpha=0.025,
            n_paths=384,
        ),
        description="Lower-turnover profile favoring persistence and deeper risk buffers.",
    ),
}


def get_strategy_profile(name: str | StrategyName | None = None) -> StrategyProfile:
    """Return one of the built-in strategy profiles."""
    strategy_name = StrategyName(name or StrategyName.MEDIUM_FREQUENCY)
    return _BUILTIN_PROFILES[strategy_name]


def available_strategy_names() -> tuple[str, ...]:
    """List supported strategy names for CLI/config validation."""
    return tuple(profile.value for profile in StrategyName)


__all__ = [
    "RiskProfileOverride",
    "StrategyName",
    "StrategyProfile",
    "available_strategy_names",
    "get_strategy_profile",
]
