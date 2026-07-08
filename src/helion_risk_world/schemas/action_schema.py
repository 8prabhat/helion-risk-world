"""Action / decision schemas. NO_TRADE is a first-class action (SPEC.md §4, §21).

``FinalDecision`` is the full audit record emitted for every decision in backtest and paper trading
(SPEC.md §28). It must be reconstructable into a human-readable explanation.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ActionType(StrEnum):
    NO_TRADE = "no_trade"   # first-class: an explicit intelligent decision, not absence of signal
    ENTER_LONG = "enter_long"
    ENTER_SHORT = "enter_short"
    EXIT = "exit"
    REDUCE = "reduce"
    INCREASE = "increase"


# Allowed position sizes as fraction of the allowed risk unit.
SIZE_GRID: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0)


class CandidateAction(BaseModel):
    """One candidate the planner scores."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    action_type: ActionType
    size_fraction: float = Field(ge=0.0, le=1.0)


class RiskDecision(BaseModel):
    """Output of the Risk Shield.

    The Risk Shield can override the planner; ML cannot override it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    allowed: bool
    reason_code: str
    final_action: CandidateAction
    adjusted_size: float = Field(ge=0.0, le=1.0)


class ScoredCandidate(BaseModel):
    """A candidate action with its planner score and component penalties (for audit)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    action: CandidateAction
    score: float
    components: dict[str, float] = Field(default_factory=dict)


class FinalDecision(BaseModel):
    """Full audit record for a single decision step (SPEC.md §28)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ts: datetime
    symbol: str
    strategy_name: str | None = None
    market_summary: dict[str, float]
    latent_regime: str
    uncertainty: float
    ood_score: float
    portfolio_summary: dict[str, float]
    candidates: list[ScoredCandidate]
    chosen_before_shield: CandidateAction
    execution_realism: str
    risk_shield: RiskDecision
    final_action: CandidateAction
    reason_code: str
    expected_cost: float
    expected_risk: float
    expected_reward: float
    rejected_actions: list[ScoredCandidate] = Field(default_factory=list)
    executed_action: CandidateAction | None = None
    execution_status: str | None = None
    data_quality_status: dict[str, float | str] | None = None
    paper_result: float | None = None
