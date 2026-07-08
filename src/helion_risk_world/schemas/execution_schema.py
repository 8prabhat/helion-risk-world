"""Execution-plane schemas. Consumed by the Execution Reality Layer and Planner only.
See SPEC.md §15.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ExecutionRealism(StrEnum):
    HIGH = "high"      # trade likely survives costs and fill assumptions
    MEDIUM = "medium"  # reduce position or require higher edge
    LOW = "low"        # block trade


class ExecutionState(BaseModel):
    """Observed microstructure at a decision step (V1: live only; no historical depth)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    ts: datetime
    available_at: datetime
    bid: float | None = None
    ask: float | None = None
    spread: float | None = None
    depth: float | None = Field(default=None, description="Top-of-book depth if available (V2+).")
    latency_ms: float | None = None


class CandidateOrder(BaseModel):
    """A concrete order derived from a candidate action, fed to the Execution Reality Layer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    side: str  # buy / sell
    qty: float
    notional: float


class CostEstimate(BaseModel):
    """Execution Reality Layer output for one candidate order."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    spread_cost: float = 0.0
    statutory_fees: float = 0.0  # brokerage+STT+exchange+GST+SEBI+stamp duty
    slippage: float = 0.0
    total_cost: float
    fill_prob: float = Field(ge=0.0, le=1.0)
    partial_fill_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    reject_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    latency_ms: float = 0.0
    realism: ExecutionRealism
