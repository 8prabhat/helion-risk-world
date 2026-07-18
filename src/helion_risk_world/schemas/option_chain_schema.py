"""Option-chain schemas. Options are represented as an ATM-RELATIVE surface, never naively
flattened to hundreds of static columns (except as an explicit baseline). Market plane only.
See SPEC.md §16 (option-surface encoder) and §9.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OptionType(StrEnum):
    CALL = "CE"
    PUT = "PE"


class OptionContractSnapshot(BaseModel):
    """A single option contract observed point-in-time."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    underlying: str
    strike: float
    opt_type: OptionType
    ts: datetime
    available_at: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    oi: float
    d_oi: float | None = None
    iv: float | None = None
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    dte: float = Field(description="Days to expiry.")

    @model_validator(mode="after")
    def _pit(self) -> OptionContractSnapshot:
        if self.available_at > self.ts:
            raise ValueError("point-in-time violation: available_at > ts")
        return self


class StrikeRow(BaseModel):
    """Call+put values aligned to one ATM-relative strike token (e.g. ATM-2 .. ATM+2)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strike: float
    token: int = Field(description="Offset from ATM in strike steps; 0 == ATM.")
    is_masked: bool = Field(default=False, description="True if the strike was missing/illiquid.")
    call_oi: float | None = None
    put_oi: float | None = None
    call_d_oi: float | None = None
    put_d_oi: float | None = None
    call_volume: float | None = None
    put_volume: float | None = None
    call_iv: float | None = None
    put_iv: float | None = None
    call_delta: float | None = None
    put_delta: float | None = None
    call_gamma: float | None = None
    put_gamma: float | None = None
    call_theta: float | None = None
    put_theta: float | None = None
    call_vega: float | None = None
    put_vega: float | None = None


class OptionSurfaceSnapshot(BaseModel):
    """ATM-relative option surface at a point in time. The canonical encoder input."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    underlying: str
    ts: datetime
    available_at: datetime
    atm_strike: float
    dte: float
    strikes: list[StrikeRow] = Field(description="Ordered ATM-N .. ATM .. ATM+N rows.")
    # Derived surface features (computed by OptionSurfaceBuilder; market plane).
    pcr: float | None = None
    iv_skew: float | None = None
    gamma_concentration: float | None = None
    call_wall_strength: float | None = None
    put_wall_strength: float | None = None
    oi_wall_strength: float | None = None
    max_pain_proxy: float | None = None
    expiry_pressure: float | None = None
    atm_iv: float | None = None
    wing_iv: float | None = None
    # 2026-07-15: added after the feature IC diagnostic found these to be the single
    # strongest directional signal in a 124-feature evaluation (rank #1/#2 by |IC|
    # against both forward return and barrier-edge, fold-stable across all 5
    # chronological folds -- runs/feature_ic_report_expanded.csv). Locally computed
    # from the ATM strike row's own call_delta/put_delta as a fallback, overridden by
    # alpha_data's dedicated rolling-ATM greeks series when available (same pattern as
    # atm_iv above -- see AlphaDataAtmGreeksLoader).
    atm_call_delta: float | None = None
    atm_put_delta: float | None = None

    @model_validator(mode="after")
    def _pit_and_symmetry(self) -> OptionSurfaceSnapshot:
        if self.available_at > self.ts:
            raise ValueError("point-in-time violation: available_at > ts")
        tokens = [r.token for r in self.strikes]
        if tokens != sorted(tokens):
            raise ValueError("strike rows must be ordered by token ascending")
        return self
