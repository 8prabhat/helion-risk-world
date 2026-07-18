"""Path-aware label schemas (SPEC.md §11).

Triple-barrier labels are computed on the TRADED instrument (BankNIFTY futures continuous).
``LabelRecord.label_realized_at`` is strictly after ``ts`` — enforced so a label can never
become a feature. ``barrier`` uses the same 3-class vocabulary as the BarrierHead.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Barrier(StrEnum):
    STOP = "stop"       # lower barrier touched first
    TARGET = "target"   # upper barrier touched first
    TIMEOUT = "timeout" # neither barrier hit within H bars
    AMBIGUOUS = "ambiguous" # both barriers touched inside the same OHLC bar; order unknowable


BARRIER_SIGMA_COLUMN = "barrier_sigma"
BARRIER_STOP_RETURN_COLUMN = "barrier_stop_return"
BARRIER_TARGET_RETURN_COLUMN = "barrier_target_return"
BARRIER_STOP_MULT_COLUMN = "barrier_stop_mult"
BARRIER_TARGET_MULT_COLUMN = "barrier_target_mult"
BARRIER_VOL_SPAN_COLUMN = "barrier_vol_span"
BARRIER_COST_FLOOR_COLUMN = "barrier_cost_floor_frac"

# Meta-labeling columns (2026-07-18, see labeling/meta_labels.py): a momentum-based
# PRIMARY signal proposes a trade side; the META label is the binary, cost-aware
# question "would taking a trade in that direction, held via the same triple-barrier
# exit mechanics, have netted more than round-trip cost." Trains a dedicated
# MetaLabelHead as a trade-quality GATE on top of the existing barrier/quantile
# machinery, rather than replacing it outright (see docs/investigation_log.md).
PRIMARY_SIDE_COLUMN = "primary_side"
META_LABEL_COLUMN = "meta_label"


class LabelRecord(BaseModel):
    """Triple-barrier label for one entry bar. NEVER a feature (label_realized_at > ts)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    ts: datetime
    label_realized_at: datetime  # bar at which the label was resolved
    horizon_bars: int = Field(gt=0, description="Maximum holding horizon H.")
    barrier: Barrier
    barrier_valid: bool = Field(
        default=True,
        description="Whether the first-hit barrier class is reliable enough for supervision.",
    )
    entry_price: float | None = Field(
        default=None,
        gt=0.0,
        description="Trade entry price under the label convention.",
    )
    exit_price: float | None = Field(
        default=None,
        gt=0.0,
        description="Observed or implied exit price under the label convention.",
    )
    exit_return: float = Field(description="Realised futures return from entry to exit.")
    exit_t: int = Field(
        gt=0,
        description="Exit bar offset from the decision bar (1..H). Entry is never same-bar.",
    )
    realized_vol: float = Field(ge=0.0, description="Realised vol over [ts, exit_t].")
    barrier_sigma: float = Field(
        default=0.01,
        ge=0.0,
        description="Decision-time EWMA sigma used to size the triple barrier width.",
    )
    barrier_stop_return: float = Field(
        default=-0.02,
        le=0.0,
        description="Signed stop-barrier return from the next-bar entry price.",
    )
    barrier_target_return: float = Field(
        default=0.02,
        ge=0.0,
        description="Signed target-barrier return from the next-bar entry price.",
    )
    barrier_stop_mult: float = Field(default=2.0, gt=0.0, description="Stop barrier multiple of sigma.")
    barrier_target_mult: float = Field(default=2.0, gt=0.0, description="Target barrier multiple of sigma.")
    barrier_vol_span: int = Field(default=50, gt=0, description="EWMA span used for barrier sigma.")
    barrier_cost_floor_frac: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Minimum barrier half-width as a return fraction, floored at round-trip "
            "transaction cost (feature/label overhaul Phase 1) so sub-cost noise isn't "
            "labeled as a real directional win/loss. 0.0 = purely vol-scaled (old behavior)."
        ),
    )
    mae: float = Field(ge=0.0, description="Max adverse excursion fraction from entry price.")
    mfe: float = Field(ge=0.0, description="Max favourable excursion fraction from entry price.")
    uniqueness_weight: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Sample uniqueness ū_i; None until apply_uniqueness_weights is called.",
    )
    primary_side: int = Field(
        default=0, ge=-1, le=1,
        description=(
            "Momentum-based primary signal's proposed side at decision time: "
            "+1 long, -1 short, 0 = no signal (flat trailing momentum; excluded from "
            "meta-label training). See labeling/meta_labels.py."
        ),
    )
    meta_label: int | None = Field(
        default=None, ge=0, le=1,
        description=(
            "1 if taking a trade in primary_side's direction nets more than round-trip "
            "cost via the same triple-barrier exit path; 0 otherwise. None when "
            "primary_side == 0 (no trade proposed, no profitability question to label)."
        ),
    )

    @model_validator(mode="after")
    def _future_only(self) -> LabelRecord:
        if self.label_realized_at <= self.ts:
            raise ValueError(
                f"label leakage: label_realized_at {self.label_realized_at} <= ts {self.ts}"
            )
        if self.exit_t > self.horizon_bars:
            raise ValueError(
                f"exit_t {self.exit_t} must be <= horizon_bars {self.horizon_bars}"
            )
        if self.barrier is Barrier.AMBIGUOUS and self.barrier_valid:
            raise ValueError("ambiguous barrier labels must set barrier_valid=False")
        if self.barrier is not Barrier.AMBIGUOUS and not self.barrier_valid:
            raise ValueError("non-ambiguous barrier labels must set barrier_valid=True")
        if self.barrier_stop_return > 0.0:
            raise ValueError("barrier_stop_return must be <= 0")
        if self.barrier_target_return < 0.0:
            raise ValueError("barrier_target_return must be >= 0")
        return self


def horizon_return_column(horizon_bars: int) -> str:
    return f"horizon_return_{int(horizon_bars)}"


def horizon_volatility_column(horizon_bars: int) -> str:
    return f"horizon_vol_{int(horizon_bars)}"


def horizon_mae_column(horizon_bars: int) -> str:
    return f"horizon_mae_{int(horizon_bars)}"


def horizon_mfe_column(horizon_bars: int) -> str:
    return f"horizon_mfe_{int(horizon_bars)}"


def horizon_realized_at_column(horizon_bars: int) -> str:
    return f"horizon_realized_at_{int(horizon_bars)}"


__all__ = [
    "BARRIER_SIGMA_COLUMN",
    "BARRIER_STOP_MULT_COLUMN",
    "BARRIER_STOP_RETURN_COLUMN",
    "BARRIER_TARGET_MULT_COLUMN",
    "BARRIER_TARGET_RETURN_COLUMN",
    "BARRIER_VOL_SPAN_COLUMN",
    "META_LABEL_COLUMN",
    "PRIMARY_SIDE_COLUMN",
    "Barrier",
    "LabelRecord",
    "horizon_mae_column",
    "horizon_mfe_column",
    "horizon_realized_at_column",
    "horizon_return_column",
    "horizon_volatility_column",
]
