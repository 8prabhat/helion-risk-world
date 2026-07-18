"""Market-plane schemas. These carry MARKET features ONLY (no portfolio/account fields).

Point-in-time rule: every record carries ``ts`` (observation/decision time) and ``available_at``
(the earliest wall-clock time the value could be known). Builders assert ``available_at <= ts``.
See SPEC.md §5 (causal boundary) and §9 (schemas).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _PITModel(BaseModel):
    """Base for point-in-time market records."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    ts: datetime = Field(description="Observation/decision timestamp.")
    available_at: datetime = Field(description="Earliest time this value could be known.")

    @model_validator(mode="after")
    def _check_point_in_time(self) -> _PITModel:
        if self.available_at > self.ts:
            raise ValueError(
                f"point-in-time violation: available_at {self.available_at} > ts {self.ts}"
            )
        return self


class MarketCandle(_PITModel):
    """A single OHLCV(+OI) candle for an index or equity."""

    open: float
    high: float
    low: float
    close: float
    volume: float
    oi: float | None = None
    d_oi: float | None = None


class FuturesCandle(_PITModel):
    """A futures candle with derivatives-specific context."""

    expiry: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    oi: float
    d_oi: float | None = None
    basis: float | None = Field(default=None, description="future - spot.")
    calendar_spread: float | None = Field(default=None, description="near - next month.")


class Regime(StrEnum):
    TREND = "trend"
    RANGE = "range"
    EVENT = "event"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"
    CHOP = "chop"


class RegimeContext(_PITModel):
    """Slow-moving regime context (market plane)."""

    vix: float
    vix_pct: float = Field(ge=0.0, le=1.0, description="Rolling percentile of India VIX.")
    atm_iv: float | None = None
    iv_skew: float | None = None
    regime_probs: dict[Regime, float] | None = None


class EventType(StrEnum):
    NONE = "none"
    RBI = "rbi"
    FED = "fed"
    CPI = "cpi"
    BUDGET = "budget"
    ELECTION = "election"
    EXPIRY = "expiry"


class EventContext(_PITModel):
    """Event/calendar context (market plane). FII/DII is daily granularity only."""

    expiry_flag: bool = False
    event_day_flag: bool = False
    blackout_active: bool = False
    event_type: EventType = EventType.NONE
    fii_dii_net: float | None = None
    usdinr: float | None = None
    crude: float | None = None
    pc_oi_ratio: float | None = None   # put-call OI ratio from F&O bhavcopy
    basis_daily: float | None = None   # daily futures basis (fut-spot)/spot
    # Stabilized derivatives of the raw fields above (see
    # data/daily_context_loader.py::_add_derived_macro_columns). Rolling z-scores for
    # mean-reverting flow/sentiment variables (fii_dii_net, pc_oi_ratio); short-horizon
    # rate-of-change + rolling vol for trending macro drivers (usdinr, crude), since a
    # level z-score of a secularly-trending series mostly re-encodes the trend rather
    # than a genuine "how extreme is this" signal.
    fii_dii_net_z: float | None = None
    pc_oi_ratio_z: float | None = None
    usdinr_ret_5d: float | None = None
    crude_ret_5d: float | None = None
    usdinr_vol: float | None = None
    crude_vol: float | None = None
    # Realized-vol/VIX ratio (alpha_data pipelines/cross_asset_features.py::compute_vol_ratio) --
    # the model's own strongest known walk-forward signal (see configs/v1.yaml's volatility loss
    # weight comment). breadth_index_divergence (alpha_data pipelines/breadth_features.py) is the
    # mean-constituent-return-vs-index-return spread -- a distribution signal a price-only index
    # model can't see on its own.
    realized_vol_vix_ratio: float | None = None
    breadth_index_divergence: float | None = None
