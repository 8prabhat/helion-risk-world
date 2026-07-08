"""Typed execution-cost configuration (Indian-market statutory charges; SPEC.md §15).

V1 uses conservative defaults. Numbers are illustrative placeholders — calibrate against your
broker's contract notes and the latest exchange/SEBI circulars before any paper/live use.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class InstrumentSpecConfig:
    """Execution contract metadata for one tradeable symbol family."""

    lot_size: float = 1.0
    tick_size: float | None = None
    margin_fraction: float = 1.0
    quantity_step: int = 1

    def __post_init__(self) -> None:
        if self.lot_size <= 0:
            raise ValueError("lot_size must be positive")
        if self.tick_size is not None and self.tick_size <= 0:
            raise ValueError("tick_size must be positive when provided")
        if not 0.0 < self.margin_fraction <= 1.0:
            raise ValueError("margin_fraction must be in (0, 1]")
        if self.quantity_step <= 0:
            raise ValueError("quantity_step must be positive")


def _default_instrument_specs() -> dict[str, InstrumentSpecConfig]:
    # BankNIFTY futures are the only live tradeable path in V1. These defaults keep execution
    # physically executable even when higher-level configs omit instrument metadata.
    banknifty = InstrumentSpecConfig(lot_size=30.0, tick_size=0.2, margin_fraction=0.25)
    return {
        "BANKNIFTY": banknifty,
        "BANKNIFTY_FUT": banknifty,
        "BANKNIFTY_FUT_CONTINUOUS": banknifty,
    }


@dataclass(frozen=True)
class CostModelConfig:
    """Conservative cost assumptions for the Execution Reality Layer.

    All rates are fractions unless stated. These are NOT authoritative tax values; treat as a
    config-driven starting point to be replaced with calibrated values (V2 uses own fill logs).
    """

    brokerage_per_order: float = 20.0      # flat per executed order (INR), conservative
    # securities transaction tax (options sell side, illustrative)
    stt_rate: float = 0.000625
    exchange_txn_rate: float = 0.00035
    gst_rate: float = 0.18                 # on (brokerage + exchange txn)
    sebi_rate: float = 0.000001
    stamp_duty_rate: float = 0.00003       # buy side
    # Microstructure assumptions (conservative when no live depth). Costs are NOTIONAL-relative so
    # they stay sane whether or not a per-unit price is known (V1 often only knows notional).
    half_spread_bps: float = 0.0003        # half-spread as fraction of notional when price unknown
    slippage_bps: float = 0.0002           # slippage as fraction of notional
    default_spread_ticks: float = 1.0      # retained for reference / future per-unit modelling
    tick_size: float = 0.05
    base_fill_prob: float = 0.95
    realism_high_cost_frac: float = 0.25   # cost <= 25% of edge -> high realism
    realism_low_cost_frac: float = 0.75    # cost >= 75% of edge -> low realism / block
    # Overnight NRML-carry financing (feature/label overhaul Phase 4a): a conservative
    # ~8bps/trading-night approximation. The prior intraday-only (~60 min) horizon never
    # held a position across a session close, so this cost had no effect; the new ~192-bar
    # (~2 trading day) management horizon routinely does, and the cost model previously had
    # no concept of holding-period cost at all (only statutory + spread/slippage).
    overnight_financing_rate_per_day: float = 0.0008
    instrument_specs: dict[str, InstrumentSpecConfig] = field(default_factory=_default_instrument_specs)

    def __post_init__(self) -> None:
        if not 0.0 <= self.realism_high_cost_frac < self.realism_low_cost_frac <= 1.0:
            raise ValueError("require 0 <= realism_high < realism_low <= 1")
