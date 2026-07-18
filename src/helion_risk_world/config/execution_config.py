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
    # Cost-model audit (2026-07-18): the previous defaults for the next three rates were
    # wrong for what this repo actually trades (BANKNIFTY FUTURES), not just conservative.
    # Combined with `statutory()` charging every rate on EVERY order regardless of side
    # (see that method's docstring), the effective round-trip statutory cost came out to
    # ~21.7bps -- roughly 7x a realistic value -- which was silently eating most of the
    # model's already-small edge in every backtest before this date. Corrected to
    # illustrative CURRENT (2024+) NSE index-FUTURES rates (still approximate: verify
    # against your broker's live contract note before paper/live use, exchange/SEBI/state
    # rates do get revised):
    #   - STT on index futures is ~0.02% (0.0002) of notional, charged on the SELL leg
    #     ONLY -- the prior 0.000625 was the OPTIONS-premium STT rate (mislabeled in this
    #     field's old comment), roughly 3x too high even before the side-charging bug.
    #   - NSE F&O futures exchange transaction charge is ~0.0019% (0.000019) of notional
    #     -- the prior 0.00035 was roughly 18x too high (looks like a stale/placeholder
    #     equity-segment or pre-revision value).
    stt_rate: float = 0.0002               # index futures STT, SELL side only
    exchange_txn_rate: float = 0.000019    # NSE F&O futures transaction charge, both sides
    gst_rate: float = 0.18                 # on (brokerage + exchange txn)
    sebi_rate: float = 0.000001
    stamp_duty_rate: float = 0.00002       # ~0.002% of notional, BUY side only
    # Microstructure assumptions (conservative when no live depth). Costs are NOTIONAL-relative so
    # they stay sane whether or not a per-unit price is known (V1 often only knows notional).
    half_spread_bps: float = 0.0003        # half-spread as fraction of notional when price unknown
    slippage_bps: float = 0.0002           # slippage as fraction of notional
    default_spread_ticks: float = 1.0      # retained for reference / future per-unit modelling
    tick_size: float = 0.05
    base_fill_prob: float = 0.95
    realism_high_cost_frac: float = 0.25   # cost <= 25% of edge -> high realism
    realism_low_cost_frac: float = 0.75    # cost >= 75% of edge -> low realism / block
    # Overnight NRML-carry financing (feature/label overhaul Phase 4a; rate corrected
    # 2026-07-18). Standard NRML index-futures carrying at Indian discount brokers does
    # NOT incur an explicit daily interest/financing charge the way margin-funded equity
    # delivery (MTF) does -- SPAN/exposure margin is blocked (an opportunity cost, already
    # reflected via `margin_fraction` sizing) but no contract-note line item charges
    # interest for holding NRML futures across a session close. The original 0.0008
    # (8bps/night) had no cited basis and, for the ~192-bar (later H=48) management
    # horizon, was large enough to materially bias multi-day-hold economics. Reduced to a
    # small residual buffer for the (real but usually negligible) implicit cost of capital
    # tied up in blocked margin overnight -- verify against your own broker/segment if you
    # rely on this number rather than treating it as a conservative floor.
    overnight_financing_rate_per_day: float = 0.0001
    instrument_specs: dict[str, InstrumentSpecConfig] = field(default_factory=_default_instrument_specs)

    def __post_init__(self) -> None:
        if not 0.0 <= self.realism_high_cost_frac < self.realism_low_cost_frac <= 1.0:
            raise ValueError("require 0 <= realism_high < realism_low <= 1")
