"""Conservative Indian statutory + spread cost model (SPEC.md §15, Day 5).

Statutory rates are config-driven (``CostModelConfig``) and are illustrative placeholders —
calibrate against broker contract notes / SEBI circulars before paper/live use. SRP: cost
arithmetic only; knows nothing of the planner or model. DIP target: ``CostModelProtocol``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from helion_risk_world.config.execution_config import CostModelConfig
from helion_risk_world.schemas.execution_schema import CandidateOrder, ExecutionState


@runtime_checkable
class CostModelProtocol(Protocol):
    """Pluggable cost model (SPEC.md §26 DIP — planner depends on this, not a concrete broker)."""

    def spread_cost(self, order: CandidateOrder, market: ExecutionState) -> float: ...

    def statutory(self, order: CandidateOrder) -> float: ...


def round_trip_cost_frac(
    cfg: CostModelConfig, reference_notional: float = 1_500_000.0
) -> float:
    """Round-trip (entry + exit) transaction cost as a fraction of notional.

    Combines THIS project's own documented statutory + microstructure cost assumptions
    (``CostModelConfig`` — the same numbers ``ConservativeIndianCostModel.statutory()``
    already charges a real order) into a single scalar fraction, rather than borrowing a
    generic "25bps round trip" assumption from elsewhere. Used as a barrier-labeling cost
    floor (review finding, feature/label overhaul Phase 1): a barrier width below this
    fraction represents a price move too small to survive round-trip costs, so it
    shouldn't be labeled as a real directional win/loss.

    ``reference_notional`` converts the flat per-order ``brokerage_per_order`` fee into a
    fraction of notional; it defaults to roughly one BANKNIFTY_FUT lot (30 x ~50,000) at a
    representative price level. Brokerage is small relative to that notional, so this
    choice has minor sensitivity on the total.
    """
    brokerage_frac = cfg.brokerage_per_order / max(reference_notional, 1.0)
    gst_frac = cfg.gst_rate * (brokerage_frac + cfg.exchange_txn_rate)
    one_way = (
        brokerage_frac
        + cfg.stt_rate
        + cfg.exchange_txn_rate
        + gst_frac
        + cfg.sebi_rate
        + cfg.stamp_duty_rate
        + cfg.half_spread_bps
        + cfg.slippage_bps
    )
    return float(2.0 * one_way)


def overnight_financing_cost(
    cfg: CostModelConfig, notional: float, nights_held: int
) -> float:
    """NRML-carry financing cost (INR) for a position held across ``nights_held`` session closes.

    Feature/label overhaul Phase 4a: the cost model previously had no concept of
    holding-period cost at all -- only statutory charges (paid once, at entry/exit) and
    spread/slippage (paid once, at fill). Those are correct for the prior intraday-only
    (~60 min) horizon, where a position never crossed a session boundary. The new ~192-bar
    (~2 trading day) management horizon routinely does, and NRML futures margin financing
    is charged per night held, not per trade -- omitting it would understate backtest cost
    for exactly the trades this horizon extension is meant to evaluate.
    """
    if nights_held < 0:
        raise ValueError("nights_held must be >= 0")
    return float(cfg.overnight_financing_rate_per_day * nights_held * abs(notional))


class ConservativeIndianCostModel:
    """Conservative Indian statutory + spread cost model (V1). SRP: cost arithmetic only."""

    def __init__(self, cfg: CostModelConfig) -> None:
        self._cfg = cfg

    def spread_cost(self, order: CandidateOrder, market: ExecutionState) -> float:
        """Half-spread crossing cost (INR), notional-relative. Uses the live book when available."""
        if market.spread is not None and market.bid and market.ask:
            mid = (market.bid + market.ask) / 2.0
            half_frac = (market.spread / 2.0) / mid if mid > 0 else self._cfg.half_spread_bps
        else:
            half_frac = self._cfg.half_spread_bps
        return float(half_frac * abs(order.notional))

    def statutory(self, order: CandidateOrder) -> float:
        """Brokerage + STT + exchange txn + GST(on brokerage+exchange) + SEBI + stamp duty (INR)."""
        cfg = self._cfg
        notional = abs(order.notional)
        brokerage = cfg.brokerage_per_order
        stt = cfg.stt_rate * notional
        exchange = cfg.exchange_txn_rate * notional
        gst = cfg.gst_rate * (brokerage + exchange)
        sebi = cfg.sebi_rate * notional
        stamp = cfg.stamp_duty_rate * notional
        return float(brokerage + stt + exchange + gst + sebi + stamp)
