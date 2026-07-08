"""Latency model (SPEC.md §15, Day 5).

Returns an estimated round-trip latency in milliseconds (from the live book when known, else a
conservative default). SRP: latency only.
"""

from __future__ import annotations

from helion_risk_world.schemas.execution_schema import CandidateOrder, ExecutionState


class LatencyModel:
    """Estimate latency impact in milliseconds. SRP: latency only (SPEC.md §15)."""

    def __init__(self, default_latency_ms: float = 250.0) -> None:
        self._default = default_latency_ms

    def impact(self, order: CandidateOrder, market: ExecutionState) -> float:
        """Latency in ms; uses ``market.latency_ms`` when present, else the conservative default."""
        return float(market.latency_ms if market.latency_ms is not None else self._default)
