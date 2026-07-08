"""Execution Reality Layer (SPEC.md §15, §27, Day 5)."""

from __future__ import annotations

from datetime import datetime

import pytest

from helion_risk_world.config.execution_config import CostModelConfig
from helion_risk_world.execution.execution_reality import ExecutionReality
from helion_risk_world.schemas import CandidateOrder, ExecutionState
from helion_risk_world.schemas.execution_schema import ExecutionRealism

TS = datetime(2026, 6, 25, 10, 0)


def _order(qty: float = 15, notional: float = 750_000) -> CandidateOrder:
    return CandidateOrder(symbol="BANKNIFTY", side="buy", qty=qty, notional=notional)


def _market(spread: float | None = 0.2, depth: float | None = None,
            latency: float | None = None) -> ExecutionState:
    return ExecutionState(symbol="BANKNIFTY", ts=TS, available_at=TS, bid=99.9, ask=100.1,
                          spread=spread, depth=depth, latency_ms=latency)


def test_cost_config_realism_band_ordering() -> None:
    with pytest.raises(ValueError):
        CostModelConfig(realism_high_cost_frac=0.8, realism_low_cost_frac=0.5)


def test_estimate_cost_decomposition_and_fill_probs() -> None:
    est = ExecutionReality(CostModelConfig()).estimate(_order(), _market())
    assert est.total_cost == pytest.approx(est.spread_cost + est.statutory_fees + est.slippage)
    assert est.spread_cost > 0 and est.statutory_fees > 0 and est.slippage > 0
    assert 0.0 <= est.fill_prob <= 1.0
    assert est.fill_prob + est.partial_fill_prob + est.reject_prob == pytest.approx(1.0)


def test_statutory_scales_with_notional() -> None:
    cm = ExecutionReality(CostModelConfig())
    small = cm.estimate(_order(notional=100_000), _market()).statutory_fees
    big = cm.estimate(_order(notional=1_000_000), _market()).statutory_fees
    assert big > small  # STT/exchange/GST/SEBI/stamp all scale with notional


def test_realism_microstructure_high_when_liquid() -> None:
    # Deep book -> liquidity 1.0, base fill 0.95 -> HIGH (no edge supplied).
    est = ExecutionReality(CostModelConfig()).estimate(_order(qty=10), _market(depth=1000))
    assert est.realism is ExecutionRealism.HIGH


def test_realism_low_when_illiquid() -> None:
    est = ExecutionReality(CostModelConfig()).estimate(_order(qty=1000), _market(depth=10))
    assert est.realism is ExecutionRealism.LOW  # depth/qty tiny -> liquidity < 0.3


def test_realism_edge_aware_blocks_thin_edge() -> None:
    er = ExecutionReality(CostModelConfig())
    est = er.estimate(_order(), _market(depth=1000))
    # Edge smaller than total cost -> burden >= low band -> LOW.
    thin = er.estimate(_order(), _market(depth=1000), expected_edge=est.total_cost * 0.5)
    fat = er.estimate(_order(), _market(depth=1000), expected_edge=est.total_cost * 100)
    assert thin.realism is ExecutionRealism.LOW
    assert fat.realism is ExecutionRealism.HIGH


def test_latency_default_and_override() -> None:
    er = ExecutionReality(CostModelConfig())
    assert er.estimate(_order(), _market()).latency_ms == 250.0
    assert er.estimate(_order(), _market(latency=12.0)).latency_ms == 12.0
