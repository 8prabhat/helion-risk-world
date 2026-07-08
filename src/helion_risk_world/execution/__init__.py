"""Execution Reality Layer (SPEC.md §15)."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

_LAZY: dict[str, str] = {
    "ConservativeIndianCostModel": "cost_model",
    "CostModelProtocol": "cost_model",
    "ExecutionReality": "execution_reality",
    "EntryFeasibilityReport": "feasibility",
    "analyze_entry_feasibility": "feasibility",
    "assert_entry_feasible": "feasibility",
    "FillSimulator": "fill_simulator",
    "resolve_instrument_spec": "instrument_specs",
    "symbol_lookup_keys": "instrument_specs",
    "LatencyModel": "latency_model",
    "LiquidityModel": "liquidity_model",
    "build_candidate_order": "order_builder",
    "SlippageModel": "slippage_model",
}

__all__ = [
    "ConservativeIndianCostModel",
    "CostModelProtocol",
    "ExecutionReality",
    "EntryFeasibilityReport",
    "FillSimulator",
    "analyze_entry_feasibility",
    "assert_entry_feasible",
    "resolve_instrument_spec",
    "symbol_lookup_keys",
    "LatencyModel",
    "LiquidityModel",
    "build_candidate_order",
    "SlippageModel",
]


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = importlib.import_module(f"{__name__}.{module}")
    return getattr(mod, name)


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:
    from helion_risk_world.execution.cost_model import (
        ConservativeIndianCostModel,
        CostModelProtocol,
    )
    from helion_risk_world.execution.execution_reality import ExecutionReality
    from helion_risk_world.execution.feasibility import (
        EntryFeasibilityReport,
        analyze_entry_feasibility,
        assert_entry_feasible,
    )
    from helion_risk_world.execution.fill_simulator import FillSimulator
    from helion_risk_world.execution.instrument_specs import (
        resolve_instrument_spec,
        symbol_lookup_keys,
    )
    from helion_risk_world.execution.latency_model import LatencyModel
    from helion_risk_world.execution.liquidity_model import LiquidityModel
    from helion_risk_world.execution.order_builder import build_candidate_order
    from helion_risk_world.execution.slippage_model import SlippageModel
