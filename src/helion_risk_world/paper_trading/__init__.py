"""Paper trading (SPEC.md §24). Never silently executes; every decision audited. Lazy import."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

_LAZY: dict[str, str] = {
    "BrokerAdapterProtocol": "broker_adapter_interface",
    "DryRunBrokerAdapter": "broker_adapter_interface",
    "ExecutionRealityBrokerAdapter": "broker_adapter_interface",
    "PaperFill": "broker_adapter_interface",
    "DecisionLogger": "decision_logger",
    "ExecutionLogger": "execution_logger",
    "PaperTradingEngine": "paper_engine",
}

__all__ = [
    "BrokerAdapterProtocol",
    "DecisionLogger",
    "DryRunBrokerAdapter",
    "ExecutionRealityBrokerAdapter",
    "ExecutionLogger",
    "PaperFill",
    "PaperTradingEngine",
]


def __getattr__(name: str) -> Any:  # PEP 562 lazy attribute access
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = importlib.import_module(f"{__name__}.{module}")
    return getattr(mod, name)


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # eager imports for type checkers / IDEs only
    from helion_risk_world.paper_trading.broker_adapter_interface import (
        BrokerAdapterProtocol,
        DryRunBrokerAdapter,
        ExecutionRealityBrokerAdapter,
        PaperFill,
    )
    from helion_risk_world.paper_trading.decision_logger import DecisionLogger
    from helion_risk_world.paper_trading.execution_logger import ExecutionLogger
    from helion_risk_world.paper_trading.paper_engine import PaperTradingEngine
