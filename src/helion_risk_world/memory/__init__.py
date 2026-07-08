"""Slow-state memory: regime, calibration, drift (SPEC.md §23). Lazy import."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

_LAZY: dict[str, str] = {
    "CalibrationMonitor": "calibration_memory",
    "DataFreshnessMonitor": "data_quality_memory",
    "DriftMonitor": "drift_memory",
    "RegimeMemory": "regime_memory",
}

__all__: list[str] = [
    "CalibrationMonitor",
    "DataFreshnessMonitor",
    "DriftMonitor",
    "RegimeMemory",
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
    from helion_risk_world.memory.calibration_memory import CalibrationMonitor
    from helion_risk_world.memory.data_quality_memory import DataFreshnessMonitor
    from helion_risk_world.memory.drift_memory import DriftMonitor
    from helion_risk_world.memory.regime_memory import RegimeMemory
