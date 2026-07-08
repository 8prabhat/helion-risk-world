"""Strategy profiles and runtime adapters."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

_LAZY: dict[str, str] = {
    "StrategyPlanner": "planner",
    "RiskProfileOverride": "profiles",
    "StrategyName": "profiles",
    "StrategyProfile": "profiles",
    "available_strategy_names": "profiles",
    "get_strategy_profile": "profiles",
}

__all__ = [
    "RiskProfileOverride",
    "StrategyName",
    "StrategyPlanner",
    "StrategyProfile",
    "available_strategy_names",
    "get_strategy_profile",
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
    from helion_risk_world.strategy.planner import StrategyPlanner
    from helion_risk_world.strategy.profiles import (
        RiskProfileOverride,
        StrategyName,
        StrategyProfile,
        available_strategy_names,
        get_strategy_profile,
    )
