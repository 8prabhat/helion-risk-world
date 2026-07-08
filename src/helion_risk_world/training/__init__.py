"""Staged training (SPEC.md §20). Stage 2 is LOCAL — never imports msh_jepa. Lazy import."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

_LAZY: dict[str, str] = {
    "MarketStatePretrainer": "pretrain_market_state",
    "ForecasterTrainer": "train_forecaster",
    "PlannerEvaluator": "train_planner",
    "WorldModelTrainer": "train_world_model",
    "HRWTrainer": "trainer",
    "ForecastBatch": "trainer",
    "resolve_device": "trainer",
}

__all__ = [
    "ForecastBatch",
    "ForecasterTrainer",
    "HRWTrainer",
    "MarketStatePretrainer",
    "PlannerEvaluator",
    "WorldModelTrainer",
    "resolve_device",
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
    from helion_risk_world.training.pretrain_market_state import MarketStatePretrainer
    from helion_risk_world.training.train_forecaster import ForecasterTrainer
    from helion_risk_world.training.train_planner import PlannerEvaluator
    from helion_risk_world.training.train_world_model import WorldModelTrainer
    from helion_risk_world.training.trainer import ForecastBatch, HRWTrainer, resolve_device
