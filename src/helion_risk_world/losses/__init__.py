"""Loss functions (SPEC.md §21). Lazy import — torch loaded on first use."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

_LAZY: dict[str, str] = {
    "BarrierLoss": "barrier_loss",
    "CalibrationLoss": "calibration_loss",
    "CompositeLoss": "composite_loss",
    "ForecasterLoss": "composite_loss",
    "PlannerLoss": "planner_loss",
    "QuantileLoss": "quantile_loss",
    "RiskLoss": "risk_loss",
    "UncertaintyLoss": "uncertainty_loss",
}

__all__ = [
    "BarrierLoss",
    "CalibrationLoss",
    "CompositeLoss",
    "ForecasterLoss",
    "PlannerLoss",
    "QuantileLoss",
    "RiskLoss",
    "UncertaintyLoss",
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
    from helion_risk_world.losses.barrier_loss import BarrierLoss
    from helion_risk_world.losses.calibration_loss import CalibrationLoss
    from helion_risk_world.losses.composite_loss import CompositeLoss, ForecasterLoss
    from helion_risk_world.losses.planner_loss import PlannerLoss
    from helion_risk_world.losses.quantile_loss import QuantileLoss
    from helion_risk_world.losses.risk_loss import RiskLoss
    from helion_risk_world.losses.uncertainty_loss import UncertaintyLoss
