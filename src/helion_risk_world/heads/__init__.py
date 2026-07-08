"""Distributional output heads (SPEC.md §17). Lazy import — torch loaded on first use."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

_LAZY: dict[str, str] = {
    "BarrierHead": "barrier_head",
    "ExcursionBarrierHead": "excursion_barrier_head",
    "ExcursionHead": "excursion_head",
    "OODHead": "ood_head",
    "RegimeHead": "regime_head",
    "ReturnQuantileHead": "return_head",
    "UncertaintyHead": "uncertainty_head",
    "VolatilityHead": "volatility_head",
}

__all__ = [
    "BarrierHead",
    "ExcursionBarrierHead",
    "ExcursionHead",
    "OODHead",
    "RegimeHead",
    "ReturnQuantileHead",
    "UncertaintyHead",
    "VolatilityHead",
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
    from helion_risk_world.heads.barrier_head import BarrierHead
    from helion_risk_world.heads.excursion_barrier_head import ExcursionBarrierHead
    from helion_risk_world.heads.excursion_head import ExcursionHead
    from helion_risk_world.heads.ood_head import OODHead
    from helion_risk_world.heads.regime_head import RegimeHead
    from helion_risk_world.heads.return_head import ReturnQuantileHead
    from helion_risk_world.heads.uncertainty_head import UncertaintyHead
    from helion_risk_world.heads.volatility_head import VolatilityHead
