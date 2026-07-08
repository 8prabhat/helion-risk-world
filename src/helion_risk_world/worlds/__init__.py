"""Market World + Portfolio World + latent rollout. Lazy import — torch loaded on first use."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

_LAZY: dict[str, str] = {
    "MarketWorld": "market_world",
    "PortfolioWorld": "portfolio_world",
    "RolloutEngine": "rollout_engine",
}

__all__ = [
    "MarketWorld",
    "PortfolioWorld",
    "RolloutEngine",
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
    from helion_risk_world.worlds.market_world import MarketWorld
    from helion_risk_world.worlds.portfolio_world import PortfolioWorld
    from helion_risk_world.worlds.rollout_engine import RolloutEngine
