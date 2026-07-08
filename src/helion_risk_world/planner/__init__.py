"""Conservative MPC planner (SPEC.md §18). NO_TRADE is first-class. Lazy import."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

_LAZY: dict[str, str] = {
    "ActionAuditor": "action_auditor",
    "ActionSampler": "action_sampler",
    "MPCPlanner": "mpc_planner",
    "PositionSizer": "position_sizer",
    "RewardScorer": "reward_scorer",
}

__all__ = [
    "ActionAuditor",
    "ActionSampler",
    "MPCPlanner",
    "PositionSizer",
    "RewardScorer",
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
    from helion_risk_world.planner.action_auditor import ActionAuditor
    from helion_risk_world.planner.action_sampler import ActionSampler
    from helion_risk_world.planner.mpc_planner import MPCPlanner
    from helion_risk_world.planner.position_sizer import PositionSizer
    from helion_risk_world.planner.reward_scorer import RewardScorer
