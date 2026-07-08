"""Optional hooks to register HRW stages with Quanthelion's pipeline runner.

The real Quanthelion CLI is stage-based (``quanthelion run --stage <name>``), not the
``quanthelion run --project ... --task ...`` form sketched in the brief. The *primary, supported*
way to drive HRW is the project-local ``scripts/*.py`` entry points. These hooks are an **optional**
convenience: if a deployment wants HRW stages reachable from ``quanthelion run``, call
``register_hrw_stages()`` at import time of a Quanthelion stage-plugin module.

Kept deliberately thin and side-effect-free until explicitly invoked, so importing HRW never mutates
the framework's global registry.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from helion_risk_world.integration.quanthelion_adapter import (
    QUANTHELION_AVAILABLE,
    HelionIntegrationError,
    get_logger,
)

_log = get_logger("hrw.integration.registry")

# stage name -> callable(config) -> result
_HRW_STAGES: dict[str, Callable[[Any], Any]] = {}


def hrw_stage(name: str) -> Callable[[Callable[[Any], Any]], Callable[[Any], Any]]:
    """Decorator registering an HRW stage function under ``name``."""

    def _decorator(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
        if name in _HRW_STAGES:
            raise HelionIntegrationError(f"HRW stage already registered: {name}")
        _HRW_STAGES[name] = fn
        return fn

    return _decorator


def register_hrw_stages() -> dict[str, Callable[[Any], Any]]:
    """Expose HRW stages to a Quanthelion stage plugin. Returns the registry mapping.

    No-op-safe: if the framework is unavailable this simply returns the local mapping so tests can
    assert registration without a live pipeline.
    """
    if not QUANTHELION_AVAILABLE:  # pragma: no cover - environment dependent
        _log.warning("quanthelion unavailable; HRW stages registered locally only")
    return dict(_HRW_STAGES)
