"""Runtime account-risk profile loading.

Scripts should resolve account size and risk limits from the YAML registry instead of
hardcoding them in multiple places.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from helion_risk_world.schemas.portfolio_schema import RiskProfile

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PATH = _PROJECT_ROOT / "configs" / "risk_profiles.yaml"


@dataclass(frozen=True)
class AccountRiskProfile:
    """Resolved account bootstrap capital plus the typed risk policy."""

    capital0: float
    risk: RiskProfile


def _resolve_path(path: str | Path | None) -> Path:
    if path is None:
        return _DEFAULT_PATH
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (_PROJECT_ROOT / candidate)


def load_account_risk_profile(
    name: str,
    *,
    path: str | Path | None = None,
) -> AccountRiskProfile:
    """Load one named account profile from ``configs/risk_profiles.yaml``."""
    profile_path = _resolve_path(path)
    with profile_path.open(encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}

    profiles = payload.get("profiles", {})
    if name not in profiles:
        available = ", ".join(sorted(profiles))
        raise KeyError(f"unknown risk profile '{name}' in {profile_path}; available: {available}")

    row = dict(profiles[name])
    capital0 = float(row.pop("capital0"))
    return AccountRiskProfile(capital0=capital0, risk=RiskProfile(name=name, **row))


__all__ = ["AccountRiskProfile", "load_account_risk_profile"]
