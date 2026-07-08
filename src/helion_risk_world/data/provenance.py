"""Data-source provenance checks for the Upstox-only research constraint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ALLOWED_PROVIDERS = frozenset({"upstox", "derived_from_upstox"})


def validate_upstox_only_sources(path: str | Path) -> dict[str, object]:
    """Validate that every configured source is Upstox or causally derived from Upstox."""
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"data source provenance config is missing: {source_path}")
    payload = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
    sources = payload.get("sources")
    if not isinstance(sources, dict):
        raise ValueError(f"{source_path} must contain a mapping at key 'sources'")

    rejected: dict[str, str] = {}
    accepted: dict[str, str] = {}
    for name, raw in sources.items():
        if not isinstance(raw, dict):
            rejected[str(name)] = "source entry is not a mapping"
            continue
        provider = str(raw.get("provider", "")).strip().lower()
        if provider not in ALLOWED_PROVIDERS:
            rejected[str(name)] = provider or "missing provider"
        else:
            accepted[str(name)] = provider
    if rejected:
        details = ", ".join(f"{name}={provider}" for name, provider in sorted(rejected.items()))
        raise ValueError(f"non-Upstox data sources configured: {details}")
    return {
        "passed": True,
        "path": str(source_path),
        "accepted_sources": accepted,
        "allowed_providers": sorted(ALLOWED_PROVIDERS),
    }


__all__ = ["ALLOWED_PROVIDERS", "validate_upstox_only_sources"]
