"""DEPRECATED — migrated to quanthelion.transforms.continuous_futures.

Kept as a re-export shim so existing imports keep working; the migrated version
is functionally identical (same signature, same expiry-driven roll-boundary logic)
and consumes quanthelion.calendars.expiry_calendar (the same holiday set that
helion_risk_world.data.expiry_calendar now also re-exports). Original
implementation backed up in .pre_quanthelion_migration_backup/.
"""

from __future__ import annotations

from quanthelion.transforms.continuous_futures import build_continuous, save_continuous

__all__ = ["build_continuous", "save_continuous"]
