"""DEPRECATED — migrated to quanthelion.data.features.daily_context.

Kept as a re-export shim so existing imports keep working. Original implementation
backed up in .pre_quanthelion_migration_backup/.
"""

from __future__ import annotations

from quanthelion.data.features.daily_context import (
    DailyContextLoader,
    _ZSCORE_MIN_PERIODS,
    _ZSCORE_WINDOW_DAYS,
)

__all__ = ["DailyContextLoader", "_ZSCORE_MIN_PERIODS", "_ZSCORE_WINDOW_DAYS"]
