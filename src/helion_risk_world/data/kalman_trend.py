"""DEPRECATED — migrated to quanthelion.data.transforms.kalman_trend.

Kept as a re-export shim so existing imports keep working. Original implementation
backed up in .pre_quanthelion_migration_backup/.
"""

from __future__ import annotations

from quanthelion.data.transforms.kalman_trend import local_linear_trend_filter

__all__ = ["local_linear_trend_filter"]
