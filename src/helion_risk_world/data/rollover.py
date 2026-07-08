"""DEPRECATED — migrated to quanthelion.transforms.rollover.

Kept as a re-export shim so existing imports keep working. Original
implementation backed up in .pre_quanthelion_migration_backup/.
"""

from __future__ import annotations

from quanthelion.transforms.rollover import (
    ROLL_GAP_THRESHOLD,
    count_roll_gaps,
    detect_roll_gaps,
    flag_and_clip,
)

__all__ = ["ROLL_GAP_THRESHOLD", "detect_roll_gaps", "flag_and_clip", "count_roll_gaps"]
