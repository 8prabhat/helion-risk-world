"""DEPRECATED — migrated to quanthelion.transforms.contiguity.

This logic is now the reusable, single-source-of-truth implementation in
quanthelion (identical behavior; helion is a sibling editable install of
quanthelion). Kept as a re-export shim so existing imports keep working.
Original implementation backed up in .pre_quanthelion_migration_backup/.
"""

from __future__ import annotations

from quanthelion.transforms.contiguity import DEFAULT_MAX_GAP, contiguous_segment_ids

__all__ = ["DEFAULT_MAX_GAP", "contiguous_segment_ids"]
