"""DEPRECATED — migrated to quanthelion.labels.barrier_context.

This logic is now the reusable, single-source-of-truth implementation in
quanthelion (Phase 2 alpha_data migration, verified byte-identical against this
file's original implementation on real data). Kept as a re-export shim so existing
imports keep working. Original implementation preserved in git history
(pre-migration commit).
"""

from __future__ import annotations

from quanthelion.labels.barrier_context import (
    BarrierContext,
    BarrierSpec,
    barrier_context_from_sigma,
    barrier_context_series,
    ewma_barrier_sigma,
)

__all__ = [
    "BarrierContext",
    "BarrierSpec",
    "barrier_context_from_sigma",
    "barrier_context_series",
    "ewma_barrier_sigma",
]
