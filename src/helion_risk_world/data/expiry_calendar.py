"""DEPRECATED — migrated to quanthelion.calendars.expiry_calendar.

Same last-Thursday + holiday roll-back rule, same NSE holiday set (helion's copy
was one of two duplicates consolidated into quanthelion.calendars.exchange_calendar.
NSE_HOLIDAYS). ``dte()`` here drops the old ``year``/``month`` kwargs — no call
site in this repo used them (verified); it now takes an optional ``weekday`` kwarg
instead, defaulting to Thursday (unchanged behavior for all existing callers).

CAVEAT (unchanged from before migration): NSE shifted index-derivative expiry
weekdays during 2024-2025 — this theoretical calendar is a fallback; prefer real
expiries from quanthelion.ingestion.providers.upstox.UpstoxExpiries where possible.

Original implementation backed up in .pre_quanthelion_migration_backup/.
"""

from __future__ import annotations

from quanthelion.calendars.expiry_calendar import (
    ROLL_FLAG_DAYS,
    dte,
    dte_norm,
    monthly_expiry,
    roll_flag,
)

__all__ = ["monthly_expiry", "dte", "dte_norm", "roll_flag", "ROLL_FLAG_DAYS"]
