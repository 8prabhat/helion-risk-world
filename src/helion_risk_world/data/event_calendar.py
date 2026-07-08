"""NSE/RBI/macro event calendar — hybrid shim (event date table migrated; this
project's ``EventType`` enum wrapping stays local).

The reusable event-date table + priority resolution now live in
``quanthelion.data.calendars.event_calendar`` as plain string category labels (so that
module has no dependency on any downstream project's schema). This file wraps those
strings in this project's own ``EventType`` enum, preserving the original public API.

Original implementation backed up in .pre_quanthelion_migration_backup/.
"""

from __future__ import annotations

from datetime import date

from quanthelion.data.calendars.event_calendar import event_type_for as _event_type_for
from quanthelion.data.calendars.event_calendar import is_event_day

from helion_risk_world.schemas.market_schema import EventType


def event_type_for(dt: date) -> EventType:
    """Return the dominant event type for a given date, or NONE."""
    return EventType(_event_type_for(dt))


__all__ = ["event_type_for", "is_event_day", "EventType"]
