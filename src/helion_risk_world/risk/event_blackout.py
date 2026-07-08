from __future__ import annotations

from helion_risk_world.schemas.market_schema import EventContext, EventType


class EventBlackout:
    """Blocks/sizes-down around RBI/Fed/CPI/budget/election/expiry events. SRP: blackout only."""

    def is_active(self, event: EventContext) -> bool:
        return bool(
            event.blackout_active
            or event.event_day_flag
            or event.expiry_flag
            or event.event_type is not EventType.NONE
        )
