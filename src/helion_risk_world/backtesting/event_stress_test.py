from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from helion_risk_world.backtesting.backtest_engine import BacktestReport
from helion_risk_world.data.event_calendar import event_type_for, is_event_day
from quanthelion.calendars.expiry_calendar import monthly_expiry
from helion_risk_world.evaluation import trading_metrics as tm


class EventStressTest:
    """Event-day and expiry-day stress tests (SPEC.md §23)."""

    def run(self, report: BacktestReport) -> dict[str, Any]:
        """Return event-day, expiry-day, and regular-day performance slices."""
        step_timestamps = list(report.timestamps[1:]) if len(report.timestamps) > 1 else list(report.timestamps)
        return self.run_series(step_timestamps, report.step_returns)

    def run_series(
        self,
        timestamps: Sequence[Any],
        returns: Sequence[float],
    ) -> dict[str, Any]:
        if len(timestamps) != len(returns):
            raise ValueError("timestamps and returns must have the same length")
        if not timestamps:
            return {
                "event_days": _slice_summary([], []),
                "expiry_days": _slice_summary([], []),
                "regular_days": _slice_summary([], []),
                "by_event_type": {},
            }

        event_returns: list[float] = []
        event_ts: list[Any] = []
        expiry_returns: list[float] = []
        expiry_ts: list[Any] = []
        regular_returns: list[float] = []
        regular_ts: list[Any] = []
        by_event_type: dict[str, list[tuple[Any, float]]] = {}

        for ts, ret in zip(timestamps, returns, strict=True):
            trade_date = ts.date()
            is_expiry = trade_date == monthly_expiry(trade_date.year, trade_date.month)
            event_name = "expiry" if is_expiry else event_type_for(trade_date).value
            if is_expiry:
                expiry_ts.append(ts)
                expiry_returns.append(float(ret))
            if is_event_day(trade_date):
                event_ts.append(ts)
                event_returns.append(float(ret))
            if not is_expiry and not is_event_day(trade_date):
                regular_ts.append(ts)
                regular_returns.append(float(ret))
            if event_name != "none":
                by_event_type.setdefault(event_name, []).append((ts, float(ret)))

        return {
            "event_days": _slice_summary(event_ts, event_returns),
            "expiry_days": _slice_summary(expiry_ts, expiry_returns),
            "regular_days": _slice_summary(regular_ts, regular_returns),
            "by_event_type": {
                name: _slice_summary(
                    [ts for ts, _ in rows],
                    [ret for _, ret in rows],
                )
                for name, rows in sorted(by_event_type.items())
            },
        }


def _slice_summary(timestamps: Sequence[Any], returns: Sequence[float]) -> dict[str, float]:
    values = np.asarray(returns, dtype=float)
    if values.size == 0:
        return {
            "count": 0.0,
            "mean_return": 0.0,
            "total_return": 0.0,
            "sharpe": 0.0,
            "hit_rate": 0.0,
            "worst_return": 0.0,
        }
    equity = np.cumprod(1.0 + values)
    total_return = float(equity[-1] - 1.0)
    return {
        "count": float(values.size),
        "mean_return": float(values.mean()),
        "total_return": total_return,
        "sharpe": tm.sharpe(values, periods_per_year=tm.infer_periods_per_year(list(timestamps))),
        "hit_rate": tm.hit_rate(values),
        "worst_return": float(values.min()),
    }
