from __future__ import annotations

from helion_risk_world.schemas.execution_schema import ExecutionState
from helion_risk_world.schemas.prediction_schema import ModelPrediction


class DataFreshnessMonitor:
    """Tracks live/paper input freshness and timestamp integrity for operational monitoring."""

    def __init__(
        self,
        *,
        max_market_staleness_seconds: float = 300.0,
        max_prediction_skew_seconds: float = 60.0,
        require_quotes: bool = False,
        alert_threshold: float = 0.1,
    ) -> None:
        self._max_market_staleness = float(max_market_staleness_seconds)
        self._max_prediction_skew = float(max_prediction_skew_seconds)
        self._require_quotes = bool(require_quotes)
        self._alert_threshold = float(alert_threshold)

        self._samples = 0
        self._failed = 0
        self._stale_market = 0
        self._prediction_skew = 0
        self._pit_violations = 0
        self._missing_quotes = 0
        self._max_seen_staleness = 0.0
        self._max_seen_skew = 0.0
        self._last_report: dict[str, float | str] = {"status": "ok", "data_alert": 0.0}

    def observe(
        self,
        prediction: ModelPrediction,
        market: ExecutionState,
    ) -> dict[str, float | str]:
        market_staleness = max(0.0, (market.ts - market.available_at).total_seconds())
        prediction_skew = abs((prediction.ts - market.ts).total_seconds())
        pit_violation = market.available_at > market.ts or market.available_at > prediction.ts
        missing_quotes = int(
            sum(value is None for value in (market.bid, market.ask, market.spread))
        )

        stale_market = market_staleness > self._max_market_staleness
        skew_alert = prediction_skew > self._max_prediction_skew
        quote_alert = self._require_quotes and missing_quotes > 0
        failed = pit_violation or stale_market or skew_alert or quote_alert

        self._samples += 1
        self._failed += int(failed)
        self._stale_market += int(stale_market)
        self._prediction_skew += int(skew_alert)
        self._pit_violations += int(pit_violation)
        self._missing_quotes += int(missing_quotes > 0)
        self._max_seen_staleness = max(self._max_seen_staleness, market_staleness)
        self._max_seen_skew = max(self._max_seen_skew, prediction_skew)

        report: dict[str, float | str] = {
            "status": "alert" if failed else "ok",
            "market_staleness_seconds": float(market_staleness),
            "prediction_skew_seconds": float(prediction_skew),
            "missing_quote_fields": float(missing_quotes),
            "stale_market": float(stale_market),
            "prediction_skew_alert": float(skew_alert),
            "pit_violation": float(pit_violation),
            "quote_alert": float(quote_alert),
            "data_alert": float(failed),
        }
        self._last_report = report
        return dict(report)

    def snapshot(self) -> dict[str, float | str]:
        if self._samples == 0:
            return {"samples": 0.0, "status": "ok", "data_alert": 0.0}
        failure_rate = self._failed / max(self._samples, 1)
        alert = self._pit_violations > 0 or failure_rate >= self._alert_threshold
        return {
            "samples": float(self._samples),
            "failed_checks": float(self._failed),
            "failure_rate": float(failure_rate),
            "stale_market_count": float(self._stale_market),
            "prediction_skew_count": float(self._prediction_skew),
            "pit_violation_count": float(self._pit_violations),
            "missing_quote_count": float(self._missing_quotes),
            "max_market_staleness_seconds": float(self._max_seen_staleness),
            "max_prediction_skew_seconds": float(self._max_seen_skew),
            "latest_market_staleness_seconds": float(
                self._last_report.get("market_staleness_seconds", 0.0)
            ),
            "latest_prediction_skew_seconds": float(
                self._last_report.get("prediction_skew_seconds", 0.0)
            ),
            "status": "alert" if alert else str(self._last_report.get("status", "ok")),
            "data_alert": float(alert),
        }


__all__ = ["DataFreshnessMonitor"]
