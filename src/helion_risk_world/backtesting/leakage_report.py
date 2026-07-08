"""Backtest leakage report (SPEC.md §11.10, §23, Day 7).

Re-runs the causal + temporal leakage invariants over a backtest window and FAILS LOUDLY on any
violation: (1) no portfolio field among the Market World feature names; (2) every feature row is
point-in-time (``available_at <= ts``); (3) every label is strictly in the future
(``label_realized_at > ts``). Returns a structured report; raises ``LeakageError`` on breach.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any

from helion_risk_world.data.leakage_checks import (
    LeakageError,
    assert_no_portfolio_in_market,
    assert_point_in_time,
)


class LeakageReport:
    """Re-run the leakage invariants over a backtest window (SPEC.md §11.10)."""

    def run(
        self,
        market_feature_names: Sequence[str],
        feature_rows: Iterable[Mapping[str, datetime]] | None = None,
        labels: Iterable[tuple[datetime, datetime]] | None = None,
    ) -> dict[str, Any]:
        """Validate causal + temporal separation. Raises ``LeakageError`` on any violation."""
        # (1) Causal-plane separation.
        assert_no_portfolio_in_market(market_feature_names)

        # (2) Point-in-time feature availability.
        n_feature_rows = 0
        if feature_rows is not None:
            rows = list(feature_rows)
            assert_point_in_time(rows)
            n_feature_rows = len(rows)

        # (3) Labels strictly in the future.
        n_labels = 0
        if labels is not None:
            for ts, realized_at in labels:
                if realized_at <= ts:
                    raise LeakageError(
                        f"label leakage: label_realized_at {realized_at} <= ts {ts} (SPEC.md §5)"
                    )
                n_labels += 1

        return {
            "passed": True,
            "n_market_features": len(market_feature_names),
            "n_feature_rows_checked": n_feature_rows,
            "n_labels_checked": n_labels,
        }
