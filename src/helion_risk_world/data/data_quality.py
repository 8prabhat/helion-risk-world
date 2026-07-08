"""DEPRECATED — migrated to quanthelion.data.quality.report.RecordQualityReport.

The record-iteration quality checker (duplicates/missing/zero-volume/zero-OI/PIT/
future-label checks over an iterable of record-like objects) is now the reusable
quanthelion implementation. The hardcoded structural-zero-volume symbol set is now
config-driven via ``DataQualityThresholdsConfig.allow_zero_volume_symbols`` — this shim
pre-configures it with this project's original index-symbol allow-list so
``DataQualityReport().validate(records)`` behaves identically to before.

Original implementation backed up in .pre_quanthelion_migration_backup/.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from quanthelion.config.data_platform import DataQualityThresholdsConfig
from quanthelion.data.quality.report import RecordQualityReport as _RecordQualityReport

_STRUCTURAL_ZERO_VOLUME_SYMBOLS = frozenset({"BANKNIFTY", "NIFTY", "NIFTY50", "FINNIFTY", "INDIAVIX"})

_THRESHOLDS = DataQualityThresholdsConfig(
    allow_zero_volume_symbols=list(_STRUCTURAL_ZERO_VOLUME_SYMBOLS)
)


class DataQualityReport(_RecordQualityReport):
    """Validate timestamps, missing values, instrument/expiry mapping, rollover.

    See SPEC.md §20 stage 1.
    """

    def __init__(self) -> None:
        super().__init__(thresholds=_THRESHOLDS)

    def validate(self, records: Iterable[Any]) -> dict[str, Any]:
        return super().validate(records)


__all__ = ["DataQualityReport"]
