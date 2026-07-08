from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from helion_risk_world.integration.quanthelion_adapter import DatasetProtocol


class HRWWindowDataset:
    """Map-style dataset of (market window, label) pairs. Satisfies DatasetProtocol (LSP).

    May optionally wrap quanthelion.data.MultiSymbolSupervisedDataset. Yields MARKET-plane inputs;
    labels are kept separate and are future-only (SPEC.md §5, §9).
    """

    def __init__(
        self,
        index: list[Any],
        resolver: Callable[[Any], Any] | None = None,
        *,
        fields: Mapping[str, Callable[[Any], Any]] | None = None,
    ) -> None:
        self._index = index
        self._resolver = resolver
        self._fields = dict(fields or {})

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Any:
        item = self._index[idx]
        if self._resolver is not None:
            return self._resolver(item)
        if self._fields:
            return {name: fn(item) for name, fn in self._fields.items()}
        return item


# Structural conformance check (cheap, import-time documentation of intent).
_PROTO: type[DatasetProtocol] = HRWWindowDataset  # type: ignore[assignment]
