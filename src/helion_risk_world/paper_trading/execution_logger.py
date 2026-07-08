from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


class ExecutionLogger:
    """Logs simulated fills/slippage for later Execution Reality calibration (SPEC.md §24)."""

    def __init__(self, path: str | Path = "runs/paper_trading/executions.jsonl") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, fill: Any) -> None:
        payload = _jsonable(asdict(fill) if is_dataclass(fill) else fill)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
