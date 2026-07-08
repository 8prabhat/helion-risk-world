from __future__ import annotations

from pathlib import Path

from helion_risk_world.schemas.action_schema import FinalDecision


class DecisionLogger:
    """Persists the full FinalDecision audit record for every decision (SPEC.md §24, §28)."""

    def __init__(self, path: str | Path = "runs/paper_trading/decisions.jsonl") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, decision: FinalDecision) -> None:
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(decision.model_dump_json() + "\n")
