"""Render a summary from a FinalDecision audit stream (SPEC.md §23, §28, Day 7).

Reads the JSONL audit file written by scripts/backtest.py (or paper trading) and aggregates the
decision mix, risk-shield reason codes, regime breakdown and expected reward/cost — the
human-readable explanation layer over the raw decisions.

Usage:
    python scripts/generate_report.py --config configs/v1.yaml [--audit <decisions.jsonl>]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from _common import log, setup
from helion_risk_world.reporting import build_report, write_json_report


def main() -> None:
    args, _cfg = setup(
        "Render a summary from a FinalDecision audit stream (SPEC.md §28).",
        option_groups=("audit", "out_path"),
    )
    audit_path = Path(getattr(args, "audit", None) or "runs/backtest/decisions.jsonl")
    if not audit_path.exists():
        log.warning(
            "generate_report.no_audit path=%s note=%s",
            audit_path,
            "Run `scripts/backtest.py --demo` first to produce an audit stream.",
        )
        sys.exit(0)
    payload = build_report(audit_path)
    out_path = getattr(args, "out_path", None)
    if out_path:
        saved = write_json_report(payload, out_path)
        log.info("generate_report.saved", path=str(saved), kind=payload.get("kind"))
    log.info("generate_report.summary", **payload)


if __name__ == "__main__":
    main()
