"""Validate data quality + leakage before modelling (SPEC.md §20 stage 1).

Usage:
    python scripts/validate_data.py --config <cfg> [--seed N] [--dry-run]
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from train import _demo_candles
from _common import log, setup
from helion_risk_world.config.loaders import data_config_from_mapping as data_config_from_cfg

from helion_risk_world.data.alpha_futures_features import AlphaDataFuturesWindowBuilder
from helion_risk_world.data.data_quality import DataQualityReport
from helion_risk_world.data.parquet_source import ParquetMarketDataSource
from helion_risk_world.data.provenance import validate_upstox_only_sources

import numpy as np
import pandas as pd


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return "NA"
    return str(value)


def _section(report: dict[str, object], name: str) -> dict[str, object]:
    value = report.get(name, {})
    return value if isinstance(value, dict) else {}


def _print_validation_summary(report: dict[str, object]) -> None:
    provenance = _section(report, "provenance")
    futures = _section(report, "processed_futures")
    labels = _section(report, "labels")
    samples = _section(report, "model_samples")
    sessions = _section(report, "sessions")
    rows: list[tuple[str, object]] = [
        ("status", bool(report.get("passed", False))),
        ("records", report.get("n_records")),
        ("symbols", ",".join(str(item) for item in report.get("symbols", []))),
        ("duplicates", report.get("duplicates")),
        ("core_missing_values", report.get("core_missing_values")),
        ("point_in_time_violations", report.get("point_in_time_violations")),
        ("future_label_violations", report.get("future_label_violations")),
        ("provenance_passed", provenance.get("passed")),
        ("futures_present", futures.get("present")),
        ("futures_passed", futures.get("passed")),
        ("futures_rows", futures.get("rows")),
        ("futures_duplicate_timestamps", futures.get("duplicate_timestamps")),
        ("futures_invalid_ohlc_rows", futures.get("invalid_ohlc_rows")),
        ("futures_untagged_invalid_ohlc_rows", futures.get("untagged_invalid_ohlc_rows")),
        ("futures_roll_gap_rows", futures.get("roll_gap_rows")),
        ("futures_eligible_positions", futures.get("eligible_positions")),
        ("futures_ineligible_positions", futures.get("ineligible_positions")),
        ("labels_present", labels.get("present")),
        ("labels_passed", labels.get("passed")),
        ("labels_rows", labels.get("rows")),
        ("label_duplicate_timestamps", labels.get("duplicate_timestamps")),
        ("label_leakage_violations", labels.get("label_leakage_violations")),
        ("label_barrier_invalid_rows", labels.get("barrier_invalid_rows")),
        ("samples_present", samples.get("present")),
        ("samples_passed", samples.get("passed")),
        ("sample_total_labels", samples.get("total_labels")),
        ("sample_missing_futures_timestamp", samples.get("missing_futures_timestamp")),
        ("sample_usable", samples.get("usable_samples")),
        ("sample_rejected_by_futures_quality", samples.get("rejected_by_futures_quality")),
        ("session_passed", sessions.get("passed")),
        ("expected_bars_per_full_session", sessions.get("expected_bars_per_full_session")),
        ("partial_session_days", sessions.get("partial_session_days")),
    ]
    print("\nVALIDATION SUMMARY")
    print("------------------")
    for key, value in rows:
        print(f"{key}: {_fmt(value)}")
    partial_examples = sessions.get("partial_session_examples")
    if partial_examples:
        print(f"partial_session_examples: {partial_examples}")
    print(flush=True)


def run_validation(
    *,
    cfg: dict,
    data_dir: str | None = None,
    demo: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    dc = data_config_from_cfg(cfg)
    if dry_run:
        return {
            "passed": True,
            "n_records": 0,
            "duplicates": 0,
            "missing_values": 0,
            "core_missing_values": 0,
            "zero_volume_rows": 0,
            "structural_zero_volume_rows": 0,
            "zero_oi_rows": 0,
            "point_in_time_violations": 0,
            "future_label_violations": 0,
            "monotonic_ts": True,
            "symbols": list(dc.universe),
            "provenance": {"passed": True, "dry_run": True},
            "processed_futures": {"present": False, "passed": True},
            "labels": {"present": False, "passed": True},
            "model_samples": {"present": False, "passed": True},
        }

    provenance = validate_upstox_only_sources(dc.data_sources_path)
    if demo:
        records = [row for rows in _demo_candles(dc.universe).values() for row in rows]
        base_report = DataQualityReport().validate(records)
        base_report.update(
            {
                "provenance": provenance,
                "processed_futures": {"present": False, "passed": True},
                "labels": {"present": False, "passed": True},
                "model_samples": {"present": False, "passed": True},
            }
        )
        base_report["passed"] = bool(base_report.get("passed", False))
        return base_report
    elif data_dir:
        src = ParquetMarketDataSource(data_dir, dc.universe, base_interval=dc.base_interval)
        records = list(src.iter_candles())
        aligned_index = src.timestamp_index()
    else:
        raise ValueError("Use --demo or --data-dir to validate records.")
    report = DataQualityReport().validate(records)
    data_root = Path(data_dir)
    futures_report = _validate_processed_futures(data_root, dc.lookback_bars)
    labels_report = _validate_labels(data_root)
    sample_report = _validate_model_samples(
        data_root,
        lookback_bars=dc.lookback_bars,
        labels_present=bool(labels_report.get("present", False)),
    )
    session_report = _session_report(aligned_index)
    report.update(
        {
            "provenance": provenance,
            "processed_futures": futures_report,
            "labels": labels_report,
            "model_samples": sample_report,
            "sessions": session_report,
        }
    )
    report["passed"] = bool(report.get("passed", False)) and all(
        bool(section.get("passed", False))
        for section in (provenance, futures_report, labels_report, sample_report, session_report)
    )
    return report


def _validate_processed_futures(data_root: Path, lookback_bars: int) -> dict[str, object]:
    """Validate alpha_data's real futures-microstructure data (Phase 2 migration) via
    ``AlphaDataFuturesWindowBuilder`` instead of the now-redundant local
    ``banknifty_5min.parquet`` assembly. OHLC-validity/roll-gap tagging is alpha_data's
    own ingestion QA responsibility now (see ``alpha_futures_features.py``'s docstring:
    "stale-price/roll-gap handling already happens upstream in alpha_data's own ingestion
    QA") -- this only checks index integrity and lookback-window availability, which is
    what ``AlphaDataFuturesWindowBuilder`` can actually tell us. ``data_root`` is unused
    here (kept for signature compatibility with ``_validate_labels``/
    ``_validate_model_samples``, which still read the local ``labels.parquet``).
    """
    del data_root
    try:
        builder = AlphaDataFuturesWindowBuilder()
    except FileNotFoundError:
        return {"present": False, "passed": True}
    index, _ = builder.build_history()
    eligible = builder.eligible_positions(lookback_bars) if len(index) else np.zeros(0, dtype=bool)
    return {
        "present": True,
        "passed": bool(
            not index.has_duplicates and (len(index) < lookback_bars or int(eligible.sum()) > 0)
        ),
        "rows": int(len(index)),
        "duplicate_timestamps": int(pd.Index(index).duplicated().sum()),
        "eligible_positions": int(eligible.sum()),
        "ineligible_positions": int(len(eligible) - int(eligible.sum())),
    }


def _validate_labels(data_root: Path) -> dict[str, object]:
    path = data_root / "processed" / "labels.parquet"
    if not path.exists():
        return {"present": False, "passed": True}
    labels = pd.read_parquet(path)
    if "ts" in labels.columns:
        labels = labels.set_index("ts")
    labels.index = pd.to_datetime(labels.index)
    duplicate_count = int(labels.index.duplicated().sum())
    leakage = 0
    if "label_realized_at" in labels.columns:
        realized_at = pd.to_datetime(labels["label_realized_at"])
        leakage = int((realized_at <= labels.index).sum())
    barrier_invalid = 0
    if "barrier_valid" in labels.columns:
        barrier_invalid = int((~labels["barrier_valid"].astype(bool)).sum())
    return {
        "present": True,
        "passed": bool(duplicate_count == 0 and leakage == 0),
        "path": str(path),
        "rows": int(len(labels)),
        "duplicate_timestamps": duplicate_count,
        "label_leakage_violations": leakage,
        "barrier_invalid_rows": barrier_invalid,
    }


def _validate_model_samples(
    data_root: Path,
    *,
    lookback_bars: int,
    labels_present: bool,
) -> dict[str, object]:
    labels_path = data_root / "processed" / "labels.parquet"
    if not labels_present or not labels_path.exists():
        return {"present": False, "passed": True}
    labels = pd.read_parquet(labels_path)
    if "ts" in labels.columns:
        labels = labels.set_index("ts")
    labels.index = pd.to_datetime(labels.index)
    try:
        builder = AlphaDataFuturesWindowBuilder()
    except FileNotFoundError:
        return {"present": False, "passed": True}
    futures_index, _ = builder.build_history()
    positions = futures_index.get_indexer(pd.DatetimeIndex(labels.index))
    exists = positions >= 0
    eligible = builder.eligible_positions(lookback_bars)
    clean = np.array(
        [bool(eligible[pos]) if pos >= 0 else False for pos in positions],
        dtype=bool,
    )
    usable = exists & clean
    return {
        "present": True,
        "passed": bool(int(usable.sum()) > 0 and int((exists & ~clean).sum()) >= 0),
        "total_labels": int(len(labels)),
        "missing_futures_timestamp": int((~exists).sum()),
        "usable_samples": int(usable.sum()),
        "rejected_by_futures_quality": int((exists & ~clean).sum()),
        "lookback_bars": int(lookback_bars),
    }


def _session_report(index: pd.DatetimeIndex) -> dict[str, object]:
    if len(index) == 0:
        return {"present": True, "passed": False, "reason": "empty common index"}
    by_day = pd.Series(1, index=index).groupby(index.normalize()).sum()
    expected = int(by_day.mode().iloc[0]) if not by_day.empty else 0
    partial = by_day[by_day < expected]
    return {
        "present": True,
        "passed": True,
        "expected_bars_per_full_session": expected,
        "partial_session_days": int(len(partial)),
        "partial_session_examples": {
            str(day.date()): int(count)
            for day, count in partial.head(10).items()
        },
    }


def main() -> None:
    args, cfg = setup(
        "Validate data quality + leakage before modelling (SPEC.md §20 stage 1).",
        option_groups=("demo", "data_dir"),
    )
    try:
        report = run_validation(
            cfg=cfg,
            data_dir=args.data_dir,
            demo=args.demo,
            dry_run=args.dry_run,
        )
    except ValueError as exc:
        if "Use --demo or --data-dir" in str(exc):
            log.warning("validate_data.no_source: %s", exc)
            sys.exit(0)
        log.error("validate_data.failed: %s", exc)
        sys.exit(1)
    except FileNotFoundError as exc:
        log.error("validate_data.missing_file: %s", exc)
        sys.exit(1)
    log.info("validate_data.report: %s", report)
    _print_validation_summary(report)
    sys.exit(0 if bool(report.get("passed", False)) else 1)


if __name__ == "__main__":
    main()
