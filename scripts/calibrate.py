"""Run the calibration gate on a saved HRW model artifact and real labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from _bootstrap import ensure_src_path

ensure_src_path()

import numpy as np
import pandas as pd
from helion_risk_world.config.loaders import (
    data_config_from_mapping as data_config_from_cfg,
    management_horizon_from_mapping as management_horizon_from_cfg,
)

from helion_risk_world.data.parquet_source import ParquetMarketDataSource
from helion_risk_world.evaluation.calibration_metrics import (
    CalibrationGate,
    compute as calibration_metrics_compute,
)
from helion_risk_world.evaluation.ml_metrics import classification_report
from helion_risk_world.evaluation.predictive_diagnostics import evaluate_predictive_outputs
from helion_risk_world.integration import get_logger, load_config
from helion_risk_world.runtime import (
    build_runtime_inputs,
    load_model_runtime,
    predict_snapshot,
)
from helion_risk_world.schemas.label_schema import (
    Barrier,
    horizon_return_column,
    horizon_volatility_column,
)
from helion_risk_world.training.split_manifest import ChronoSplitManifest

log = get_logger("hrw.calibrate")

_BARRIER_IDX = {Barrier.STOP: 0, Barrier.TARGET: 1, Barrier.TIMEOUT: 2}
_MIN_LABEL_SCHEMA_VERSION = 5


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, float):
        if value == float("inf") or value == float("-inf"):
            return str(value)
        return f"{value:.6g}"
    if value is None:
        return "NA"
    return str(value)


def _print_calibration_summary(report: dict[str, Any]) -> None:
    gate = report.get("gate", {})
    metrics = report.get("metrics", {})
    counts = report.get("sample_counts", {})
    classification = report.get("classification", {})
    if not isinstance(gate, dict):
        gate = {}
    if not isinstance(metrics, dict):
        metrics = {}
    if not isinstance(counts, dict):
        counts = {}
    if not isinstance(classification, dict):
        classification = {}
    reasons = gate.get("reasons", {})
    failure_reasons = {
        key: value
        for key, value in reasons.items()
        if isinstance(value, str) and value.startswith("FAIL")
    } if isinstance(reasons, dict) else {}
    rows = [
        ("status", gate.get("passed")),
        ("evaluation_split", report.get("evaluation_split")),
        ("management_horizon", gate.get("management_horizon")),
        ("evaluated_horizons", gate.get("evaluated_horizons")),
        ("total_label_rows", counts.get("total_label_rows")),
        ("selected_rows", counts.get("selected_rows")),
        ("usable_rows", counts.get("usable_rows")),
        ("skipped_rows", counts.get("skipped_rows")),
        ("barrier_usable_rows", counts.get("barrier_usable_rows")),
        ("rollout_mae", metrics.get("rollout_mae")),
        ("coverage_error", metrics.get("coverage_error")),
        ("interval_width", metrics.get("interval_width")),
        ("barrier_brier", metrics.get("barrier_brier")),
        ("barrier_ece", metrics.get("barrier_ece")),
        ("volatility_mae", metrics.get("volatility_mae")),
        ("barrier_accuracy", classification.get("accuracy")),
        ("barrier_macro_f1", classification.get("macro_f1")),
        ("gate_failures", failure_reasons if failure_reasons else "none"),
    ]
    print("\nCALIBRATION SUMMARY")
    print("-------------------")
    for key, value in rows:
        print(f"{key}: {_fmt(value)}")
    per_class = classification.get("per_class", {})
    if isinstance(per_class, dict):
        for name, stats in per_class.items():
            if not isinstance(stats, dict):
                continue
            print(
                "barrier_class={name} precision={precision} recall={recall} f1={f1} support={support}".format(
                    name=name,
                    precision=_fmt(stats.get("precision")),
                    recall=_fmt(stats.get("recall")),
                    f1=_fmt(stats.get("f1")),
                    support=_fmt(stats.get("support")),
                )
            )
    print(flush=True)


def run_calibration(
    config_path: Path,
    model_path: Path,
    labels_path: Path,
    data_dir: Path,
    *,
    coverage_tol: float = 0.05,
    brier_max: float = 0.25,
    ece_max: float = 0.10,
    max_samples: int = 1000,
    baseline_min_history: int = 32,
    calibration_split: str = "test",
    report_out: Path | None = None,
    dry_run: bool = False,
) -> bool:
    cfg = load_config(str(config_path))
    dc = data_config_from_cfg(cfg)
    cfg_horizon = management_horizon_from_cfg(cfg)

    if dry_run:
        log.info(
            "calibrate.dry_run: config=%s model=%s labels=%s data_dir=%s report_out=%s",
            config_path,
            model_path,
            labels_path,
            data_dir,
            report_out if report_out is not None else None,
        )
        _log_reasons(True, {"dry_run": "PASS dry-run only"})
        _print_calibration_summary(
            {
                "gate": {"passed": True, "reasons": {"dry_run": "PASS dry-run only"}},
                "sample_counts": {
                    "total_label_rows": 0,
                    "selected_rows": 0,
                    "usable_rows": 0,
                    "skipped_rows": 0,
                    "barrier_usable_rows": 0,
                },
                "metrics": {},
                "classification": {},
            }
        )
        return True

    runtime = load_model_runtime(model_path)
    log.info(
        "calibrate.runtime",
        model_kind=runtime.model_kind,
        enabled_encoders=list(runtime.enabled_encoders),
        disabled_optional_modules=list(runtime.disabled_optional_modules),
        split_manifest=runtime.split_manifest,
    )
    artifact_horizon = runtime.horizon_bars
    quantiles = runtime.quantiles
    if artifact_horizon != cfg_horizon:
        log.warning(
            "calibrate.horizon_mismatch: artifact_horizon=%s config_horizon=%s",
            artifact_horizon,
            cfg_horizon,
        )

    source = ParquetMarketDataSource(
        data_dir=str(data_dir),
        universe=dc.universe,
        base_interval=dc.base_interval,
    )
    inputs = build_runtime_inputs(
        dc,
        source,
        data_dir=data_dir,
        runtime=runtime,
    )

    labels = pd.read_parquet(labels_path)
    if "ts" in labels.columns:
        labels = labels.set_index("ts")
    labels.index = pd.to_datetime(labels.index)
    if (
        "label_schema_version" not in labels.columns
        or int(labels["label_schema_version"].max()) < _MIN_LABEL_SCHEMA_VERSION
    ):
        raise ValueError(
            "labels.parquet predates the point-in-time regime/weight schema; rerun scripts/label.py"
        )
    manifest = _split_manifest_from_runtime(runtime)
    total_label_rows = len(labels)
    if manifest is not None:
        labels = manifest.filter_labels(labels.sort_index(), calibration_split)
        log.info(
            "calibrate.split_filter",
            split=calibration_split,
            total_rows=total_label_rows,
            selected_rows=len(labels),
            train_end=manifest.train_end,
            val_end=manifest.val_end,
            val_start=manifest.val_start,
            test_start=manifest.test_start,
        )
    else:
        log.warning(
            "calibrate.split_manifest_missing: artifact has no split manifest; evaluating split=%s over all rows",
            calibration_split,
        )
        labels = labels.sort_index()
    if labels.empty:
        log.error(
            "calibrate.no_rows_for_split: split=%s total_rows=%s selected_rows=0",
            calibration_split,
            total_label_rows,
        )
        return False
    available = set(source.timestamps())
    candidate_rows: list[tuple[object, pd.Series]] = []
    skipped = 0
    for ts, row in labels.iterrows():
        if max_samples > 0 and len(candidate_rows) >= max_samples:
            break
        py_ts = pd.Timestamp(ts).to_pydatetime()
        if py_ts not in available:
            skipped += 1
            continue
        candidate_rows.append((py_ts, row))

    precomputed_inputs = None
    build_many = getattr(inputs, "build_many", None)
    if callable(build_many) and candidate_rows:
        precomputed_inputs = build_many([ts for ts, _ in candidate_rows])

    horizon_buffers: dict[int, dict[str, list[float] | list[list[float]] | list[object]]] = {}
    barrier_probs, barrier_labels = [], []
    management_epistemic, management_ood = [], []
    usable = barrier_usable = 0
    horizons = tuple(int(h) for h in runtime.available_horizons)
    management_horizon = int(runtime.horizon_bars)
    for py_ts, row in candidate_rows:
        snapshot = precomputed_inputs.get(py_ts) if precomputed_inputs is not None else None
        if snapshot is None:
            try:
                snapshot = inputs.build(py_ts)
            except ValueError:
                skipped += 1
                continue
        pred = predict_snapshot(
            runtime,
            snapshot,
            symbol=runtime.target_symbol or dc.universe[0],
            ts=py_ts,
        )
        used_row = False
        for horizon in horizons:
            try:
                hp = pred.horizon(horizon)
            except KeyError:
                continue
            realized_return = _label_horizon_return(row, horizon)
            realized_vol = _label_horizon_volatility(row, horizon, realized_return)
            if realized_return is None or realized_vol is None:
                continue
            bundle = horizon_buffers.setdefault(
                horizon,
                {
                    "pred_quantiles": [],
                    "realized": [],
                    "predicted_volatility": [],
                    "realized_volatility": [],
                    "regime_labels": [],
                },
            )
            cast_quantiles = bundle["pred_quantiles"]
            assert isinstance(cast_quantiles, list)
            cast_quantiles.append([hp.return_quantiles[q] for q in sorted(hp.return_quantiles)])
            cast_realized = bundle["realized"]
            assert isinstance(cast_realized, list)
            cast_realized.append(float(realized_return))
            cast_pred_vol = bundle["predicted_volatility"]
            assert isinstance(cast_pred_vol, list)
            cast_pred_vol.append(float(hp.volatility))
            cast_real_vol = bundle["realized_volatility"]
            assert isinstance(cast_real_vol, list)
            cast_real_vol.append(float(realized_vol))
            cast_regimes = bundle["regime_labels"]
            assert isinstance(cast_regimes, list)
            cast_regimes.append(row.get("regime"))
            used_row = True
        if not used_row:
            skipped += 1
            continue
        if _barrier_row_valid(row):
            barrier_probs.append([pred.barrier.stop, pred.barrier.target, pred.barrier.timeout])
            barrier_labels.append(_BARRIER_IDX[Barrier(row["barrier"])])
            barrier_usable += 1
        management_epistemic.append(float(pred.epistemic))
        management_ood.append(float(pred.ood_score))
        usable += 1

    if usable == 0:
        log.error("calibrate.no_usable_rows: skipped=%s", skipped)
        return False

    quantile_levels = np.asarray(sorted(quantiles), dtype=float)

    gate = CalibrationGate(
        coverage_tol=coverage_tol,
        barrier_brier_max=brier_max,
        barrier_ece_max=ece_max,
    )
    per_horizon_reports: dict[str, dict[str, object]] = {}
    overall_passed = True
    overall_reasons: dict[str, str] = {}
    for horizon in horizons:
        bundle = horizon_buffers.get(horizon)
        if not bundle:
            continue
        horizon_regime_labels = bundle["regime_labels"]
        assert isinstance(horizon_regime_labels, list)
        horizon_barrier_probs = (
            np.asarray(barrier_probs, dtype=float)
            if horizon == management_horizon and barrier_probs
            else None
        )
        horizon_barrier_labels = (
            np.asarray(barrier_labels, dtype=int)
            if horizon == management_horizon and barrier_labels
            else None
        )
        bundle_realized = np.asarray(bundle["realized"], dtype=float)
        barrier_aligned = (
            horizon_barrier_probs is not None
            and horizon_barrier_labels is not None
            and horizon_barrier_probs.shape[0] == bundle_realized.shape[0]
        )
        passed_h, reasons_h = gate.check(
            pred_quantiles=np.asarray(bundle["pred_quantiles"], dtype=float),
            realized=bundle_realized,
            barrier_probs=horizon_barrier_probs,
            barrier_labels=horizon_barrier_labels,
            regime_labels=(
                np.asarray(horizon_regime_labels, dtype=object)
                if any(label is not None for label in horizon_regime_labels)
                else None
            ),
            quantile_levels=quantile_levels,
        )
        diagnostic = evaluate_predictive_outputs(
            pred_quantiles=np.asarray(bundle["pred_quantiles"], dtype=float),
            realized=bundle_realized,
            barrier_probs=horizon_barrier_probs if barrier_aligned else None,
            barrier_labels=horizon_barrier_labels if barrier_aligned else None,
            quantile_levels=quantile_levels,
            predicted_volatility=np.asarray(bundle["predicted_volatility"], dtype=float),
            realized_volatility=np.asarray(bundle["realized_volatility"], dtype=float),
            regime_labels=(
                np.asarray(horizon_regime_labels, dtype=object)
                if any(label is not None for label in horizon_regime_labels)
                else None
            ),
            epistemic=(
                np.asarray(management_epistemic, dtype=float)
                if horizon == management_horizon
                else None
            ),
            ood_scores=(
                np.asarray(management_ood, dtype=float)
                if horizon == management_horizon
                else None
            ),
            baseline_min_history=baseline_min_history,
        )
        if horizon_barrier_probs is not None and horizon_barrier_labels is not None:
            barrier_metrics = calibration_metrics_compute(
                barrier_probs=horizon_barrier_probs,
                barrier_labels=horizon_barrier_labels,
            )
            diagnostic["metrics"].update(barrier_metrics)
            diagnostic["classification"] = classification_report(
                horizon_barrier_probs,
                horizon_barrier_labels,
                class_names=("stop", "target", "timeout"),
            )
            diagnostic["barrier_evaluation"] = {
                "samples": int(horizon_barrier_labels.shape[0]),
                "subset_only": not barrier_aligned,
                "metrics": barrier_metrics,
            }
        diagnostic["gate"] = {"passed": bool(passed_h), "reasons": reasons_h}
        per_horizon_reports[str(horizon)] = diagnostic
        overall_passed = overall_passed and passed_h
        for check, reason in reasons_h.items():
            overall_reasons[f"h{horizon}_{check}"] = reason

    if str(management_horizon) not in per_horizon_reports:
        log.error("calibrate.management_horizon_missing: horizon=%s usable=%s", management_horizon, usable)
        return False

    report = dict(per_horizon_reports[str(management_horizon)])
    report["gate"] = {
        "passed": bool(overall_passed),
        "reasons": overall_reasons,
        "management_horizon": management_horizon,
        "evaluated_horizons": [int(h) for h in per_horizon_reports],
        "per_horizon": {
            horizon: {
                "passed": bool(payload["gate"]["passed"]),
                "reasons": payload["gate"]["reasons"],
            }
            for horizon, payload in per_horizon_reports.items()
        },
    }
    report["per_horizon"] = per_horizon_reports
    report["evaluation_split"] = calibration_split
    report["split_manifest"] = manifest.to_metadata() if manifest is not None else None
    report["sample_counts"] = {
        "total_label_rows": total_label_rows,
        "selected_rows": len(labels),
        "usable_rows": usable,
        "skipped_rows": skipped,
        "per_horizon_rows": {
            horizon: int(payload["samples"])
            for horizon, payload in per_horizon_reports.items()
        },
        "barrier_usable_rows": barrier_usable,
    }
    log.info("calibrate.samples: usable=%s skipped=%s barrier_usable=%s", usable, skipped, barrier_usable)
    log.info(
        "calibrate.metrics: rollout_mae=%s coverage_error=%s barrier_brier=%s volatility_mae=%s",
        round(report["metrics"].get("rollout_mae", 0.0), 6),
        round(report["metrics"].get("coverage_error", 0.0), 6),
        round(report["metrics"].get("barrier_brier", 0.0), 6),
        round(report["metrics"].get("volatility_mae", 0.0), 6),
    )
    _log_reasons(overall_passed, overall_reasons)
    _print_calibration_summary(report)
    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log.info("calibrate.report_saved: %s", report_out)
    return overall_passed


def _log_reasons(passed: bool, reasons: dict[str, str]) -> None:
    log.info("calibrate.gate: %s", "PASS" if passed else "FAIL")
    for check, result in reasons.items():
        level = "info" if "PASS" in result or "SKIP" in result else "warning"
        getattr(log, level)("calibrate.%s: %s", check, result)


def _split_manifest_from_runtime(runtime) -> ChronoSplitManifest | None:
    payload = runtime.split_manifest
    if not isinstance(payload, dict):
        return None
    return ChronoSplitManifest.from_metadata(payload)


def _label_horizon_return(row: pd.Series, horizon: int) -> float | None:
    col = horizon_return_column(horizon)
    value = row.get(col)
    if value is not None and pd.notna(value):
        return float(value)
    if horizon == int(row.get("horizon_bars", horizon)):
        fallback = row.get("exit_return")
        if fallback is not None and pd.notna(fallback):
            return float(fallback)
    return None


def _barrier_row_valid(row: pd.Series) -> bool:
    barrier_valid = row.get("barrier_valid")
    if barrier_valid is not None and pd.notna(barrier_valid):
        return bool(barrier_valid)
    barrier = row.get("barrier")
    if barrier is None or pd.isna(barrier):
        return False
    return Barrier(barrier) is not Barrier.AMBIGUOUS


def _label_horizon_volatility(
    row: pd.Series,
    horizon: int,
    realized_return: float | None,
) -> float | None:
    col = horizon_volatility_column(horizon)
    value = row.get(col)
    if value is not None and pd.notna(value):
        return max(float(value), 1e-6)
    if horizon == int(row.get("horizon_bars", horizon)):
        fallback = row.get("realized_vol")
        if fallback is not None and pd.notna(fallback):
            return max(float(fallback), 1e-6)
    if realized_return is None:
        return None
    return max(abs(float(realized_return)), 1e-6)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--model-path", required=True, type=Path)
    p.add_argument("--labels-path", required=True, type=Path)
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--coverage-tol", type=float, default=0.05)
    p.add_argument("--brier-max", type=float, default=0.25)
    p.add_argument("--ece-max", type=float, default=0.10)
    p.add_argument("--max-samples", type=int, default=1000)
    p.add_argument(
        "--baseline-min-history",
        type=int,
        default=32,
        help="Minimum expanding history used by the causal baseline diagnostics.",
    )
    p.add_argument(
        "--calibration-split",
        choices=("train", "val", "test", "holdout", "all"),
        default="test",
        help="Which chronological split to score. Defaults to the strict held-out test slice.",
    )
    p.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help="Optional path for a detailed predictive diagnostics JSON report.",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    passed = run_calibration(
        config_path=args.config,
        model_path=args.model_path,
        labels_path=args.labels_path,
        data_dir=args.data_dir,
        coverage_tol=args.coverage_tol,
        brier_max=args.brier_max,
        ece_max=args.ece_max,
        max_samples=args.max_samples,
        baseline_min_history=args.baseline_min_history,
        calibration_split=args.calibration_split,
        report_out=args.report_out,
        dry_run=args.dry_run,
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
