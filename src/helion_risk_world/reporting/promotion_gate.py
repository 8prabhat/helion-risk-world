from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PromotionThresholds:
    require_calibration_pass: bool = True
    require_paper_report: bool = False
    require_ab_gate: bool = False
    min_total_return: float = 0.0
    min_annualized_return: float = 0.15
    min_sharpe: float = 1.0
    min_profit_factor: float = 1.1
    max_drawdown: float = 0.15
    min_hit_rate: float = 0.45
    min_trades: int = 10
    max_paper_drift_score: float = 0.3
    max_paper_data_failure_rate: float = 0.1
    max_paper_reject_rate: float = 0.2
    fail_workflow_on_reject: bool = False

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "PromotionThresholds":
        values = payload or {}
        return cls(
            require_calibration_pass=bool(
                values.get("require_calibration_pass", cls.require_calibration_pass)
            ),
            require_paper_report=bool(
                values.get("require_paper_report", cls.require_paper_report)
            ),
            require_ab_gate=bool(values.get("require_ab_gate", cls.require_ab_gate)),
            min_total_return=float(values.get("min_total_return", cls.min_total_return)),
            min_annualized_return=float(
                values.get("min_annualized_return", cls.min_annualized_return)
            ),
            min_sharpe=float(values.get("min_sharpe", cls.min_sharpe)),
            min_profit_factor=float(values.get("min_profit_factor", cls.min_profit_factor)),
            max_drawdown=float(values.get("max_drawdown", cls.max_drawdown)),
            min_hit_rate=float(values.get("min_hit_rate", cls.min_hit_rate)),
            min_trades=int(values.get("min_trades", cls.min_trades)),
            max_paper_drift_score=float(
                values.get("max_paper_drift_score", cls.max_paper_drift_score)
            ),
            max_paper_data_failure_rate=float(
                values.get(
                    "max_paper_data_failure_rate",
                    cls.max_paper_data_failure_rate,
                )
            ),
            max_paper_reject_rate=float(
                values.get("max_paper_reject_rate", cls.max_paper_reject_rate)
            ),
            fail_workflow_on_reject=bool(
                values.get("fail_workflow_on_reject", cls.fail_workflow_on_reject)
            ),
        )


def evaluate_promotion(
    *,
    calibration: dict[str, Any] | None,
    report: dict[str, Any] | None,
    paper: dict[str, Any] | None = None,
    incumbent_calibration: dict[str, Any] | None = None,
    incumbent_report: dict[str, Any] | None = None,
    incumbent_paper: dict[str, Any] | None = None,
    thresholds: PromotionThresholds,
) -> dict[str, object]:
    """Evaluate calibration + backtest outputs into a clear promotion decision."""
    checks: dict[str, str] = {}
    candidate = _extract_candidate(report)
    metrics = _extract_metrics(report)
    incumbent_candidate = _extract_candidate(incumbent_report)
    incumbent_metrics = _extract_metrics(incumbent_report)

    if thresholds.require_calibration_pass:
        passed = bool((calibration or {}).get("gate", {}).get("passed", False))
        checks["calibration_gate"] = "PASS" if passed else "FAIL calibration gate rejected"
    else:
        checks["calibration_gate"] = "SKIP calibration gate disabled"

    if not metrics:
        checks["backtest_metrics"] = "FAIL no backtest metrics available for promotion"
        return _decision(
            checks,
            candidate,
            metrics,
            incumbent_candidate,
            incumbent_metrics,
            thresholds,
        )

    _numeric_check(
        checks,
        "total_return",
        float(metrics.get("total_return", 0.0)),
        ">=",
        thresholds.min_total_return,
    )
    _numeric_check(
        checks,
        "annualized_return",
        float(metrics.get("annualized_return", 0.0)),
        ">=",
        thresholds.min_annualized_return,
    )
    _numeric_check(
        checks,
        "sharpe",
        float(metrics.get("sharpe", 0.0)),
        ">=",
        thresholds.min_sharpe,
    )
    _numeric_check(
        checks,
        "profit_factor",
        float(metrics.get("profit_factor", 0.0)),
        ">=",
        thresholds.min_profit_factor,
    )
    _numeric_check(
        checks,
        "hit_rate",
        float(metrics.get("hit_rate", 0.0)),
        ">=",
        thresholds.min_hit_rate,
    )
    _numeric_check(
        checks,
        "max_drawdown",
        float(metrics.get("max_drawdown", 0.0)),
        "<=",
        thresholds.max_drawdown,
    )
    _numeric_check(
        checks,
        "n_trades",
        float(metrics.get("n_trades", 0.0)),
        ">=",
        float(thresholds.min_trades),
    )
    _paper_checks(checks, paper, thresholds)
    _ab_checks(
        checks,
        calibration=calibration,
        report=report,
        paper=paper,
        incumbent_calibration=incumbent_calibration,
        incumbent_report=incumbent_report,
        incumbent_paper=incumbent_paper,
        thresholds=thresholds,
    )
    return _decision(
        checks,
        candidate,
        metrics,
        incumbent_candidate,
        incumbent_metrics,
        thresholds,
    )


def _decision(
    checks: dict[str, str],
    candidate: str | None,
    metrics: dict[str, float],
    incumbent_candidate: str | None,
    incumbent_metrics: dict[str, float],
    thresholds: PromotionThresholds,
) -> dict[str, object]:
    passed = all(result.startswith("PASS") or result.startswith("SKIP") for result in checks.values())
    return {
        "passed": passed,
        "status": "pass" if passed else "fail",
        "candidate": candidate,
        "checks": checks,
        "metrics": metrics,
        "incumbent": {
            "candidate": incumbent_candidate,
            "metrics": incumbent_metrics,
        },
        "thresholds": {
            "require_calibration_pass": thresholds.require_calibration_pass,
            "require_paper_report": thresholds.require_paper_report,
            "require_ab_gate": thresholds.require_ab_gate,
            "min_total_return": thresholds.min_total_return,
            "min_annualized_return": thresholds.min_annualized_return,
            "min_sharpe": thresholds.min_sharpe,
            "min_profit_factor": thresholds.min_profit_factor,
            "max_drawdown": thresholds.max_drawdown,
            "min_hit_rate": thresholds.min_hit_rate,
            "min_trades": thresholds.min_trades,
            "max_paper_drift_score": thresholds.max_paper_drift_score,
            "max_paper_data_failure_rate": thresholds.max_paper_data_failure_rate,
            "max_paper_reject_rate": thresholds.max_paper_reject_rate,
            "fail_workflow_on_reject": thresholds.fail_workflow_on_reject,
        },
    }


def _extract_candidate(report: dict[str, Any] | None) -> str | None:
    if not report:
        return None
    if report.get("kind") == "single_backtest":
        return str(report.get("strategy")) if report.get("strategy") is not None else None
    if report.get("kind") in {"strategy_comparison", "walk_forward"}:
        best = report.get("best_strategy")
        return str(best) if best is not None else None
    return None


def _extract_metrics(report: dict[str, Any] | None) -> dict[str, float]:
    if not report:
        return {}
    if report.get("kind") == "single_backtest":
        metrics = _coerce_metrics(report.get("summary"))
        return metrics or _coerce_metrics(report)
    if report.get("kind") == "strategy_comparison":
        strategies = report.get("strategies", [])
        if strategies:
            return _coerce_metrics(strategies[0])
        return {}
    if report.get("kind") == "walk_forward":
        strategies = report.get("strategies", [])
        if strategies:
            return _coerce_metrics((strategies[0] or {}).get("out_of_sample"))
        return {}
    return {}


def _coerce_metrics(payload: Any) -> dict[str, float]:
    if not isinstance(payload, dict):
        return {}
    keys = (
        "n_trades",
        "total_return",
        "annualized_return",
        "sharpe",
        "profit_factor",
        "max_drawdown",
        "hit_rate",
    )
    out: dict[str, float] = {}
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _paper_checks(
    checks: dict[str, str],
    paper: dict[str, Any] | None,
    thresholds: PromotionThresholds,
) -> None:
    if paper is None:
        if thresholds.require_paper_report:
            checks["paper_report"] = "FAIL paper report required for promotion"
        else:
            checks["paper_report"] = "SKIP paper report not required"
        return
    if paper.get("kind") != "paper_trading":
        checks["paper_report"] = "FAIL invalid paper report payload"
        return
    checks["paper_report"] = "PASS paper report available"
    metrics = _extract_paper_metrics(paper)
    _numeric_check(
        checks,
        "paper_reject_rate",
        metrics.get("reject_rate", 0.0),
        "<=",
        thresholds.max_paper_reject_rate,
    )
    _numeric_check(
        checks,
        "paper_data_failure_rate",
        metrics.get("failure_rate", 0.0),
        "<=",
        thresholds.max_paper_data_failure_rate,
    )
    _numeric_check(
        checks,
        "paper_drift_score",
        metrics.get("drift_score", 0.0),
        "<=",
        thresholds.max_paper_drift_score,
    )


def _ab_checks(
    checks: dict[str, str],
    *,
    calibration: dict[str, Any] | None,
    report: dict[str, Any] | None,
    paper: dict[str, Any] | None,
    incumbent_calibration: dict[str, Any] | None,
    incumbent_report: dict[str, Any] | None,
    incumbent_paper: dict[str, Any] | None,
    thresholds: PromotionThresholds,
) -> None:
    if (
        incumbent_calibration is None
        and incumbent_report is None
        and incumbent_paper is None
    ):
        if thresholds.require_ab_gate:
            checks["ab_gate"] = "FAIL incumbent comparison required but no incumbent artifacts provided"
        else:
            checks["ab_gate"] = "SKIP no incumbent artifacts provided"
        return

    checks["ab_gate"] = "PASS incumbent artifacts available"
    _ab_calibration_checks(checks, calibration, incumbent_calibration)
    _ab_backtest_checks(checks, report, incumbent_report)
    _ab_paper_checks(checks, paper, incumbent_paper)


def _ab_calibration_checks(
    checks: dict[str, str],
    candidate: dict[str, Any] | None,
    incumbent: dict[str, Any] | None,
) -> None:
    if incumbent is None:
        checks["ab_calibration"] = "SKIP no incumbent calibration report"
        return
    candidate_passed = bool((candidate or {}).get("gate", {}).get("passed", False))
    incumbent_passed = bool((incumbent or {}).get("gate", {}).get("passed", False))
    if incumbent_passed and not candidate_passed:
        checks["ab_calibration"] = "FAIL candidate calibration gate regressed vs incumbent"
        return
    checks["ab_calibration"] = "PASS calibration gate no worse than incumbent"
    candidate_metrics = (candidate or {}).get("metrics", {}) if isinstance(candidate, dict) else {}
    incumbent_metrics = (incumbent or {}).get("metrics", {}) if isinstance(incumbent, dict) else {}
    _ab_numeric_check(
        checks,
        "ab_coverage_error",
        candidate_metrics,
        incumbent_metrics,
        "coverage_error",
        "<=",
    )
    _ab_numeric_check(
        checks,
        "ab_barrier_brier",
        candidate_metrics,
        incumbent_metrics,
        "barrier_brier",
        "<=",
    )


def _ab_backtest_checks(
    checks: dict[str, str],
    candidate: dict[str, Any] | None,
    incumbent: dict[str, Any] | None,
) -> None:
    incumbent_metrics = _extract_metrics(incumbent)
    if not incumbent_metrics:
        checks["ab_backtest"] = "SKIP no incumbent backtest metrics"
        return
    checks["ab_backtest"] = "PASS incumbent backtest metrics available"
    candidate_metrics = _extract_metrics(candidate)
    _ab_numeric_check(checks, "ab_total_return", candidate_metrics, incumbent_metrics, "total_return", ">=")
    _ab_numeric_check(
        checks,
        "ab_annualized_return",
        candidate_metrics,
        incumbent_metrics,
        "annualized_return",
        ">=",
    )
    _ab_numeric_check(checks, "ab_sharpe", candidate_metrics, incumbent_metrics, "sharpe", ">=")
    _ab_numeric_check(
        checks,
        "ab_profit_factor",
        candidate_metrics,
        incumbent_metrics,
        "profit_factor",
        ">=",
    )
    _ab_numeric_check(checks, "ab_hit_rate", candidate_metrics, incumbent_metrics, "hit_rate", ">=")
    _ab_numeric_check(
        checks,
        "ab_max_drawdown",
        candidate_metrics,
        incumbent_metrics,
        "max_drawdown",
        "<=",
    )


def _ab_paper_checks(
    checks: dict[str, str],
    candidate: dict[str, Any] | None,
    incumbent: dict[str, Any] | None,
) -> None:
    if incumbent is None:
        checks["ab_paper"] = "SKIP no incumbent paper report"
        return
    incumbent_metrics = _extract_paper_metrics(incumbent)
    if not incumbent_metrics:
        checks["ab_paper"] = "SKIP no incumbent paper metrics"
        return
    checks["ab_paper"] = "PASS incumbent paper metrics available"
    candidate_metrics = _extract_paper_metrics(candidate)
    _ab_numeric_check(
        checks,
        "ab_paper_reject_rate",
        candidate_metrics,
        incumbent_metrics,
        "reject_rate",
        "<=",
    )
    _ab_numeric_check(
        checks,
        "ab_paper_data_failure_rate",
        candidate_metrics,
        incumbent_metrics,
        "failure_rate",
        "<=",
    )
    _ab_numeric_check(
        checks,
        "ab_paper_drift_score",
        candidate_metrics,
        incumbent_metrics,
        "drift_score",
        "<=",
    )


def _extract_paper_metrics(paper: dict[str, Any] | None) -> dict[str, float]:
    if not isinstance(paper, dict):
        return {}
    monitor = paper.get("monitor_summary", {})
    if not isinstance(monitor, dict):
        return {}
    live = monitor.get("live_stats", {})
    drift = monitor.get("drift", {})
    data_quality = monitor.get("data_quality", {})
    if not isinstance(live, dict):
        live = {}
    if not isinstance(drift, dict):
        drift = {}
    if not isinstance(data_quality, dict):
        data_quality = {}
    return {
        "reject_rate": _to_float(live.get("reject_rate")),
        "failure_rate": _to_float(data_quality.get("failure_rate")),
        "drift_score": _to_float(drift.get("drift_score")),
    }


def _ab_numeric_check(
    checks: dict[str, str],
    name: str,
    candidate_metrics: dict[str, float],
    incumbent_metrics: dict[str, float],
    key: str,
    op: str,
) -> None:
    if key not in incumbent_metrics:
        checks[name] = f"SKIP incumbent missing {key}"
        return
    if key not in candidate_metrics:
        checks[name] = f"FAIL candidate missing {key}"
        return
    actual = float(candidate_metrics[key])
    threshold = float(incumbent_metrics[key])
    passed = actual >= threshold if op == ">=" else actual <= threshold
    comparator = ">=" if op == ">=" else "<="
    checks[name] = (
        f"PASS {actual:.6f} {comparator} incumbent {threshold:.6f}"
        if passed
        else f"FAIL {actual:.6f} not {comparator} incumbent {threshold:.6f}"
    )


def _numeric_check(
    checks: dict[str, str],
    name: str,
    actual: float,
    op: str,
    threshold: float,
) -> None:
    passed = actual >= threshold if op == ">=" else actual <= threshold
    comparator = ">=" if op == ">=" else "<="
    checks[name] = (
        f"PASS {actual:.6f} {comparator} {threshold:.6f}"
        if passed
        else f"FAIL {actual:.6f} not {comparator} {threshold:.6f}"
    )


def _to_float(value: object) -> float:
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


__all__ = ["PromotionThresholds", "evaluate_promotion"]
