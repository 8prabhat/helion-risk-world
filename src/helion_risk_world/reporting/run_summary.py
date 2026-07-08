from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from helion_risk_world.reporting.promotion_gate import PromotionThresholds, evaluate_promotion


def summarize_decision_audit(audit_path: str | Path) -> dict[str, object]:
    """Summarize one decisions.jsonl audit stream into a compact review payload."""
    path = Path(audit_path)
    actions: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    execution_statuses: Counter[str] = Counter()
    regime_pnl: dict[str, float] = defaultdict(float)
    strategy_name: str | None = None
    n = 0
    exp_reward = 0.0
    exp_cost = 0.0
    data_alerts = 0
    data_blocks = 0

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            decision = json.loads(line)
            n += 1
            strategy_name = strategy_name or decision.get("strategy_name")
            actions[decision["final_action"]["action_type"]] += 1
            reasons[decision["reason_code"]] += 1
            status = decision.get("execution_status")
            if status:
                execution_statuses[str(status)] += 1
            regime_pnl[str(decision.get("latent_regime", "unknown"))] += float(
                decision.get("expected_reward", 0.0)
            )
            exp_reward += float(decision.get("expected_reward", 0.0))
            exp_cost += float(decision.get("expected_cost", 0.0))
            data_status = decision.get("data_quality_status") or {}
            if float(data_status.get("data_alert", 0.0)):
                data_alerts += 1
            if data_status.get("execution_override") == "blocked_risk_increase":
                data_blocks += 1

    return {
        "kind": "decision_audit",
        "path": str(path),
        "strategy": strategy_name,
        "n_decisions": n,
        "action_mix": dict(actions),
        "reason_codes": dict(reasons),
        "execution_statuses": dict(execution_statuses),
        "data_alert_count": data_alerts,
        "data_block_count": data_blocks,
        "regime_expected_reward": {k: round(v, 6) for k, v in regime_pnl.items()},
        "total_expected_reward": round(exp_reward, 6),
        "total_expected_cost": round(exp_cost, 6),
    }


def build_report(audit_path: str | Path) -> dict[str, object]:
    """Build a durable report summary from a backtest/paper/workflow output path."""
    path = Path(audit_path)
    if path.is_dir():
        single_backtest_path = path / "backtest_report.json"
        if single_backtest_path.exists():
            return _summarize_single_backtest(_read_json(single_backtest_path), single_backtest_path)

        walk_forward_path = path / "walk_forward.json"
        if walk_forward_path.exists():
            return _summarize_walk_forward(_read_json(walk_forward_path), walk_forward_path)

        comparison_path = path / "strategy_comparison.json"
        if comparison_path.exists():
            return _summarize_strategy_comparison(_read_json(comparison_path), comparison_path)

        decisions_path = path / "decisions.jsonl"
        monitor_path = path / "monitor_summary.json"
        if monitor_path.exists() or decisions_path.exists():
            report: dict[str, object] = {
                "kind": "paper_trading" if monitor_path.exists() else "decision_audit",
                "path": str(path),
            }
            if decisions_path.exists():
                report["decision_summary"] = summarize_decision_audit(decisions_path)
            if monitor_path.exists():
                report["monitor_summary"] = _read_json(monitor_path)
            return report

        files = sorted(path.glob("*/decisions.jsonl"))
        if files:
            return {
                "kind": "multi_strategy_audits",
                "path": str(path),
                "n_strategies": len(files),
                "strategies": [summarize_decision_audit(file) for file in files],
            }

    if path.suffix == ".json":
        payload = _read_json(path)
        if "summary" in payload and "diagnostics" in payload and "strategy" in payload:
            return _summarize_single_backtest(payload, path)
        if "best_strategy" in payload and "strategies" in payload:
            if payload.get("strategies") and "out_of_sample" in payload["strategies"][0]:
                return _summarize_walk_forward(payload, path)
            return _summarize_strategy_comparison(payload, path)

    return summarize_decision_audit(path)


def build_workflow_summary(
    *,
    config_path: str | Path,
    data_dir: str | Path,
    run_dir: str | Path,
    assembled_path: str | Path,
    labels_path: str | Path,
    model_path: str | Path,
    calibration_report_path: str | Path,
    backtest_output_path: str | Path | None = None,
    report_summary_path: str | Path | None = None,
    paper_output_path: str | Path | None = None,
    paper_report_summary_path: str | Path | None = None,
    incumbent_calibration_report_path: str | Path | None = None,
    incumbent_report_summary_path: str | Path | None = None,
    incumbent_paper_report_summary_path: str | Path | None = None,
    promotion_cfg: dict[str, Any] | None = None,
    strategy: str | None = None,
    walk_forward: bool = False,
    all_strategies: bool = False,
    pretrain_epochs: int | None = None,
    pretrain_gap_bars: int | None = None,
) -> dict[str, object]:
    """Build a compact end-to-end workflow summary from stage artifacts."""
    calibration_payload = _read_json_if_exists(calibration_report_path)
    report_payload = _read_json_if_exists(report_summary_path)
    if report_payload is None and backtest_output_path is not None:
        backtest_path = Path(backtest_output_path)
        if backtest_path.exists():
            report_payload = build_report(backtest_path)
    paper_payload = _read_json_if_exists(paper_report_summary_path)
    if paper_payload is None and paper_output_path is not None:
        paper_path = Path(paper_output_path)
        if paper_path.exists():
            paper_payload = build_report(paper_path)
    incumbent_calibration_payload = _read_json_if_exists(incumbent_calibration_report_path)
    incumbent_report_payload = _read_json_if_exists(incumbent_report_summary_path)
    incumbent_paper_payload = _read_json_if_exists(incumbent_paper_report_summary_path)
    thresholds = PromotionThresholds.from_mapping(promotion_cfg)
    promotion = evaluate_promotion(
        calibration=calibration_payload,
        report=report_payload,
        paper=paper_payload,
        incumbent_calibration=incumbent_calibration_payload,
        incumbent_report=incumbent_report_payload,
        incumbent_paper=incumbent_paper_payload,
        thresholds=thresholds,
    )

    return {
        "kind": "workflow_summary",
        "config_path": str(config_path),
        "data_dir": str(data_dir),
        "run_dir": str(run_dir),
        "mode": {
            "strategy": strategy,
            "walk_forward": bool(walk_forward),
            "all_strategies": bool(all_strategies),
        },
        "pretraining": {
            "pretrain_epochs": pretrain_epochs,
            "pretrain_gap_bars": pretrain_gap_bars,
        },
        "artifacts": {
            "assembled": _artifact_entry(assembled_path),
            "labels": _artifact_entry(labels_path),
            "model": _artifact_entry(model_path),
            "calibration_report": _artifact_entry(calibration_report_path),
            "backtest_output": _artifact_entry(backtest_output_path) if backtest_output_path is not None else None,
            "report_summary": _artifact_entry(report_summary_path) if report_summary_path is not None else None,
            "paper_output": _artifact_entry(paper_output_path) if paper_output_path is not None else None,
            "paper_report_summary": (
                _artifact_entry(paper_report_summary_path)
                if paper_report_summary_path is not None
                else None
            ),
            "incumbent_calibration_report": (
                _artifact_entry(incumbent_calibration_report_path)
                if incumbent_calibration_report_path is not None
                else None
            ),
            "incumbent_report_summary": (
                _artifact_entry(incumbent_report_summary_path)
                if incumbent_report_summary_path is not None
                else None
            ),
            "incumbent_paper_report_summary": (
                _artifact_entry(incumbent_paper_report_summary_path)
                if incumbent_paper_report_summary_path is not None
                else None
            ),
        },
        "calibration": _summarize_calibration(calibration_payload, calibration_report_path),
        "report": report_payload,
        "paper_report": paper_payload,
        "incumbent_calibration": _summarize_calibration(
            incumbent_calibration_payload,
            incumbent_calibration_report_path,
        ),
        "incumbent_report": incumbent_report_payload,
        "incumbent_paper_report": incumbent_paper_payload,
        "promotion": promotion,
    }


def write_json_report(payload: dict[str, object], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def _summarize_strategy_comparison(payload: dict[str, Any], source_path: Path) -> dict[str, object]:
    strategies = payload.get("strategies", [])
    return {
        "kind": "strategy_comparison",
        "path": str(source_path),
        "ranking_metric": payload.get("ranking_metric", "sharpe"),
        "best_strategy": payload.get("best_strategy"),
        "n_strategies": len(strategies),
        "strategies": [
            {
                "strategy": row.get("strategy"),
                "description": row.get("description"),
                **dict(row.get("summary", {})),
            }
            for row in strategies
        ],
    }


def _summarize_single_backtest(payload: dict[str, Any], source_path: Path) -> dict[str, object]:
    return {
        "kind": "single_backtest",
        "path": str(source_path),
        "strategy": payload.get("strategy"),
        "predictor": payload.get("predictor"),
        **dict(payload.get("summary", {})),
        "diagnostics": payload.get("diagnostics", {}),
    }


def _summarize_walk_forward(payload: dict[str, Any], source_path: Path) -> dict[str, object]:
    strategies = payload.get("strategies", [])
    return {
        "kind": "walk_forward",
        "path": str(source_path),
        "ranking_metric": payload.get("ranking_metric", "sharpe"),
        "best_strategy": payload.get("best_strategy"),
        "n_strategies": len(strategies),
        "strategies": [
            {
                "strategy": row.get("strategy"),
                "n_folds": row.get("n_folds"),
                "in_sample": row.get("in_sample"),
                "out_of_sample": row.get("out_of_sample"),
            }
            for row in strategies
        ],
    }


def _summarize_calibration(
    payload: dict[str, Any] | None,
    source_path: str | Path,
) -> dict[str, object] | None:
    if payload is None:
        return None
    gate = payload.get("gate", {})
    return {
        "path": str(source_path),
        "gate": gate,
        "metrics": payload.get("metrics", {}),
        "regime_parity": payload.get("regime_parity"),
    }


def _artifact_entry(path: str | Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    resolved = Path(path)
    return {"path": str(resolved), "exists": resolved.exists()}


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_json_if_exists(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.exists():
        return None
    return _read_json(resolved)
