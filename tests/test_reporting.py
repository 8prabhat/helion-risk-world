"""Workflow/report summary helpers."""

from __future__ import annotations

import json
from pathlib import Path

from helion_risk_world.reporting import (
    PromotionThresholds,
    build_report,
    build_workflow_summary,
    evaluate_promotion,
)


def test_build_report_combines_paper_decisions_and_monitor_summary(tmp_path: Path) -> None:
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "strategy_name": "scalping",
                        "final_action": {"action_type": "enter_long", "size_fraction": 0.5},
                        "reason_code": "OK",
                        "latent_regime": "trend",
                        "expected_reward": 0.02,
                        "expected_cost": 0.001,
                        "execution_status": "accepted",
                        "data_quality_status": {"status": "ok", "data_alert": 0.0},
                    }
                ),
                json.dumps(
                    {
                        "strategy_name": "scalping",
                        "final_action": {"action_type": "no_trade", "size_fraction": 0.0},
                        "reason_code": "EVENT_BLACKOUT",
                        "latent_regime": "event",
                        "expected_reward": 0.0,
                        "expected_cost": 0.0,
                        "execution_status": "data_blocked",
                        "data_quality_status": {
                            "status": "alert",
                            "data_alert": 1.0,
                            "execution_override": "blocked_risk_increase",
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "monitor_summary.json").write_text(
        json.dumps({"data_quality": {"data_alert": 1.0}, "live_stats": {"net_pnl": 10.0}}),
        encoding="utf-8",
    )

    payload = build_report(tmp_path)

    assert payload["kind"] == "paper_trading"
    decision_summary = payload["decision_summary"]
    assert decision_summary["n_decisions"] == 2
    assert decision_summary["data_alert_count"] == 1
    assert decision_summary["data_block_count"] == 1
    assert payload["monitor_summary"]["live_stats"]["net_pnl"] == 10.0


def test_build_report_reads_single_backtest_summary_json(tmp_path: Path) -> None:
    report_path = tmp_path / "backtest_report.json"
    report_path.write_text(
        json.dumps(
            {
                "strategy": "medium_frequency",
                "predictor": "forecaster_artifact",
                "summary": {
                    "n_trades": 24,
                    "total_return": 0.18,
                    "annualized_return": 0.22,
                    "sharpe": 1.7,
                    "profit_factor": 1.4,
                    "max_drawdown": 0.08,
                    "hit_rate": 0.52,
                },
                "diagnostics": {"deflated_sharpe": {"dsr": 0.9}},
            }
        ),
        encoding="utf-8",
    )

    payload = build_report(tmp_path)

    assert payload["kind"] == "single_backtest"
    assert payload["strategy"] == "medium_frequency"
    assert payload["predictor"] == "forecaster_artifact"
    assert payload["total_return"] == 0.18
    assert payload["annualized_return"] == 0.22
    assert payload["diagnostics"]["deflated_sharpe"]["dsr"] == 0.9


def test_build_workflow_summary_embeds_calibration_and_report_payloads(tmp_path: Path) -> None:
    calibration_report = tmp_path / "calibration.json"
    calibration_report.write_text(
        json.dumps(
            {
                "gate": {"passed": False, "reasons": {"coverage": "FAIL too wide"}},
                "metrics": {"coverage_error": 0.12, "barrier_brier": 0.21},
            }
        ),
        encoding="utf-8",
    )
    report_summary = tmp_path / "report_summary.json"
    report_summary.write_text(
        json.dumps({"kind": "strategy_comparison", "best_strategy": "scalping"}),
        encoding="utf-8",
    )
    model_path = tmp_path / "forecaster.pt"
    model_path.write_text("x", encoding="utf-8")

    summary = build_workflow_summary(
        config_path="configs/v1.yaml",
        data_dir="data",
        run_dir=tmp_path,
        assembled_path="data/processed/banknifty_5min.parquet",
        labels_path="data/processed/labels.parquet",
        model_path=model_path,
        calibration_report_path=calibration_report,
        backtest_output_path="runs/backtest/strategy_comparison.json",
        report_summary_path=report_summary,
        strategy="scalping",
        walk_forward=False,
        all_strategies=False,
        pretrain_epochs=5,
        pretrain_gap_bars=2,
    )

    assert summary["kind"] == "workflow_summary"
    assert summary["mode"]["strategy"] == "scalping"
    assert summary["pretraining"]["pretrain_epochs"] == 5
    assert summary["calibration"]["gate"]["passed"] is False
    assert summary["report"]["best_strategy"] == "scalping"
    assert summary["artifacts"]["model"]["exists"] is True
    assert summary["promotion"]["passed"] is False


def test_promotion_gate_passes_strong_single_backtest() -> None:
    decision = evaluate_promotion(
        calibration={"gate": {"passed": True}},
        report={
            "kind": "single_backtest",
            "strategy": "scalping",
            "summary": {
                "n_trades": 30,
                "total_return": 0.20,
                "annualized_return": 0.21,
                "sharpe": 1.8,
                "profit_factor": 1.5,
                "max_drawdown": 0.08,
                "hit_rate": 0.51,
            },
        },
        thresholds=PromotionThresholds(),
    )

    assert decision["passed"] is True
    assert decision["candidate"] == "scalping"
    assert all(
        result.startswith("PASS") or result.startswith("SKIP")
        for result in decision["checks"].values()
    )
    assert decision["checks"]["paper_report"] == "SKIP paper report not required"


def test_promotion_gate_passes_flattened_single_backtest_report_with_paper_metrics() -> None:
    decision = evaluate_promotion(
        calibration={"gate": {"passed": True}},
        report={
            "kind": "single_backtest",
            "strategy": "scalping",
            "n_trades": 30,
            "total_return": 0.20,
            "annualized_return": 0.21,
            "sharpe": 1.8,
            "profit_factor": 1.5,
            "max_drawdown": 0.08,
            "hit_rate": 0.51,
        },
        paper={
            "kind": "paper_trading",
            "monitor_summary": {
                "live_stats": {"reject_rate": 0.05},
                "data_quality": {"failure_rate": 0.01},
                "drift": {"drift_score": 0.1},
            },
        },
        thresholds=PromotionThresholds(require_paper_report=True),
    )

    assert decision["passed"] is True
    assert decision["candidate"] == "scalping"
    assert decision["checks"]["paper_report"].startswith("PASS")
    assert decision["checks"]["paper_reject_rate"].startswith("PASS")
    assert decision["checks"]["paper_data_failure_rate"].startswith("PASS")
    assert decision["checks"]["paper_drift_score"].startswith("PASS")


def test_promotion_gate_requires_paper_report_when_enabled() -> None:
    decision = evaluate_promotion(
        calibration={"gate": {"passed": True}},
        report={
            "kind": "single_backtest",
            "strategy": "scalping",
            "n_trades": 30,
            "total_return": 0.20,
            "annualized_return": 0.21,
            "sharpe": 1.8,
            "profit_factor": 1.5,
            "max_drawdown": 0.08,
            "hit_rate": 0.51,
        },
        paper=None,
        thresholds=PromotionThresholds(require_paper_report=True),
    )

    assert decision["passed"] is False
    assert decision["checks"]["paper_report"] == "FAIL paper report required for promotion"


def test_promotion_gate_compares_against_incumbent_when_provided() -> None:
    decision = evaluate_promotion(
        calibration={
            "gate": {"passed": True},
            "metrics": {"coverage_error": 0.03, "barrier_brier": 0.12},
        },
        report={
            "kind": "single_backtest",
            "strategy": "candidate",
            "n_trades": 30,
            "total_return": 0.20,
            "annualized_return": 0.21,
            "sharpe": 1.8,
            "profit_factor": 1.5,
            "max_drawdown": 0.08,
            "hit_rate": 0.51,
        },
        incumbent_calibration={
            "gate": {"passed": True},
            "metrics": {"coverage_error": 0.05, "barrier_brier": 0.15},
        },
        incumbent_report={
            "kind": "single_backtest",
            "strategy": "incumbent",
            "n_trades": 28,
            "total_return": 0.18,
            "annualized_return": 0.19,
            "sharpe": 1.6,
            "profit_factor": 1.3,
            "max_drawdown": 0.09,
            "hit_rate": 0.49,
        },
        thresholds=PromotionThresholds(require_ab_gate=True),
    )

    assert decision["passed"] is True
    assert decision["checks"]["ab_gate"].startswith("PASS")
    assert decision["checks"]["ab_sharpe"].startswith("PASS")
    assert decision["incumbent"]["candidate"] == "incumbent"


def test_promotion_gate_fails_when_ab_required_without_incumbent() -> None:
    decision = evaluate_promotion(
        calibration={"gate": {"passed": True}},
        report={
            "kind": "single_backtest",
            "strategy": "candidate",
            "n_trades": 30,
            "total_return": 0.20,
            "annualized_return": 0.21,
            "sharpe": 1.8,
            "profit_factor": 1.5,
            "max_drawdown": 0.08,
            "hit_rate": 0.51,
        },
        thresholds=PromotionThresholds(require_ab_gate=True),
    )

    assert decision["passed"] is False
    assert decision["checks"]["ab_gate"] == "FAIL incumbent comparison required but no incumbent artifacts provided"


def test_build_workflow_summary_embeds_paper_report_payload(tmp_path: Path) -> None:
    calibration_report = tmp_path / "calibration.json"
    calibration_report.write_text(
        json.dumps({"gate": {"passed": True}, "metrics": {"coverage_error": 0.01}}),
        encoding="utf-8",
    )
    report_summary = tmp_path / "report_summary.json"
    report_summary.write_text(
        json.dumps(
            {
                "kind": "single_backtest",
                "strategy": "medium_frequency",
                "n_trades": 24,
                "total_return": 0.18,
                "annualized_return": 0.22,
                "sharpe": 1.7,
                "profit_factor": 1.4,
                "max_drawdown": 0.08,
                "hit_rate": 0.52,
            }
        ),
        encoding="utf-8",
    )
    paper_report = tmp_path / "paper_report_summary.json"
    paper_report.write_text(
        json.dumps(
            {
                "kind": "paper_trading",
                "monitor_summary": {
                    "live_stats": {"reject_rate": 0.03},
                    "data_quality": {"failure_rate": 0.01},
                    "drift": {"drift_score": 0.1},
                },
            }
        ),
        encoding="utf-8",
    )
    model_path = tmp_path / "forecaster.pt"
    model_path.write_text("x", encoding="utf-8")

    summary = build_workflow_summary(
        config_path="configs/v1.yaml",
        data_dir="data",
        run_dir=tmp_path,
        assembled_path="data/processed/banknifty_5min.parquet",
        labels_path="data/processed/labels.parquet",
        model_path=model_path,
        calibration_report_path=calibration_report,
        backtest_output_path="runs/backtest/backtest_report.json",
        report_summary_path=report_summary,
        paper_report_summary_path=paper_report,
        promotion_cfg={"require_paper_report": True},
        strategy="medium_frequency",
        walk_forward=False,
        all_strategies=False,
    )

    assert summary["paper_report"]["kind"] == "paper_trading"
    assert summary["artifacts"]["paper_report_summary"]["exists"] is True
    assert summary["promotion"]["passed"] is True
