"""Smoke tests for user-facing scripts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_train_demo_dry_run() -> None:
    proc = _run("scripts/train.py", "--config", "configs/v1.yaml", "--demo", "--dry-run")
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_train_demo_dry_run_with_pretraining() -> None:
    proc = _run(
        "scripts/train.py",
        "--config",
        "configs/v1.yaml",
        "--demo",
        "--dry-run",
        "--pretrain-epochs",
        "1",
        "--pretrain-gap-bars",
        "2",
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_train_world_model_demo_dry_run() -> None:
    proc = _run(
        "scripts/train.py",
        "--config",
        "configs/v1.yaml",
        "--demo",
        "--model-kind",
        "world_model",
        "--dry-run",
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_backtest_model_demo_dry_run() -> None:
    proc = _run("scripts/backtest.py", "--config", "configs/v1.yaml", "--demo", "--model", "--dry-run")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "BACKTEST SUMMARY" in proc.stdout or "BACKTEST SETUP FAILED" in proc.stdout


def test_backtest_demo_dry_run_prints_single_summary() -> None:
    proc = _run("scripts/backtest.py", "--config", "configs/v1.yaml", "--demo", "--dry-run")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "BACKTEST SUMMARY" in proc.stdout
    assert "leakage_passed: PASS" in proc.stdout
    assert "sharpe_at_5bps:" in proc.stdout
    assert "sharpe_at_25bps:" in proc.stdout


def test_backtest_model_demo_dry_run_no_persist_state() -> None:
    # review finding H1 A/B knob: --no-persist-state must parse and not crash the
    # heuristic/demo (non-world-model) path, which ignores it.
    proc = _run(
        "scripts/backtest.py",
        "--config",
        "configs/v1.yaml",
        "--demo",
        "--model",
        "--dry-run",
        "--no-persist-state",
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_backtest_all_strategies_demo_dry_run() -> None:
    proc = _run("scripts/backtest.py", "--config", "configs/v1.yaml", "--demo", "--dry-run", "--all-strategies")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "BACKTEST STRATEGY COMPARISON" in proc.stdout
    assert "sharpe_at_25bps=" in proc.stdout


def test_backtest_walk_forward_demo_dry_run() -> None:
    proc = _run("scripts/backtest.py", "--config", "configs/v1.yaml", "--demo", "--dry-run", "--walk-forward")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "BACKTEST WALK-FORWARD SUMMARY" in proc.stdout
    assert "sharpe_at_25bps=" in proc.stdout


def test_backtest_walk_forward_warns_without_crashing_when_embargo_below_horizon(tmp_path) -> None:
    # LOW item: embargo_bars should be >= the largest strategy horizon under
    # walk-forward evaluation; this exercises the new sanity warning in
    # scripts/backtest.py and confirms it uses %s-style logging (a plain
    # stdlib Logger crashes on kwargs-style calls at WARNING level).
    import yaml

    with (_ROOT / "configs" / "v1.yaml").open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg.setdefault("training", {})["embargo_bars"] = 1
    config_path = tmp_path / "low_embargo.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    proc = _run(
        "scripts/backtest.py",
        "--config",
        str(config_path),
        "--demo",
        "--dry-run",
        "--all-strategies",
        "--walk-forward",
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "embargo_bars_below_horizon" in proc.stderr


def test_calibrate_dry_run_does_not_require_real_files() -> None:
    proc = _run(
        "scripts/calibrate.py",
        "--config",
        "configs/v1.yaml",
        "--model-path",
        "/tmp/model.pt",
        "--labels-path",
        "/tmp/labels.parquet",
        "--data-dir",
        "/tmp/data",
        "--report-out",
        "/tmp/calibration_report.json",
        "--dry-run",
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "CALIBRATION SUMMARY" in proc.stdout
    assert "status: PASS" in proc.stdout


def test_validate_data_demo_dry_run_prints_summary() -> None:
    proc = _run("scripts/validate_data.py", "--config", "configs/v1.yaml", "--demo", "--dry-run")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "VALIDATION SUMMARY" in proc.stdout
    assert "status: PASS" in proc.stdout
    assert "provenance_passed: PASS" in proc.stdout


def test_train_workflow_dry_run_does_not_require_real_files() -> None:
    proc = _run(
        "scripts/train_workflow.py",
        "--config",
        "configs/v1.yaml",
        "--data-dir",
        "/tmp/data",
        "--dry-run",
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_train_workflow_paper_dry_run_does_not_require_real_files() -> None:
    proc = _run(
        "scripts/train_workflow.py",
        "--config",
        "configs/v1.yaml",
        "--paper-config",
        "configs/paper_trading.yaml",
        "--data-dir",
        "/tmp/data",
        "--paper",
        "--dry-run",
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_paper_trade_dry_run() -> None:
    proc = _run("scripts/paper_trade.py", "--config", "configs/paper_trading.yaml", "--dry-run")
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_paper_trade_model_dry_run() -> None:
    proc = _run(
        "scripts/paper_trade.py",
        "--config",
        "configs/paper_trading.yaml",
        "--data-dir",
        "/tmp/data",
        "--real",
        "--model",
        "--model-path",
        "/tmp/forecaster.pt",
        "--dry-run",
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_build_features_demo_dry_run() -> None:
    proc = _run(
        "scripts/build_features.py",
        "--config",
        "configs/v1.yaml",
        "--demo",
        "--dry-run",
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_audit_local_data_dry_run_does_not_require_real_files() -> None:
    proc = _run(
        "scripts/audit_local_data.py",
        "--config",
        "configs/v1.yaml",
        "--data-dir",
        "/tmp/data",
        "--dry-run",
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_generate_report_writes_summary_json(tmp_path) -> None:
    audit = tmp_path / "decisions.jsonl"
    audit.write_text(
        json.dumps(
            {
                "strategy_name": "medium_frequency",
                "final_action": {"action_type": "no_trade", "size_fraction": 0.0},
                "reason_code": "OK",
                "latent_regime": "range",
                "expected_reward": 0.0,
                "expected_cost": 0.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "report.json"
    proc = _run(
        "scripts/generate_report.py",
        "--config",
        "configs/v1.yaml",
        "--audit",
        str(audit),
        "--out-path",
        str(out),
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert out.exists()
