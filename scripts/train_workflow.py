"""Run the full real-data training workflow with explicit stage gates.

Stages:
  1. validate_data.py
  2. label.py
  3. train.py
  4. calibrate.py
  5. backtest.py
  6. generate_report.py

This keeps each stage's implementation single-sourced in its own script while providing one
repeatable orchestration entry point for full retraining runs.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _bootstrap import ensure_src_path

ensure_src_path()

from helion_risk_world.config.loaders import (
    management_horizon_from_mapping as management_horizon_from_cfg,
    strategy_profile_from_mapping as strategy_profile_from_cfg,
)

from helion_risk_world.integration import get_logger, load_config
from helion_risk_world.reporting import build_workflow_summary, write_json_report

log = get_logger("hrw.workflow")
_ROOT = Path(__file__).resolve().parents[1]
_BACKTEST_DIR = _ROOT / "runs" / "backtest"


@dataclass(frozen=True)
class WorkflowPaths:
    config_path: Path
    data_dir: Path
    run_dir: Path
    assembled_path: Path
    labels_path: Path
    model_path: Path
    calibration_report: Path
    report_summary_path: Path
    paper_report_summary_path: Path
    workflow_summary_path: Path

    @classmethod
    def resolve(
        cls,
        *,
        config_path: Path,
        data_dir: Path,
        run_dir: Path,
        assembled_path: Path | None = None,
        labels_path: Path | None = None,
        model_path: Path | None = None,
        calibration_report: Path | None = None,
        report_summary_path: Path | None = None,
        paper_report_summary_path: Path | None = None,
        workflow_summary_path: Path | None = None,
    ) -> WorkflowPaths:
        return cls(
            config_path=config_path,
            data_dir=data_dir,
            run_dir=run_dir,
            assembled_path=assembled_path or data_dir / "processed" / "banknifty_5min.parquet",
            labels_path=labels_path or data_dir / "processed" / "labels.parquet",
            model_path=model_path or run_dir / "forecaster.pt",
            calibration_report=calibration_report or run_dir / "calibration_report.json",
            report_summary_path=report_summary_path or run_dir / "report_summary.json",
            paper_report_summary_path=(
                paper_report_summary_path or run_dir / "paper_report_summary.json"
            ),
            workflow_summary_path=workflow_summary_path or run_dir / "workflow_summary.json",
        )

    def artifacts(self) -> dict[str, Path]:
        return {
            "assembled": self.assembled_path,
            "labels": self.labels_path,
            "model": self.model_path,
            "calibration_report": self.calibration_report,
            "report_summary": self.report_summary_path,
            "paper_report_summary": self.paper_report_summary_path,
            "workflow_summary": self.workflow_summary_path,
        }


@dataclass(frozen=True)
class WorkflowOptions:
    model_kind: str | None
    model_path_explicit: bool
    stop_mult: float
    target_mult: float
    pretrain_epochs: int | None
    pretrain_gap_bars: int | None
    head_finetune_epochs: int | None
    incumbent_calibration_report_path: Path | None
    incumbent_report_summary_path: Path | None
    incumbent_paper_report_summary_path: Path | None
    paper: bool
    paper_config_path: Path | None
    strategy: str | None
    walk_forward: bool
    all_strategies: bool
    skip_validation: bool
    skip_labeling: bool
    skip_calibration: bool
    skip_backtest: bool
    skip_report: bool
    dry_run: bool


@dataclass
class WorkflowState:
    backtest_output_path: Path | None = None
    paper_output_path: Path | None = None


@dataclass(frozen=True)
class PaperStageConfig:
    config_path: Path
    monitor_path: Path
    output_path: Path


def _section(cfg: Mapping[str, Any] | None, key: str) -> Mapping[str, Any]:
    if not isinstance(cfg, Mapping):
        return {}
    section = cfg.get(key, {})
    return section if isinstance(section, Mapping) else {}


def _paper_monitor_path(cfg: Mapping[str, Any] | None) -> Path:
    return Path(_section(cfg, "paper").get("monitor_report_path", "runs/paper/monitor_summary.json"))


def _default_model_kind(*, strategy_horizon: int, management_horizon: int) -> str:
    return "forecaster" if strategy_horizon == management_horizon else "world_model"


def _normalize_model_artifact_path(
    paths: WorkflowPaths,
    *,
    model_kind: str,
    explicit: bool,
) -> WorkflowPaths:
    if explicit:
        return paths
    target_name = "world_model.pt" if model_kind == "world_model" else "forecaster.pt"
    target_path = paths.run_dir / target_name
    if paths.model_path == target_path:
        return paths
    return WorkflowPaths.resolve(
        config_path=paths.config_path,
        data_dir=paths.data_dir,
        run_dir=paths.run_dir,
        assembled_path=paths.assembled_path,
        labels_path=paths.labels_path,
        model_path=target_path,
        calibration_report=paths.calibration_report,
        report_summary_path=paths.report_summary_path,
        paper_report_summary_path=paths.paper_report_summary_path,
        workflow_summary_path=paths.workflow_summary_path,
    )


def _run_stage(name: str, cmd: list[str], *, dry_run: bool) -> None:
    rendered = " ".join(shlex.quote(part) for part in cmd)
    log.info("workflow.stage", stage=name, cmd=rendered, dry_run=dry_run)
    if dry_run:
        return
    proc = subprocess.run(cmd, cwd=_ROOT, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {proc.returncode}")


def _assert_exists(path: Path, *, stage: str, dry_run: bool) -> None:
    if dry_run:
        return
    if not path.exists():
        raise FileNotFoundError(f"{stage} did not produce expected artifact: {path}")


def _mtime(path: Path) -> float | None:
    return path.stat().st_mtime if path.exists() else None


def _assert_updated(path: Path, *, before: float | None, stage: str, dry_run: bool) -> None:
    _assert_exists(path, stage=stage, dry_run=dry_run)
    if dry_run:
        return
    if before is not None and path.stat().st_mtime < before:
        raise RuntimeError(f"{stage} did not refresh expected artifact: {path}")


def _run_refreshing_stage(
    name: str,
    cmd: list[str],
    *,
    refreshed_path: Path,
    dry_run: bool,
) -> None:
    before = _mtime(refreshed_path)
    _run_stage(name, cmd, dry_run=dry_run)
    _assert_updated(refreshed_path, before=before, stage=name, dry_run=dry_run)


def _run_output_stage(
    name: str,
    cmd: list[str],
    *,
    output_path: Path,
    dry_run: bool,
) -> None:
    _run_stage(name, cmd, dry_run=dry_run)
    _assert_exists(output_path, stage=name, dry_run=dry_run)


def _validate_cmd(paths: WorkflowPaths) -> list[str]:
    return [
        sys.executable,
        "scripts/validate_data.py",
        "--config",
        str(paths.config_path),
        "--data-dir",
        str(paths.data_dir),
    ]


def _label_cmd(
    paths: WorkflowPaths,
    *,
    horizon: int,
    target_horizons: tuple[int, ...],
    options: WorkflowOptions,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/label.py",
        "--data-path",
        str(paths.assembled_path),
        "--out-path",
        str(paths.labels_path),
        "--H",
        str(horizon),
        "--stop-mult",
        str(options.stop_mult),
        "--target-mult",
        str(options.target_mult),
    ]
    if target_horizons:
        cmd.extend(["--target-horizons", *[str(h) for h in target_horizons]])
    return cmd


def _train_cmd(paths: WorkflowPaths, options: WorkflowOptions) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/train.py",
        "--config",
        str(paths.config_path),
        "--data-dir",
        str(paths.data_dir),
        "--labels-path",
        str(paths.labels_path),
        "--model-path",
        str(paths.model_path),
    ]
    if options.model_kind is not None:
        cmd.extend(["--model-kind", options.model_kind])
    if options.pretrain_epochs is not None:
        cmd.extend(["--pretrain-epochs", str(options.pretrain_epochs)])
    if options.pretrain_gap_bars is not None:
        cmd.extend(["--pretrain-gap-bars", str(options.pretrain_gap_bars)])
    if options.head_finetune_epochs is not None:
        cmd.extend(["--head-finetune-epochs", str(options.head_finetune_epochs)])
    return cmd


def _calibrate_cmd(paths: WorkflowPaths) -> list[str]:
    return [
        sys.executable,
        "scripts/calibrate.py",
        "--config",
        str(paths.config_path),
        "--model-path",
        str(paths.model_path),
        "--labels-path",
        str(paths.labels_path),
        "--data-dir",
        str(paths.data_dir),
        "--report-out",
        str(paths.calibration_report),
    ]


def _model_execution_cmd(
    script_name: str,
    *,
    config_path: Path,
    data_dir: Path,
    model_path: Path,
    calibration_report: Path | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        script_name,
        "--config",
        str(config_path),
        "--real",
        "--data-dir",
        str(data_dir),
        "--model",
        "--model-path",
        str(model_path),
    ]
    # Review finding M13: pass the calibration report through explicitly so
    # backtest.py/paper_trade.py enforce the gate themselves, rather than
    # relying solely on train_workflow.py's own subprocess ordering to keep an
    # uncalibrated artifact from ever reaching these stages.
    if calibration_report is not None:
        cmd.extend(["--calibration-report", str(calibration_report)])
    return cmd


def _append_strategy_flag(cmd: list[str], strategy: str | None) -> None:
    if strategy:
        cmd.extend(["--strategy", strategy])


def _append_backtest_mode_flags(cmd: list[str], options: WorkflowOptions) -> None:
    _append_strategy_flag(cmd, options.strategy)
    if options.walk_forward:
        cmd.append("--walk-forward")
    if options.all_strategies:
        cmd.append("--all-strategies")


def _backtest_cmd(paths: WorkflowPaths, options: WorkflowOptions) -> list[str]:
    cmd = _model_execution_cmd(
        "scripts/backtest.py",
        config_path=paths.config_path,
        data_dir=paths.data_dir,
        model_path=paths.model_path,
        calibration_report=paths.calibration_report,
    )
    _append_backtest_mode_flags(cmd, options)
    return cmd


def _backtest_output_path(options: WorkflowOptions) -> Path:
    if options.walk_forward:
        return _BACKTEST_DIR / "walk_forward.json"
    if options.all_strategies:
        return _BACKTEST_DIR / "strategy_comparison.json"
    return _BACKTEST_DIR / "backtest_report.json"


def _report_cmd(config_path: Path, *, audit_path: Path, out_path: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/generate_report.py",
        "--config",
        str(config_path),
        "--audit",
        str(audit_path),
        "--out-path",
        str(out_path),
    ]


def _resolve_paper_stage(paths: WorkflowPaths, options: WorkflowOptions) -> PaperStageConfig:
    config_path = options.paper_config_path or paths.config_path
    paper_cfg = load_config(str(config_path))
    monitor_path = _paper_monitor_path(paper_cfg)
    return PaperStageConfig(
        config_path=config_path,
        monitor_path=monitor_path,
        output_path=monitor_path.parent,
    )


def _paper_cmd(
    paths: WorkflowPaths,
    paper_stage: PaperStageConfig,
    options: WorkflowOptions,
) -> list[str]:
    cmd = _model_execution_cmd(
        "scripts/paper_trade.py",
        config_path=paper_stage.config_path,
        data_dir=paths.data_dir,
        model_path=paths.model_path,
        calibration_report=paths.calibration_report,
    )
    _append_strategy_flag(cmd, options.strategy)
    return cmd


def _existing_path(path: Path) -> Path | None:
    return path if path.exists() else None


def _write_workflow_summary(
    *,
    paths: WorkflowPaths,
    options: WorkflowOptions,
    cfg: Mapping[str, Any] | None,
    state: WorkflowState,
) -> None:
    workflow_summary = build_workflow_summary(
        config_path=paths.config_path,
        data_dir=paths.data_dir,
        run_dir=paths.run_dir,
        assembled_path=paths.assembled_path,
        labels_path=paths.labels_path,
        model_path=paths.model_path,
        calibration_report_path=paths.calibration_report,
        backtest_output_path=state.backtest_output_path,
        report_summary_path=_existing_path(paths.report_summary_path),
        paper_output_path=state.paper_output_path,
        paper_report_summary_path=_existing_path(paths.paper_report_summary_path),
        incumbent_calibration_report_path=_existing_path(options.incumbent_calibration_report_path)
        if options.incumbent_calibration_report_path is not None
        else None,
        incumbent_report_summary_path=_existing_path(options.incumbent_report_summary_path)
        if options.incumbent_report_summary_path is not None
        else None,
        incumbent_paper_report_summary_path=_existing_path(options.incumbent_paper_report_summary_path)
        if options.incumbent_paper_report_summary_path is not None
        else None,
        strategy=options.strategy,
        walk_forward=options.walk_forward,
        all_strategies=options.all_strategies,
        pretrain_epochs=options.pretrain_epochs,
        pretrain_gap_bars=options.pretrain_gap_bars,
        promotion_cfg=dict(_section(cfg, "promotion")) or None,
    )
    write_json_report(workflow_summary, paths.workflow_summary_path)
    log.info("workflow.summary_written", path=str(paths.workflow_summary_path))
    if (
        workflow_summary["promotion"]["passed"] is False
        and bool(workflow_summary["promotion"]["thresholds"].get("fail_workflow_on_reject", False))
    ):
        raise RuntimeError("workflow promotion gate rejected the run")


def run_workflow(
    *,
    paths: WorkflowPaths,
    options: WorkflowOptions,
) -> dict[str, Path]:
    cfg = load_config(str(paths.config_path))
    horizon = management_horizon_from_cfg(cfg)
    horizons_cfg = cfg.get("horizons", {}) if isinstance(cfg, Mapping) else {}
    target_horizons = tuple(sorted(set(int(h) for h in horizons_cfg.get("horizon_steps", [horizon]))))
    strategy = strategy_profile_from_cfg(cfg, options.strategy)
    model_kind = options.model_kind or _default_model_kind(
        strategy_horizon=strategy.decision_horizon_bars,
        management_horizon=horizon,
    )
    paths = _normalize_model_artifact_path(
        paths,
        model_kind=model_kind,
        explicit=options.model_path_explicit,
    )
    options = WorkflowOptions(**{**options.__dict__, "model_kind": model_kind})
    state = WorkflowState()

    if not options.skip_validation:
        _run_stage("validate", _validate_cmd(paths), dry_run=options.dry_run)

    if options.skip_labeling:
        _assert_exists(paths.labels_path, stage="label", dry_run=options.dry_run)
    else:
        _run_refreshing_stage(
            "label",
            _label_cmd(
                paths,
                horizon=horizon,
                target_horizons=target_horizons,
                options=options,
            ),
            refreshed_path=paths.labels_path,
            dry_run=options.dry_run,
        )

    _run_refreshing_stage(
        "train",
        _train_cmd(paths, options),
        refreshed_path=paths.model_path,
        dry_run=options.dry_run,
    )

    if not options.skip_calibration:
        _run_refreshing_stage(
            "calibrate",
            _calibrate_cmd(paths),
            refreshed_path=paths.calibration_report,
            dry_run=options.dry_run,
        )

    if not options.skip_backtest:
        state.backtest_output_path = _backtest_output_path(options)
        _run_refreshing_stage(
            "backtest",
            _backtest_cmd(paths, options),
            refreshed_path=state.backtest_output_path,
            dry_run=options.dry_run,
        )

        if not options.skip_report:
            _run_output_stage(
                "report",
                _report_cmd(
                    paths.config_path,
                    audit_path=_BACKTEST_DIR,
                    out_path=paths.report_summary_path,
                ),
                output_path=paths.report_summary_path,
                dry_run=options.dry_run,
            )

    if options.paper:
        if options.walk_forward or options.all_strategies:
            raise RuntimeError("paper stage currently supports only one strategy per workflow run")
        paper_stage = _resolve_paper_stage(paths, options)
        state.paper_output_path = paper_stage.output_path
        _run_refreshing_stage(
            "paper",
            _paper_cmd(paths, paper_stage, options),
            refreshed_path=paper_stage.monitor_path,
            dry_run=options.dry_run,
        )
        if not options.skip_report:
            _run_output_stage(
                "paper_report",
                _report_cmd(
                    paper_stage.config_path,
                    audit_path=paper_stage.output_path,
                    out_path=paths.paper_report_summary_path,
                ),
                output_path=paths.paper_report_summary_path,
                dry_run=options.dry_run,
            )

    if not options.dry_run:
        _write_workflow_summary(paths=paths, options=options, cfg=cfg, state=state)

    return paths.artifacts()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--data-dir", required=True, type=Path, help="Root data directory containing ohlcv/regime/processed.")
    p.add_argument("--run-dir", type=Path, default=Path("runs/train_workflow"))
    p.add_argument("--assembled-path", type=Path, default=None)
    p.add_argument("--labels-path", type=Path, default=None)
    p.add_argument("--model-path", type=Path, default=None)
    p.add_argument("--calibration-report", type=Path, default=None)
    p.add_argument("--report-summary-path", type=Path, default=None)
    p.add_argument("--paper-report-summary-path", type=Path, default=None)
    p.add_argument("--workflow-summary-path", type=Path, default=None)
    p.add_argument("--stop-mult", type=float, default=2.0)
    p.add_argument("--target-mult", type=float, default=2.0)
    p.add_argument(
        "--model-kind",
        choices=("forecaster", "world_model"),
        default=None,
        help=(
            "Artifact family to train. Defaults to world_model when the active strategy "
            "horizon differs from the management horizon; otherwise forecaster."
        ),
    )
    p.add_argument("--pretrain-epochs", type=int, default=None)
    p.add_argument("--pretrain-gap-bars", type=int, default=None)
    p.add_argument("--head-finetune-epochs", type=int, default=None)
    p.add_argument("--incumbent-calibration-report", type=Path, default=None)
    p.add_argument("--incumbent-report-summary-path", type=Path, default=None)
    p.add_argument("--incumbent-paper-report-summary-path", type=Path, default=None)
    p.add_argument("--paper", action="store_true")
    p.add_argument("--paper-config", type=Path, default=None)
    p.add_argument("--strategy", default=None)
    p.add_argument("--walk-forward", action="store_true")
    p.add_argument("--all-strategies", action="store_true")
    p.add_argument("--skip-validation", action="store_true")
    p.add_argument("--skip-labeling", action="store_true")
    p.add_argument("--skip-calibration", action="store_true")
    p.add_argument("--skip-backtest", action="store_true")
    p.add_argument("--skip-report", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    paths = WorkflowPaths.resolve(
        config_path=args.config,
        data_dir=args.data_dir,
        run_dir=args.run_dir,
        assembled_path=args.assembled_path,
        labels_path=args.labels_path,
        model_path=args.model_path,
        calibration_report=args.calibration_report,
        report_summary_path=args.report_summary_path,
        paper_report_summary_path=args.paper_report_summary_path,
        workflow_summary_path=args.workflow_summary_path,
    )
    options = WorkflowOptions(
        model_kind=args.model_kind,
        model_path_explicit=args.model_path is not None,
        stop_mult=args.stop_mult,
        target_mult=args.target_mult,
        pretrain_epochs=args.pretrain_epochs,
        pretrain_gap_bars=args.pretrain_gap_bars,
        head_finetune_epochs=args.head_finetune_epochs,
        incumbent_calibration_report_path=args.incumbent_calibration_report,
        incumbent_report_summary_path=args.incumbent_report_summary_path,
        incumbent_paper_report_summary_path=args.incumbent_paper_report_summary_path,
        paper=args.paper,
        paper_config_path=args.paper_config,
        strategy=args.strategy,
        walk_forward=args.walk_forward,
        all_strategies=args.all_strategies,
        skip_validation=args.skip_validation,
        skip_labeling=args.skip_labeling,
        skip_calibration=args.skip_calibration,
        skip_backtest=args.skip_backtest,
        skip_report=args.skip_report,
        dry_run=args.dry_run,
    )
    run_workflow(paths=paths, options=options)


if __name__ == "__main__":
    main()
