"""Shared CLI helpers (DRY). Used by every scripts/*.py entry point."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from _bootstrap import ensure_src_path

ensure_src_path()

from helion_risk_world.integration import configure_logging, get_logger, load_config  # noqa: E402

log = get_logger("hrw.cli")


OptionBuilder = Callable[[argparse.ArgumentParser], None]


def _add_core_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Path to a YAML config (e.g. configs/v1.yaml).")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan; do not write artifacts.")


def _add_demo_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use a synthetic in-memory data source so the pipeline is runnable without real data.",
    )


def _add_audit_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--audit", default=None, help="Path to a FinalDecision JSONL audit stream.")


def _add_out_path_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out-path", default=None, help="Optional JSON output path for generated summaries.")


def _add_model_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        action="store_true",
        help="Drive the run with a trained HRW model artifact (else a heuristic predictor).",
    )


def _add_real_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--real",
        action="store_true",
        help="Backtest on real OHLCV parquet data (requires --data-dir) instead of synthetic.",
    )


def _add_data_dir_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory with ohlcv/<SYMBOL>_<interval>.parquet (native base interval or 1min).",
    )


def _add_labels_path_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--labels-path", default=None, help="Path to labels.parquet for supervised runs.")


def _add_model_path_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", default=None, help="Path to a saved model artifact.")


def _add_model_kind_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-kind",
        choices=("forecaster", "world_model"),
        default="forecaster",
        help="Artifact family to train. Defaults to the compact forecaster.",
    )


def _add_pretraining_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=None,
        help="Optional Stage-2 self-supervised encoder pretraining epochs.",
    )
    parser.add_argument(
        "--pretrain-gap-bars",
        type=int,
        default=None,
        help="Gap in bars between context and future windows for Stage-2 pretraining.",
    )
    parser.add_argument(
        "--rssm-epochs",
        type=int,
        default=None,
        help="Optional Stage-3 RSSM dynamics epochs for world-model training.",
    )
    parser.add_argument(
        "--head-finetune-epochs",
        type=int,
        default=None,
        help="Optional Stage-4 HeadTrainer fine-tuning epochs after the main fit (review finding H7).",
    )


def _add_strategy_option(parser: argparse.ArgumentParser) -> None:
    from helion_risk_world.strategy import available_strategy_names

    parser.add_argument(
        "--strategy",
        choices=available_strategy_names(),
        default=None,
        help="Override the configured trading style.",
    )


def _add_all_strategies_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--all-strategies",
        action="store_true",
        help="Backtest every built-in strategy profile and compare the reports.",
    )


def _add_walk_forward_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run purged walk-forward evaluation instead of one full backtest slice.",
    )


def _add_persist_state_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no-persist-state",
        dest="persist_state",
        action="store_false",
        default=True,
        help=(
            "For a world_model artifact, reset the RSSM belief state on every "
            "predict_one call instead of carrying it across bars (review finding "
            "H1). Default carries state; pass this flag to reproduce the pre-fix "
            "reset-per-call behavior for A/B comparison. No effect on forecaster "
            "artifacts, which have no RSSM state."
        ),
    )


def _add_stop_target_mode_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--stop-target-mode",
        choices=("barrier_context", "quantile"),
        default="barrier_context",
        help=(
            "How the planner sizes stop/target levels for a candidate trade. "
            "'barrier_context' (default) uses the fixed symmetric BarrierContext "
            "multiplier frozen at training time. 'quantile' (2026-07-16) sizes from "
            "the model's own predicted return-quantile distribution instead -- "
            "asymmetric and regime-adaptive, since it's recomputed every decision. "
            "See ModelPrediction.quantile_stop_return's docstring for the diagnosis "
            "this responds to."
        ),
    )


def _add_risk_aversion_lambda_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--risk-aversion-lambda",
        type=float,
        default=None,
        help=(
            "Override the strategy profile's PlannerConfig.risk_aversion_lambda "
            "(mean-CVaR objective U = E[dW] - lambda*CVaR - cost). Default: use "
            "whatever the chosen strategy profile already specifies (e.g. "
            "medium_frequency=3.0). Diagnostic for sweeping risk-aversion to see "
            "whether a CVaR-dominated zero-trade result is a calibration choice or a "
            "hard floor the model's edge can't clear at any reasonable lambda."
        ),
    )


def _add_eval_split_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--eval-split",
        choices=("train", "val", "test"),
        default="test",
        help=(
            "Which of the artifact's own persisted chronological splits to backtest "
            "against (default: test, the normal held-out window). 'val' is a genuinely "
            "different ~4-month window never used for weight updates (only for "
            "early-stopping/model-selection/post-hoc calibration fitting) -- useful for "
            "checking whether a result is specific to the test window's regime, without "
            "a retrain. 'train' was used to fit weights directly and is NOT a fair "
            "out-of-sample check; only use it to sanity-check general behavior, not to "
            "draw conclusions about generalization."
        ),
    )


def _add_checkpoint_metric_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--checkpoint-metric",
        choices=("loss", "trading_utility"),
        default="loss",
        help=(
            "How HRWTrainer selects the best checkpoint / drives early stopping. "
            "'loss' (default) uses the composite validation loss, unchanged from "
            "before. 'trading_utility' (2026-07-18) instead scores the meta-label "
            "head's own trade/no-trade decision rule against held-out outcomes -- "
            "see training/checkpoint_metrics.py::trading_utility_loss. Requires "
            "labels.parquet to have primary_side/meta_label columns (schema "
            "version >= 9, see labeling/meta_labels.py); otherwise the metric is "
            "always 0.0 (neutral) and checkpoint selection degenerates to 'pick "
            "the first epoch that doesn't regress', so don't use this without "
            "meta-labeled data."
        ),
    )


def _add_calibration_gate_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--calibration-report",
        default=None,
        help=(
            "Path to a calibration_report.json (from scripts/calibrate.py). If given "
            "and its gate.passed is False, refuse to run (review finding M13: previously "
            "the calibration gate was only enforced by train_workflow.py's orchestration "
            "order, so running this script directly against an uncalibrated artifact was "
            "never actually blocked)."
        ),
    )
    parser.add_argument(
        "--allow-uncalibrated",
        action="store_true",
        help="Proceed even if --calibration-report indicates a failed gate.",
    )


_OPTION_BUILDERS: dict[str, OptionBuilder] = {
    "demo": _add_demo_option,
    "audit": _add_audit_option,
    "out_path": _add_out_path_option,
    "model_flag": _add_model_flag,
    "real": _add_real_option,
    "data_dir": _add_data_dir_option,
    "labels_path": _add_labels_path_option,
    "model_path": _add_model_path_option,
    "model_kind": _add_model_kind_option,
    "pretraining": _add_pretraining_options,
    "strategy": _add_strategy_option,
    "all_strategies": _add_all_strategies_option,
    "walk_forward": _add_walk_forward_option,
    "calibration_gate": _add_calibration_gate_options,
    "persist_state": _add_persist_state_option,
    "stop_target_mode": _add_stop_target_mode_option,
    "risk_aversion_lambda": _add_risk_aversion_lambda_option,
    "eval_split": _add_eval_split_option,
    "checkpoint_metric": _add_checkpoint_metric_option,
}


def base_parser(
    description: str,
    *,
    option_groups: Iterable[str] = (),
) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    _add_core_options(p)
    seen: set[str] = set()
    for group in option_groups:
        if group in seen:
            continue
        builder = _OPTION_BUILDERS.get(group)
        if builder is None:
            raise ValueError(f"unsupported CLI option group: {group}")
        builder(p)
        seen.add(group)
    return p


def setup(
    description: str,
    *,
    option_groups: Iterable[str] = (),
) -> tuple[argparse.Namespace, dict]:
    args = base_parser(description, option_groups=option_groups).parse_args()
    configure_logging(args.log_level)
    cfg = load_config(args.config)
    cfg_seed = cfg.get("seed", 7) if isinstance(cfg, dict) else 7
    seed = args.seed if args.seed is not None else cfg_seed
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # numpy optional at CLI bootstrap
        pass
    log.info("hrw.cli.start", description=description, config=args.config, seed=seed,
             dry_run=args.dry_run)
    return args, cfg


def check_calibration_gate(args: argparse.Namespace) -> bool:
    """Return True if it's OK to proceed, False if the calibration gate should
    block this run (review finding M13).

    Requires the ``calibration_gate`` option group (``--calibration-report`` /
    ``--allow-uncalibrated``). Previously the Stage-5 calibration gate was only
    enforced by train_workflow.py's subprocess ordering (calibrate must exit 0
    before the backtest/paper stage runs) — running backtest.py or
    paper_trade.py directly against an uncalibrated artifact was never actually
    blocked. A missing/unreadable report doesn't block (there's nothing to
    enforce); only an explicit ``gate.passed: false`` does, unless overridden.
    """
    report_path = getattr(args, "calibration_report", None)
    if not report_path:
        return True
    path = Path(report_path)
    if not path.exists():
        log.warning("calibration_gate.report_not_found path=%s", path)
        return True
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("calibration_gate.report_unreadable path=%s error=%s", path, exc)
        return True
    passed = bool(payload.get("gate", {}).get("passed", False))
    if passed:
        return True
    if getattr(args, "allow_uncalibrated", False):
        log.warning("calibration_gate.bypassed path=%s", path)
        return True
    log.error(
        "calibration_gate.blocked path=%s note=%s",
        path,
        "gate.passed is False; pass --allow-uncalibrated to override",
    )
    return False
