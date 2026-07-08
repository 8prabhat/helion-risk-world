"""Run a leakage-checked backtest with either a heuristic or a trained forecaster."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from _common import check_calibration_gate, log, setup
from _backtest_runtime import build_steps_for_run
from helion_risk_world.config.loaders import (
    data_config_from_mapping as data_config_from_cfg,
    risk_profile_name_from_mapping as risk_profile_name_from_cfg,
    strategy_profile_from_mapping as strategy_profile_from_cfg,
    walk_forward_from_mapping as walk_forward_from_cfg,
)

from helion_risk_world.backtesting.backtest_engine import BacktestEngine
from helion_risk_world.backtesting.leakage_report import LeakageReport
from helion_risk_world.backtesting.strategy_diagnostics import evaluate_backtest_report
from helion_risk_world.backtesting.strategy_comparison import (
    StrategyBacktestCase,
    StrategyComparisonRunner,
)
from helion_risk_world.backtesting.transaction_costs import TransactionCosts
from helion_risk_world.backtesting.walk_forward_evaluation import WalkForwardStrategyRunner
from helion_risk_world.config.execution_config import CostModelConfig
from helion_risk_world.config.risk_profiles import load_account_risk_profile
from helion_risk_world.data.market_window_builder import CANDLE_FEATURE_NAMES
from helion_risk_world.schemas.portfolio_schema import PortfolioState
from helion_risk_world.strategy import (
    StrategyPlanner,
    available_strategy_names,
    get_strategy_profile,
)


def _write_decision_audit(path: Path, decisions: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for decision in decisions:
            fh.write(decision.model_dump_json() + "\n")


def _single_backtest_payload(
    strategy_name: str,
    predictor: str,
    report,
    leakage: dict[str, object] | None = None,
    n_bootstrap: int = 1000,
    random_trials: int = 32,
) -> dict[str, object]:
    return {
        "strategy": strategy_name,
        "predictor": predictor,
        "leakage": leakage or {},
        "summary": report.summary(),
        "diagnostics": evaluate_backtest_report(
            report,
            n_bootstrap=n_bootstrap,
            random_trials=random_trials,
        ),
    }


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value == float("inf") or value == float("-inf"):
            return str(value)
        return f"{value:.6g}"
    if value is None:
        return "NA"
    return str(value)


def _print_rows(title: str, rows: list[tuple[str, Any]]) -> None:
    print(f"\n{title}", flush=False)
    print("-" * len(title), flush=False)
    for key, value in rows:
        print(f"{key}: {_fmt(value)}", flush=False)
    print(flush=True)


def _promotion_failures(checks: dict[str, object]) -> str:
    failures = [f"{name}={status}" for name, status in checks.items() if str(status).startswith("FAIL")]
    return "; ".join(failures) if failures else "none"


def _print_single_backtest_summary(payload: dict[str, object]) -> None:
    summary = payload.get("summary", {})
    diagnostics = payload.get("diagnostics", {})
    leakage = payload.get("leakage", {})
    if not isinstance(summary, dict):
        summary = {}
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    if not isinstance(leakage, dict):
        leakage = {}
    cost = diagnostics.get("cost_sensitivity", {})
    quality = diagnostics.get("trading_quality", {})
    checks = diagnostics.get("promotion_checks", {})
    if not isinstance(cost, dict):
        cost = {}
    if not isinstance(quality, dict):
        quality = {}
    if not isinstance(checks, dict):
        checks = {}
    _print_rows(
        "BACKTEST SUMMARY",
        [
            ("strategy", payload.get("strategy")),
            ("predictor", payload.get("predictor")),
            ("leakage_passed", leakage.get("passed")),
            ("leakage_feature_rows_checked", leakage.get("n_feature_rows_checked")),
            ("leakage_labels_checked", leakage.get("n_labels_checked")),
            ("n_steps", summary.get("n_steps")),
            ("n_trades", summary.get("n_trades")),
            ("no_trade_fraction", summary.get("no_trade_fraction")),
            ("total_return", summary.get("total_return")),
            ("annualized_return", summary.get("annualized_return")),
            ("sharpe_current_cost_model", summary.get("sharpe")),
            ("sharpe_at_5bps", cost.get("sharpe_at_5bps")),
            ("sharpe_at_25bps", cost.get("sharpe_at_25bps")),
            ("total_return_at_5bps", cost.get("total_return_at_5bps")),
            ("total_return_at_25bps", cost.get("total_return_at_25bps")),
            ("max_drawdown", summary.get("max_drawdown")),
            ("max_drawdown_at_25bps", cost.get("max_drawdown_at_25bps")),
            ("hit_rate", summary.get("hit_rate")),
            ("profit_factor", summary.get("profit_factor")),
            ("average_trade_return", quality.get("average_trade_return")),
            ("trade_frequency_per_day", quality.get("trade_frequency_per_day")),
            ("turnover", summary.get("turnover")),
            ("capital_utilization", summary.get("capital_utilization")),
            ("promotion_failures", _promotion_failures(checks)),
        ],
    )


def _summary_for_strategy_payload(strategy_payload: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    if "out_of_sample" in strategy_payload:
        summary = strategy_payload.get("out_of_sample", {})
        diagnostics = strategy_payload.get("diagnostics", {})
        if isinstance(diagnostics, dict):
            diagnostics = diagnostics.get("out_of_sample", {})
    else:
        summary = strategy_payload.get("summary", {})
        diagnostics = strategy_payload.get("diagnostics", {})
    return (
        summary if isinstance(summary, dict) else {},
        diagnostics if isinstance(diagnostics, dict) else {},
    )


def _print_multi_backtest_summary(title: str, payload: dict[str, object]) -> None:
    strategies = payload.get("strategies", [])
    leakage = payload.get("leakage", {})
    leakage_passed = all(
        bool(item.get("passed", False))
        for item in leakage.values()
        if isinstance(item, dict)
    )
    _print_rows(
        title,
        [
            ("ranking_metric", payload.get("ranking_metric")),
            ("best_strategy", payload.get("best_strategy")),
            ("n_strategies", len(strategies) if isinstance(strategies, list) else 0),
            ("leakage_passed", leakage_passed),
        ],
    )
    if not isinstance(strategies, list):
        return
    for strategy_payload in strategies:
        if not isinstance(strategy_payload, dict):
            continue
        summary, diagnostics = _summary_for_strategy_payload(strategy_payload)
        cost = diagnostics.get("cost_sensitivity", {})
        checks = diagnostics.get("promotion_checks", {})
        quality = diagnostics.get("trading_quality", {})
        if not isinstance(cost, dict):
            cost = {}
        if not isinstance(checks, dict):
            checks = {}
        if not isinstance(quality, dict):
            quality = {}
        print(
            "strategy={strategy} n_trades={n_trades} no_trade_fraction={no_trade} "
            "total_return={total_return} sharpe={sharpe} sharpe_at_5bps={sharpe5} "
            "sharpe_at_25bps={sharpe25} max_drawdown={max_dd} profit_factor={pf} "
            "avg_trade_return={avg_trade} promotion_failures={failures}".format(
                strategy=strategy_payload.get("strategy"),
                n_trades=_fmt(summary.get("n_trades")),
                no_trade=_fmt(summary.get("no_trade_fraction")),
                total_return=_fmt(summary.get("total_return")),
                sharpe=_fmt(summary.get("sharpe")),
                sharpe5=_fmt(cost.get("sharpe_at_5bps")),
                sharpe25=_fmt(cost.get("sharpe_at_25bps")),
                max_dd=_fmt(summary.get("max_drawdown")),
                pf=_fmt(summary.get("profit_factor")),
                avg_trade=_fmt(quality.get("average_trade_return")),
                failures=_promotion_failures(checks),
            ),
            flush=True,
        )


def main() -> None:
    args, cfg = setup(
        "Run a leakage-checked backtest.",
        option_groups=(
            "demo",
            "real",
            "data_dir",
            "model_flag",
            "model_path",
            "strategy",
            "all_strategies",
            "walk_forward",
            "calibration_gate",
            "persist_state",
        ),
    )
    if not args.demo and not args.real:
        log.warning("backtest.no_source note=%s", "Run with --demo or --real --data-dir.")
        return
    if args.strategy and args.all_strategies:
        log.error("backtest.strategy_conflict note=%s", "Use either --strategy or --all-strategies, not both.")
        return
    if args.all_strategies and args.model and args.model_path:
        log.error(
            "backtest.multi_strategy_artifact_unsupported note=%s",
            "A single-horizon forecaster artifact cannot drive all strategies. Use heuristic or demo-trained mode.",
        )
        return
    if not check_calibration_gate(args):
        return

    diagnostic_n_bootstrap = 1 if args.dry_run else 1000
    diagnostic_random_trials = 1 if args.dry_run else 32
    dc = data_config_from_cfg(cfg)
    account_profile = load_account_risk_profile(risk_profile_name_from_cfg(cfg, "backtest"))
    strategy_names = (
        available_strategy_names()
        if args.all_strategies
        else (strategy_profile_from_cfg(cfg, args.strategy).name.value,)
    )
    strategies = [get_strategy_profile(name) for name in strategy_names]
    cases: list[StrategyBacktestCase] = []
    predictor_kinds: dict[str, str] = {}
    leakage_reports: dict[str, dict[str, object]] = {}
    for strategy in strategies:
        horizon = strategy.decision_horizon_bars
        try:
            steps, predictor_kind = build_steps_for_run(dc, cfg, args, horizon)
        except ValueError as exc:
            log.error("backtest.invalid_setup strategy=%s note=%s", strategy.name.value, str(exc))
            _print_rows(
                "BACKTEST SETUP FAILED",
                [
                    ("strategy", strategy.name.value),
                    ("horizon_bars", horizon),
                    ("reason", str(exc)),
                ],
            )
            return
        # Leakage-checked per strategy (review finding H8): previously called once
        # with no feature_rows/labels, so the real point-in-time and label-future
        # checks were silently skipped — only the static feature-name check ran.
        # Horizons differ per strategy, so this must run per-case, not once globally.
        feature_rows = [
            {"ts": step.market.ts, "available_at": step.market.available_at} for step in steps
        ]
        labels = [
            (step.prediction.ts, step.label_realized_at)
            for step in steps
            if step.label_realized_at is not None
        ]
        leakage_reports[strategy.name.value] = LeakageReport().run(
            list(CANDLE_FEATURE_NAMES),
            feature_rows=feature_rows,
            labels=labels,
        )
        cases.append(StrategyBacktestCase(profile=strategy, steps=steps))
        predictor_kinds[strategy.name.value] = predictor_kind
        log.info(
            "backtest.case",
            strategy=strategy.name.value,
            horizon_bars=horizon,
            predictor=predictor_kind,
            n_steps=len(steps),
        )

    first_steps = cases[0].steps
    capital0 = account_profile.capital0
    account = PortfolioState(
        ts=first_steps[0].prediction.ts,
        capital0=capital0,
        capital=capital0,
        cash=capital0,
        free_margin=capital0,
    )
    cost_cfg = CostModelConfig()
    walk_forward = walk_forward_from_cfg(cfg)
    max_case_horizon = max((case.profile.decision_horizon_bars for case in cases), default=0)
    if args.walk_forward and walk_forward.embargo_bars < max_case_horizon:
        # LOW item: embargo_bars is supposed to be >= the largest horizon under
        # evaluation so a fold's test labels never bleed into the next fold's
        # train set. scripts/train.py already clamps this for the main training
        # config; the walk-forward path reads backtest.embargo_bars separately
        # and had no equivalent check.
        log.warning(
            "backtest.embargo_bars_below_horizon embargo_bars=%s max_horizon_bars=%s",
            walk_forward.embargo_bars,
            max_case_horizon,
        )

    if args.walk_forward:
        comparison = WalkForwardStrategyRunner.default(
            walk_forward,
            cost_cfg=cost_cfg,
        ).run(
            cases,
            account,
            account_profile.risk,
        )
        log.info(
            "backtest.walk_forward",
            ranking_metric=comparison.ranking_metric,
            folds=max((len(result.folds) for result in comparison.results), default=0),
            strategies=comparison.summary_rows(),
        )
        payload = comparison.to_dict(
            n_bootstrap=diagnostic_n_bootstrap,
            random_trials=diagnostic_random_trials,
        )
        payload["leakage"] = leakage_reports
        _print_multi_backtest_summary("BACKTEST WALK-FORWARD SUMMARY", payload)
        if args.dry_run:
            return
        out = Path("runs/backtest")
        out.mkdir(parents=True, exist_ok=True)
        wf_path = out / "walk_forward.json"
        wf_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.info(
            "backtest.walk_forward_written",
            path=str(wf_path),
            n_strategies=len(comparison.results),
        )
        return

    if args.all_strategies:
        comparison = StrategyComparisonRunner.default(cost_cfg=cost_cfg).run(
            cases,
            account,
            account_profile.risk,
        )
        log.info(
            "backtest.comparison",
            ranking_metric=comparison.ranking_metric,
            strategies=comparison.summary_rows(),
        )
        payload = comparison.to_dict(
            n_bootstrap=diagnostic_n_bootstrap,
            random_trials=diagnostic_random_trials,
        )
        payload["leakage"] = leakage_reports
        _print_multi_backtest_summary("BACKTEST STRATEGY COMPARISON", payload)
        if args.dry_run:
            return
        out = Path("runs/backtest")
        out.mkdir(parents=True, exist_ok=True)
        comparison_path = out / "strategy_comparison.json"
        comparison_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        for result in comparison.results:
            _write_decision_audit(out / result.strategy_name / "decisions.jsonl", result.report.decisions)
        log.info(
            "backtest.comparison_written",
            path=str(comparison_path),
            n_strategies=len(comparison.results),
        )
        return

    strategy = strategies[0]
    report = BacktestEngine(
        StrategyPlanner.default(strategy, cost_cfg=cost_cfg),
        TransactionCosts(cost_cfg),
    ).run(
        first_steps,
        account,
        account_profile.risk,
    )
    log.info("backtest.report", strategy=strategy.name.value, predictor=predictor_kinds[strategy.name.value], **report.summary())

    payload = _single_backtest_payload(
        strategy.name.value,
        predictor_kinds[strategy.name.value],
        report,
        leakage_reports.get(strategy.name.value),
        n_bootstrap=diagnostic_n_bootstrap,
        random_trials=diagnostic_random_trials,
    )
    _print_single_backtest_summary(payload)
    if args.dry_run:
        return
    out = Path("runs/backtest")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "decisions.jsonl"
    _write_decision_audit(path, report.decisions)
    report_path = out / "backtest_report.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("backtest.audit_written", path=str(path), n=len(report.decisions), strategy=strategy.name.value)
    log.info("backtest.summary_written", path=str(report_path), strategy=strategy.name.value)


if __name__ == "__main__":
    main()
