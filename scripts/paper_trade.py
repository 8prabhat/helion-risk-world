"""Dry-run paper trading with full decision audit (SPEC.md §24).

Usage:
    python scripts/paper_trade.py --config <cfg> [--seed N] [--dry-run]
"""
from __future__ import annotations

import json
from pathlib import Path

from _backtest_runtime import build_steps_for_run, predictor_kind_for_run
from _common import check_calibration_gate, log, setup
from helion_risk_world.config.loaders import (
    data_config_from_mapping as data_config_from_cfg,
    execution_config_from_mapping as execution_config_from_cfg,
    risk_profile_name_from_mapping as risk_profile_name_from_cfg,
    strategy_profile_from_mapping as strategy_profile_from_cfg,
)

from helion_risk_world.backtesting.backtest_engine import BacktestEngine
from helion_risk_world.backtesting.transaction_costs import TransactionCosts
from helion_risk_world.config.risk_profiles import load_account_risk_profile
from helion_risk_world.execution import assert_entry_feasible
from helion_risk_world.memory import CalibrationMonitor, DataFreshnessMonitor, DriftMonitor
from helion_risk_world.paper_trading import (
    DecisionLogger,
    DryRunBrokerAdapter,
    ExecutionLogger,
    ExecutionRealityBrokerAdapter,
    PaperTradingEngine,
)
from helion_risk_world.schemas.portfolio_schema import PortfolioState
from helion_risk_world.strategy import StrategyPlanner


def resolve_data_monitor(paper_cfg: dict) -> DataFreshnessMonitor | None:
    """Build the paper-trading DataFreshnessMonitor from config.

    Fail-safe by default (review finding H10): omitting `data_monitor` from a
    paper config used to silently run with zero staleness/point-in-time gating.
    Default to on; set `paper.data_monitor: false` explicitly to opt out.
    (`calibration_monitor`/`drift_monitor` are left opt-in in the caller —
    `drift_monitor` in particular runs a full baseline backtest as a side
    effect, which is not free to enable silently.)
    """
    data_monitor_cfg = paper_cfg.get("data_monitor", True)
    if isinstance(data_monitor_cfg, dict):
        return DataFreshnessMonitor(
            max_market_staleness_seconds=float(
                data_monitor_cfg.get("max_market_staleness_seconds", 300.0)
            ),
            max_prediction_skew_seconds=float(
                data_monitor_cfg.get("max_prediction_skew_seconds", 60.0)
            ),
            require_quotes=bool(data_monitor_cfg.get("require_quotes", False)),
            alert_threshold=float(data_monitor_cfg.get("alert_threshold", 0.1)),
        )
    if data_monitor_cfg:
        return DataFreshnessMonitor()
    return None


def main() -> None:
    args, cfg = setup(
        "Dry-run paper trading with full decision audit (SPEC.md §24).",
        option_groups=(
            "real",
            "data_dir",
            "model_flag",
            "model_path",
            "strategy",
            "all_strategies",
            "calibration_gate",
            "persist_state",
        ),
    )
    if args.all_strategies:
        log.error(
            "paper_trade.multi_strategy_unsupported note=%s",
            "Paper trading currently runs one strategy at a time.",
        )
        return
    dc = data_config_from_cfg(cfg)
    strategy = strategy_profile_from_cfg(cfg, args.strategy)
    horizon = strategy.decision_horizon_bars
    try:
        predictor_kind = predictor_kind_for_run(args)
    except ValueError as exc:
        log.error("paper_trade.invalid_setup note=%s", str(exc))
        return
    if args.dry_run:
        log.info(
            "paper_trade.dry_run",
            strategy=strategy.name.value,
            horizon_bars=horizon,
            real=bool(args.real),
            predictor=predictor_kind,
        )
        return
    # Review finding M13: gate real execution (not the diagnostic --dry-run path
    # above) against a Stage-5 calibration failure, closing the loophole where
    # running this script directly (bypassing train_workflow.py's orchestration
    # order) never actually checked calibration status.
    if not check_calibration_gate(args):
        return
    try:
        steps, predictor_kind = build_steps_for_run(dc, cfg, args, horizon)
    except ValueError as exc:
        log.error("paper_trade.invalid_setup note=%s", str(exc))
        return
    paper_cfg = cfg.get("paper", {}) if isinstance(cfg, dict) else {}
    cost_cfg = execution_config_from_cfg(cfg)
    account_profile = load_account_risk_profile(risk_profile_name_from_cfg(cfg, "paper"))
    capital0 = account_profile.capital0
    effective_risk = strategy.apply_risk(account_profile.risk)
    try:
        assert_entry_feasible(
            [step.execution_market or step.market for step in steps],
            capital=capital0,
            max_exposure=effective_risk.max_exposure,
            cost_cfg=cost_cfg,
        )
    except ValueError as exc:
        log.error("paper_trade.untradeable_setup note=%s", str(exc))
        return
    broker_mode = str(paper_cfg.get("broker_adapter", "execution_reality"))
    seed = int(cfg.get("seed", 7)) if isinstance(cfg, dict) else 7
    if broker_mode == "execution_reality":
        broker = ExecutionRealityBrokerAdapter(cost_cfg=cost_cfg, seed=seed)
    elif broker_mode == "dry_run":
        broker = DryRunBrokerAdapter()
    else:
        log.error("paper_trade.invalid_broker_adapter note=%s", f"unsupported broker adapter: {broker_mode}")
        return
    calibration_monitor = CalibrationMonitor() if paper_cfg.get("calibration_monitor", False) else None
    data_monitor = resolve_data_monitor(paper_cfg)
    drift_monitor = DriftMonitor() if paper_cfg.get("drift_monitor", False) else None
    baseline_stats = None
    if drift_monitor is not None:
        baseline_planner = StrategyPlanner.default(strategy, cost_cfg=cost_cfg)
        baseline_report = BacktestEngine(
            planner=baseline_planner,
            costs=TransactionCosts(cost_cfg),
        ).run(
            steps,
            PortfolioState(
                ts=steps[0].prediction.ts,
                capital0=capital0,
                capital=capital0,
                cash=capital0,
                free_margin=capital0,
            ),
            account_profile.risk,
        )
        baseline_stats = baseline_report.summary()
    engine = PaperTradingEngine(
        planner=StrategyPlanner.default(strategy, cost_cfg=cost_cfg),
        broker=broker,
        decision_logger=DecisionLogger(
            paper_cfg.get("decision_log_path", "runs/paper/decisions.jsonl")
        ),
        execution_logger=ExecutionLogger(
            paper_cfg.get("execution_log_path", "runs/paper/executions.jsonl")
        ),
        calibration_monitor=calibration_monitor,
        data_monitor=data_monitor,
        drift_monitor=drift_monitor,
        backtest_stats=baseline_stats,
    )
    state = PortfolioState(
        ts=steps[0].prediction.ts,
        capital0=capital0,
        capital=capital0,
        cash=capital0,
        free_margin=capital0,
    )
    log.info(
        "paper_trade.strategy",
        strategy=strategy.name.value,
        horizon_bars=horizon,
        predictor=predictor_kind,
    )
    for step in steps:
        payload = {"prediction": step.prediction, "market": step.market, "realized_return": step.realized_return}
        state = engine.on_bar(payload, state, account_profile.risk)
    monitor_summary = engine.monitor_snapshot()
    if monitor_summary:
        log.info("paper_trade.monitor", **monitor_summary)
        report_path = Path(paper_cfg.get("monitor_report_path", "runs/paper/monitor_summary.json"))
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(monitor_summary, indent=2, default=str), encoding="utf-8")
    log.info("paper_trade.done", final_capital=round(state.capital, 2), trades=state.trades_today)


if __name__ == "__main__":
    main()
