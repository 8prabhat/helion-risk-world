"""Dry-run paper trading/logging path."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path
import json

from helion_risk_world.config.execution_config import CostModelConfig
from helion_risk_world.memory import CalibrationMonitor, DataFreshnessMonitor, DriftMonitor
from helion_risk_world.paper_trading import (
    DecisionLogger,
    DryRunBrokerAdapter,
    ExecutionLogger,
    ExecutionRealityBrokerAdapter,
    PaperFill,
    PaperTradingEngine,
)
from helion_risk_world.config.planner_config import PlannerConfig
from helion_risk_world.planner.mpc_planner import MPCPlanner
from helion_risk_world.schemas.action_schema import ActionType, CandidateAction
from helion_risk_world.schemas.portfolio_schema import PortfolioState, PositionSide, RiskProfile
from helion_risk_world.schemas.prediction_schema import BarrierProbabilities, HorizonPrediction, ModelPrediction
from helion_risk_world.schemas.execution_schema import ExecutionState

TS = datetime(2026, 6, 16, 10, 0)


def _prediction() -> ModelPrediction:
    hp = HorizonPrediction(
        horizon_bars=3,
        return_quantiles={0.1: 0.01, 0.25: 0.025, 0.5: 0.04, 0.75: 0.055, 0.9: 0.07},
        volatility=0.01,
    )
    return ModelPrediction(
        symbol="BANKNIFTY",
        ts=TS,
        horizon_preds=[hp],
        barrier=BarrierProbabilities(stop=0.05, target=0.85, timeout=0.10),
        mae=0.02,
        sigma_H=0.01,
        epistemic=0.0,
        aleatoric=0.01,
        ood_score=0.0,
    )


def _exit_prediction() -> ModelPrediction:
    hp = HorizonPrediction(
        horizon_bars=3,
        return_quantiles={0.1: -0.06, 0.25: -0.03, 0.5: -0.01, 0.75: 0.0, 0.9: 0.01},
        volatility=0.02,
    )
    return ModelPrediction(
        symbol="BANKNIFTY",
        ts=TS,
        horizon_preds=[hp],
        barrier=BarrierProbabilities(stop=0.7, target=0.1, timeout=0.2),
        mae=0.02,
        sigma_H=0.02,
        epistemic=0.0,
        aleatoric=0.01,
        ood_score=0.0,
    )


def _risk() -> RiskProfile:
    return RiskProfile(
        name="balanced", max_risk_per_trade=0.01, max_daily_loss=0.02, max_weekly_loss=0.05,
        max_drawdown=0.10, max_exposure=1.0, max_trades_per_day=100, consecutive_loss_cooldown=99,
        cvar_alpha=0.05, n_paths=256,
    )


def test_paper_engine_logs_and_updates_state(tmp_path) -> None:
    engine = PaperTradingEngine(
        planner=MPCPlanner.default(planner_cfg=PlannerConfig(risk_aversion_lambda=0.5)),
        broker=DryRunBrokerAdapter(),
        decision_logger=DecisionLogger(tmp_path / "decisions.jsonl"),
        execution_logger=ExecutionLogger(tmp_path / "executions.jsonl"),
    )
    state = PortfolioState(ts=TS, capital0=500_000.0, capital=500_000.0, cash=500_000.0, free_margin=500_000.0)
    risk = _risk()
    market = ExecutionState(symbol="BANKNIFTY", ts=TS, available_at=TS, bid=99.9, ask=100.1, spread=0.2)
    nxt = engine.on_bar(
        {"prediction": _prediction(), "market": market, "realized_return": 0.01},
        state,
        risk,
    )
    assert (tmp_path / "decisions.jsonl").exists()
    assert (tmp_path / "executions.jsonl").exists()
    assert nxt.capital != state.capital or nxt.position != state.position


def test_execution_reality_broker_charges_costs() -> None:
    broker = ExecutionRealityBrokerAdapter(cost_cfg=CostModelConfig(), seed=7)
    fill = broker.place(
        CandidateAction(action_type=ActionType.ENTER_LONG, size_fraction=0.5),
        market=ExecutionState(symbol="BANKNIFTY", ts=TS, available_at=TS, bid=99.9, ask=100.1, spread=0.2, depth=1000.0),
        portfolio_state=PortfolioState(
            ts=TS,
            capital0=500_000.0,
            capital=500_000.0,
            cash=500_000.0,
            free_margin=500_000.0,
        ),
        max_exposure=1.0,
        expected_edge=5_000.0,
    )
    assert fill.status in {"accepted", "partial"}
    assert fill.cost > 0.0
    assert fill.executed_notional > 0.0
    assert fill.execution_realism in {"high", "medium", "low"}


class RejectingBroker:
    def place(self, action, **kwargs):
        return PaperFill(
            requested_action=action,
            executed_action=CandidateAction(action_type=ActionType.NO_TRADE, size_fraction=0.0),
            status="rejected",
            note="test_reject",
        )

    def positions(self):
        return []


class RecordingBroker:
    def __init__(self) -> None:
        self.last_market = None
        self._fills: list[PaperFill] = []

    def place(self, action, **kwargs):
        self.last_market = kwargs["market"]
        fill = PaperFill(
            requested_action=action,
            executed_action=action,
            status="accepted",
            executed_notional=100_000.0,
            cost=10.0,
        )
        self._fills.append(fill)
        return fill

    def positions(self):
        return list(self._fills)


def test_paper_engine_uses_executed_action_and_monitor_snapshot(tmp_path) -> None:
    engine = PaperTradingEngine(
        planner=MPCPlanner.default(planner_cfg=PlannerConfig(risk_aversion_lambda=0.5)),
        broker=RejectingBroker(),
        decision_logger=DecisionLogger(tmp_path / "decisions.jsonl"),
        execution_logger=ExecutionLogger(tmp_path / "executions.jsonl"),
        calibration_monitor=CalibrationMonitor(min_samples=1),
        data_monitor=DataFreshnessMonitor(),
        drift_monitor=DriftMonitor(alert_threshold=0.0),
        backtest_stats={"net_pnl": 10.0, "sharpe": 1.0, "max_drawdown": 0.01},
    )
    state = PortfolioState(ts=TS, capital0=500_000.0, capital=500_000.0, cash=500_000.0, free_margin=500_000.0)
    risk = _risk()
    market = ExecutionState(symbol="BANKNIFTY", ts=TS, available_at=TS, bid=99.9, ask=100.1, spread=0.2)
    nxt = engine.on_bar(
        {"prediction": _prediction(), "market": market, "realized_return": 0.01},
        state,
        risk,
    )
    assert nxt.capital == state.capital
    payload = json.loads((tmp_path / "decisions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert payload["execution_status"] == "rejected"
    assert payload["executed_action"]["action_type"] == "no_trade"
    assert payload["data_quality_status"]["status"] == "ok"
    monitor = engine.monitor_snapshot()
    assert "calibration" in monitor
    assert "data_quality" in monitor
    assert "drift" in monitor


def test_paper_engine_blocks_risk_increase_on_data_alert(tmp_path) -> None:
    engine = PaperTradingEngine(
        planner=MPCPlanner.default(planner_cfg=PlannerConfig(risk_aversion_lambda=0.5)),
        broker=DryRunBrokerAdapter(),
        decision_logger=DecisionLogger(tmp_path / "decisions.jsonl"),
        execution_logger=ExecutionLogger(tmp_path / "executions.jsonl"),
        data_monitor=DataFreshnessMonitor(
            max_market_staleness_seconds=60.0,
            max_prediction_skew_seconds=30.0,
            require_quotes=True,
        ),
    )
    state = PortfolioState(ts=TS, capital0=500_000.0, capital=500_000.0, cash=500_000.0, free_margin=500_000.0)
    market = ExecutionState(
        symbol="BANKNIFTY",
        ts=TS.replace(minute=2),
        available_at=TS,
        bid=None,
        ask=100.1,
        spread=None,
    )

    nxt = engine.on_bar(
        {"prediction": _prediction(), "market": market, "realized_return": 0.01},
        state,
        _risk(),
    )

    payload = json.loads((tmp_path / "decisions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert nxt.capital == state.capital
    assert payload["final_action"]["action_type"] == "enter_long"
    assert payload["executed_action"]["action_type"] == "no_trade"
    assert payload["execution_status"] == "data_blocked"
    assert payload["data_quality_status"]["execution_override"] == "blocked_risk_increase"
    assert engine.monitor_snapshot()["data_quality"]["blocked_risk_increase_count"] == 1.0


def test_paper_engine_allows_exit_under_data_alert(tmp_path) -> None:
    engine = PaperTradingEngine(
        planner=MPCPlanner.default(planner_cfg=PlannerConfig(risk_aversion_lambda=0.5)),
        broker=DryRunBrokerAdapter(),
        decision_logger=DecisionLogger(tmp_path / "decisions.jsonl"),
        execution_logger=ExecutionLogger(tmp_path / "executions.jsonl"),
        data_monitor=DataFreshnessMonitor(
            max_market_staleness_seconds=60.0,
            max_prediction_skew_seconds=30.0,
            require_quotes=True,
        ),
    )
    state = PortfolioState(
        ts=TS,
        capital0=500_000.0,
        capital=500_000.0,
        cash=500_000.0,
        free_margin=500_000.0,
        position=PositionSide.LONG,
        exposure=0.5,
        margin_used=250_000.0,
    )
    market = ExecutionState(
        symbol="BANKNIFTY",
        ts=TS.replace(minute=2),
        available_at=TS,
        bid=None,
        ask=100.1,
        spread=None,
    )

    nxt = engine.on_bar(
        {"prediction": _exit_prediction(), "market": market, "realized_return": -0.01},
        state,
        _risk(),
    )

    payload = json.loads((tmp_path / "decisions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert nxt.position is PositionSide.FLAT
    assert payload["final_action"]["action_type"] == "exit"
    assert payload["executed_action"]["action_type"] == "exit"
    assert payload["execution_status"] != "data_blocked"
    assert "execution_override" not in payload["data_quality_status"]


def test_paper_engine_places_broker_order_on_execution_market(tmp_path) -> None:
    broker = RecordingBroker()
    engine = PaperTradingEngine(
        planner=MPCPlanner.default(planner_cfg=PlannerConfig(risk_aversion_lambda=0.5)),
        broker=broker,
        decision_logger=DecisionLogger(tmp_path / "decisions.jsonl"),
        execution_logger=ExecutionLogger(tmp_path / "executions.jsonl"),
    )
    state = PortfolioState(ts=TS, capital0=500_000.0, capital=500_000.0, cash=500_000.0, free_margin=500_000.0)
    decision_market = ExecutionState(symbol="BANKNIFTY", ts=TS, available_at=TS, bid=99.9, ask=100.1, spread=0.2)
    execution_market = ExecutionState(
        symbol="BANKNIFTY_FUT_continuous",
        ts=TS.replace(minute=5),
        available_at=TS.replace(minute=5),
        bid=101.0,
        ask=101.2,
        spread=0.2,
    )

    engine.on_bar(
        {
            "prediction": _prediction(),
            "market": decision_market,
            "execution_market": execution_market,
            "realized_return": 0.01,
        },
        state,
        _risk(),
    )

    assert broker.last_market == execution_market


# ── resolve_data_monitor (review finding H10: fail-safe default) ───────────────

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
_SPEC = importlib.util.spec_from_file_location(
    "paper_trade_script", _ROOT / "scripts" / "paper_trade.py"
)
assert _SPEC is not None and _SPEC.loader is not None
paper_trade_script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(paper_trade_script)


def test_resolve_data_monitor_defaults_on_when_omitted() -> None:
    """Omitting `data_monitor` from a paper config must still construct a
    DataFreshnessMonitor (fail-safe default), not silently run unmonitored."""
    monitor = paper_trade_script.resolve_data_monitor({})
    assert isinstance(monitor, DataFreshnessMonitor)


def test_resolve_data_monitor_explicit_false_disables() -> None:
    monitor = paper_trade_script.resolve_data_monitor({"data_monitor": False})
    assert monitor is None


def test_resolve_data_monitor_dict_config() -> None:
    monitor = paper_trade_script.resolve_data_monitor(
        {"data_monitor": {"max_market_staleness_seconds": 60.0}}
    )
    assert isinstance(monitor, DataFreshnessMonitor)
