from __future__ import annotations

from collections.abc import Mapping

from helion_risk_world.evaluation import risk_metrics as rm
from helion_risk_world.evaluation import trading_metrics as tm
from helion_risk_world.memory import CalibrationMonitor, DataFreshnessMonitor, DriftMonitor
from helion_risk_world.paper_trading.broker_adapter_interface import BrokerAdapterProtocol
from helion_risk_world.paper_trading.decision_logger import DecisionLogger
from helion_risk_world.paper_trading.execution_logger import ExecutionLogger
from helion_risk_world.planner.protocols import PlannerProtocol
from helion_risk_world.schemas.action_schema import ActionType, CandidateAction
from helion_risk_world.schemas.execution_schema import ExecutionState
from helion_risk_world.schemas.portfolio_schema import PortfolioState, RiskProfile
from helion_risk_world.schemas.prediction_schema import ModelPrediction
from helion_risk_world.worlds.portfolio_world import PortfolioWorld

_EPS = 1e-9
_RISK_INCREASING = {ActionType.ENTER_LONG, ActionType.ENTER_SHORT, ActionType.INCREASE}


class PaperTradingEngine:
    """Paper-trading loop. NEVER silently executes — every decision is audited (SPEC.md §24).

    DIP: depends on BrokerAdapterProtocol, not a concrete broker. SRP: orchestration only.
    """

    def __init__(self, planner: PlannerProtocol, broker: BrokerAdapterProtocol,
                 decision_logger: DecisionLogger,
                 execution_logger: ExecutionLogger | None = None,
                 calibration_monitor: CalibrationMonitor | None = None,
                 data_monitor: DataFreshnessMonitor | None = None,
                 drift_monitor: DriftMonitor | None = None,
                 backtest_stats: Mapping[str, float] | None = None) -> None:
        self._planner = planner
        self._broker = broker
        self._decision_logger = decision_logger
        self._execution_logger = execution_logger
        self._calibration_monitor = calibration_monitor
        self._data_monitor = data_monitor
        self._drift_monitor = drift_monitor
        self._backtest_stats = dict(backtest_stats or {})
        self._predictions: list[ModelPrediction] = []
        self._outcomes: list[dict[str, object]] = []
        self._step_returns: list[float] = []
        self._trade_returns: list[float] = []
        self._equity: list[float] = []
        self._timestamps: list = []
        self._turnover: list[float] = []
        self._exposure: list[float] = []
        self._slippage: list[float] = []
        self._last_calibration: dict[str, float] = {}
        self._data_blocks = 0

    @staticmethod
    def _extract(
        payload: object,
    ) -> tuple[ModelPrediction, ExecutionState, ExecutionState, float | None]:
        if isinstance(payload, Mapping):
            market = payload["market"]
            return (
                payload["prediction"],
                market,
                payload.get("execution_market", market),
                payload.get("realized_return"),
            )
        prediction = getattr(payload, "prediction")
        market = getattr(payload, "market")
        execution_market = getattr(payload, "execution_market", market)
        realized_return = getattr(payload, "realized_return", None)
        return prediction, market, execution_market, realized_return

    def on_bar(
        self,
        market_state: object,
        portfolio_state: PortfolioState,
        risk_profile: RiskProfile,
    ) -> PortfolioState:
        if not self._equity:
            self._equity = [portfolio_state.capital]
            self._timestamps = [portfolio_state.ts]
        prediction, market, execution_market, realized_return = self._extract(market_state)
        effective_risk = self._planner.adapt_risk(risk_profile)
        data_status = (
            self._data_monitor.observe(prediction, market)
            if self._data_monitor is not None
            else None
        )
        decision = self._planner.plan(prediction, portfolio_state, risk_profile, market)
        execution_action, execution_status, data_status = self._execution_guard(
            decision.final_action,
            data_status,
        )
        fill = self._broker.place(
            execution_action,
            market=execution_market,
            portfolio_state=portfolio_state,
            max_exposure=effective_risk.max_exposure,
            expected_edge=max(0.0, decision.expected_reward * portfolio_state.capital),
        )
        if self._execution_logger is not None:
            self._execution_logger.log(fill)
        executed_action = getattr(fill, "executed_action", decision.final_action)
        if realized_return is None:
            self._decision_logger.log(
                decision.model_copy(
                    update={
                        "executed_action": executed_action,
                        "execution_status": execution_status or getattr(fill, "status", None),
                        "data_quality_status": data_status,
                    }
                )
            )
            return portfolio_state
        prev_capital = portfolio_state.capital
        next_state = PortfolioWorld.apply_fill(
            portfolio_state,
            executed_action,
            realized_return,
            getattr(fill, "cost", 0.0),
            effective_risk.max_exposure,
            market=execution_market,
        )
        step_return = (next_state.capital - prev_capital) / max(prev_capital, _EPS)
        self._decision_logger.log(
            decision.model_copy(
                update={
                    "executed_action": executed_action,
                    "execution_status": execution_status or getattr(fill, "status", None),
                    "data_quality_status": data_status,
                    "paper_result": step_return,
                }
            )
        )
        self._record_monitoring(
            prediction=prediction,
            execution_market=execution_market,
            portfolio_state=portfolio_state,
            next_state=next_state,
            executed_action=executed_action,
            fill=fill,
            realized_return=realized_return,
            step_return=step_return,
        )
        return next_state

    def monitor_snapshot(self) -> dict[str, object]:
        report: dict[str, object] = {}
        if self._last_calibration:
            report["calibration"] = dict(self._last_calibration)
        if self._data_monitor is not None:
            report["data_quality"] = {
                **self._data_monitor.snapshot(),
                "blocked_risk_increase_count": float(self._data_blocks),
            }
        if not self._equity:
            return report
        live_stats = self._live_stats()
        report["live_stats"] = live_stats
        if self._drift_monitor is not None and self._backtest_stats:
            report["drift"] = self._drift_monitor.check(self._backtest_stats, live_stats)
        return report

    def _record_monitoring(
        self,
        *,
        prediction: ModelPrediction,
        execution_market: ExecutionState,
        portfolio_state: PortfolioState,
        next_state: PortfolioState,
        executed_action,
        fill: object,
        realized_return: float,
        step_return: float,
    ) -> None:
        self._predictions.append(prediction)
        self._outcomes.append(
            {
                "realized_return": realized_return,
            }
        )
        self._step_returns.append(step_return)
        self._equity.append(next_state.capital)
        self._timestamps.append(execution_market.ts)
        self._turnover.append(float(getattr(fill, "executed_notional", 0.0)) / max(portfolio_state.capital, _EPS))
        self._exposure.append(float(next_state.exposure))
        self._slippage.append(float(getattr(fill, "slippage", 0.0)))
        if executed_action.action_type is not ActionType.NO_TRADE and getattr(fill, "executed_notional", 0.0) > _EPS:
            self._trade_returns.append(step_return)
        if self._calibration_monitor is not None:
            self._last_calibration = self._calibration_monitor.check(self._predictions, self._outcomes)
            setter = getattr(self._planner, "set_confidence_scale", None)
            if callable(setter):
                setter(self._calibration_monitor.confidence_scale)

    def _live_stats(self) -> dict[str, float]:
        initial_capital = self._equity[0]
        final_capital = self._equity[-1]
        stats = {
            "net_pnl": float(final_capital - initial_capital),
            "final_capital": float(final_capital),
            "total_return": float(final_capital / max(initial_capital, _EPS) - 1.0),
            "annualized_return": tm.annualized_return(self._equity, self._timestamps),
            "total_cost": float(sum(getattr(fill, "cost", 0.0) for fill in self._broker.positions())),
            "sharpe": tm.sharpe(self._step_returns, periods_per_year=tm.infer_periods_per_year(self._timestamps)),
            "sortino": tm.sortino(self._step_returns, periods_per_year=tm.infer_periods_per_year(self._timestamps)),
            "max_drawdown": tm.max_drawdown(self._equity),
            "drawdown_duration": tm.drawdown_duration(self._equity),
            "hit_rate": tm.hit_rate(self._trade_returns),
            "profit_factor": tm.profit_factor(self._trade_returns),
            "payoff_ratio": tm.payoff_ratio(self._trade_returns),
            "expectancy": tm.expectancy(self._trade_returns),
            "turnover": tm.turnover(self._turnover),
            "capital_utilization": float(sum(self._exposure) / len(self._exposure)) if self._exposure else 0.0,
            "avg_slippage": float(sum(self._slippage) / len(self._slippage)) if self._slippage else 0.0,
            "reject_rate": float(
                sum(1 for fill in self._broker.positions() if getattr(fill, "status", "") == "rejected")
                / max(len(self._broker.positions()), 1)
            ),
        }
        stats["calmar"] = tm.calmar(stats["annualized_return"], stats["max_drawdown"])
        stats.update(
            rm.compute(
                step_returns=self._step_returns,
                equity=self._equity,
            )
        )
        return stats

    def _execution_guard(
        self,
        action: CandidateAction,
        data_status: dict[str, float | str] | None,
    ) -> tuple[CandidateAction, str | None, dict[str, float | str] | None]:
        if not _data_alert(data_status) or action.action_type not in _RISK_INCREASING:
            return action, None, data_status
        self._data_blocks += 1
        blocked = CandidateAction(action_type=ActionType.NO_TRADE, size_fraction=0.0)
        updated = dict(data_status or {})
        updated["execution_override"] = "blocked_risk_increase"
        return blocked, "data_blocked", updated


def _data_alert(data_status: dict[str, float | str] | None) -> bool:
    if not data_status:
        return False
    try:
        return float(data_status.get("data_alert", 0.0)) > _EPS
    except (TypeError, ValueError):
        return False
