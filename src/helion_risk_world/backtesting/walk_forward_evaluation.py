"""Walk-forward evaluation over strategy-specific backtest streams."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import inf

from helion_risk_world.backtesting.backtest_engine import BacktestEngine, BacktestReport, BacktestStep
from helion_risk_world.backtesting.strategy_comparison import StrategyBacktestCase
from helion_risk_world.backtesting.strategy_diagnostics import evaluate_backtest_report
from helion_risk_world.backtesting.transaction_costs import TransactionCosts
from helion_risk_world.backtesting.walk_forward import WalkForward, WalkForwardFold
from helion_risk_world.config.execution_config import CostModelConfig
from helion_risk_world.config.risk_config import RiskShieldConfig
from helion_risk_world.evaluation import trading_metrics as tm
from helion_risk_world.planner.protocols import PlannerProtocol
from helion_risk_world.schemas.action_schema import ActionType, FinalDecision
from helion_risk_world.schemas.portfolio_schema import PortfolioState, RiskProfile
from helion_risk_world.strategy import StrategyPlanner, StrategyProfile


@dataclass(frozen=True)
class WalkForwardFoldReport:
    """One fold's in-sample and out-of-sample results."""

    fold: WalkForwardFold
    in_sample: BacktestReport
    out_of_sample: BacktestReport

    def summary(self) -> dict[str, object]:
        return {
            "fold_id": self.fold.fold_id,
            "ranges": self.fold.as_dict(),
            "in_sample": self.in_sample.summary(),
            "out_of_sample": self.out_of_sample.summary(),
        }


@dataclass(frozen=True)
class WalkForwardStrategyResult:
    """Walk-forward results for one strategy."""

    strategy_name: str
    description: str
    folds: tuple[WalkForwardFoldReport, ...]
    in_sample_report: BacktestReport
    out_of_sample_report: BacktestReport

    def summary(self) -> dict[str, object]:
        return {
            "strategy": self.strategy_name,
            "in_sample": self.in_sample_report.summary(),
            "out_of_sample": self.out_of_sample_report.summary(),
            "n_folds": len(self.folds),
        }

    def to_dict(
        self,
        *,
        n_bootstrap: int = 1000,
        random_trials: int = 32,
    ) -> dict[str, object]:
        return self.to_dict_with_trials(
            1,
            n_bootstrap=n_bootstrap,
            random_trials=random_trials,
        )

    def to_dict_with_trials(
        self,
        n_trials: int,
        *,
        n_bootstrap: int = 1000,
        random_trials: int = 32,
    ) -> dict[str, object]:
        payload = self.summary()
        payload["folds"] = [fold.summary() for fold in self.folds]
        payload["diagnostics"] = {
            "in_sample": evaluate_backtest_report(
                self.in_sample_report,
                n_trials=n_trials,
                n_bootstrap=n_bootstrap,
                random_trials=random_trials,
            ),
            "out_of_sample": evaluate_backtest_report(
                self.out_of_sample_report,
                n_trials=n_trials,
                n_bootstrap=n_bootstrap,
                random_trials=random_trials,
            ),
        }
        return payload


@dataclass(frozen=True)
class WalkForwardComparisonReport:
    """Out-of-sample comparison across strategies."""

    results: tuple[WalkForwardStrategyResult, ...]
    ranking_metric: str = "sharpe"

    def ranked(self, metric: str | None = None) -> list[WalkForwardStrategyResult]:
        metric_name = metric or self.ranking_metric
        return sorted(
            self.results,
            key=lambda result: self._metric_value(result.out_of_sample_report, metric_name),
            reverse=True,
        )

    def best(self, metric: str | None = None) -> WalkForwardStrategyResult | None:
        ranked = self.ranked(metric)
        return ranked[0] if ranked else None

    def summary_rows(self, metric: str | None = None) -> list[dict[str, object]]:
        _ = metric or self.ranking_metric
        return [result.summary() for result in self.ranked(metric)]

    def to_dict(
        self,
        *,
        n_bootstrap: int = 1000,
        random_trials: int = 32,
    ) -> dict[str, object]:
        best = self.best()
        return {
            "ranking_metric": self.ranking_metric,
            "best_strategy": best.strategy_name if best is not None else None,
            "strategies": [
                result.to_dict_with_trials(
                    len(self.results),
                    n_bootstrap=n_bootstrap,
                    random_trials=random_trials,
                )
                for result in self.ranked()
            ],
        }

    @staticmethod
    def _metric_value(report: BacktestReport, metric: str) -> float:
        value = float(getattr(report, metric))
        return inf if value == inf else value


class WalkForwardStrategyRunner:
    """Run walk-forward in-sample and out-of-sample evaluations for one or more strategies."""

    def __init__(
        self,
        costs: TransactionCosts,
        planner_builder: Callable[[StrategyProfile], PlannerProtocol],
        walk_forward: WalkForward,
    ) -> None:
        self._costs = costs
        self._planner_builder = planner_builder
        self._walk_forward = walk_forward

    @classmethod
    def default(
        cls,
        walk_forward: WalkForward,
        *,
        cost_cfg: CostModelConfig | None = None,
        risk_cfg: RiskShieldConfig | None = None,
    ) -> WalkForwardStrategyRunner:
        cfg = cost_cfg or CostModelConfig()
        return cls(
            costs=TransactionCosts(cfg),
            planner_builder=lambda profile: StrategyPlanner.default(
                profile,
                risk_cfg=risk_cfg,
                cost_cfg=cfg,
            ),
            walk_forward=walk_forward,
        )

    def run(
        self,
        cases: Sequence[StrategyBacktestCase],
        account: PortfolioState,
        risk: RiskProfile,
        *,
        ranking_metric: str = "sharpe",
    ) -> WalkForwardComparisonReport:
        results = tuple(self._run_case(case, account, risk) for case in cases)
        return WalkForwardComparisonReport(results=results, ranking_metric=ranking_metric)

    def _run_case(
        self,
        case: StrategyBacktestCase,
        account: PortfolioState,
        risk: RiskProfile,
    ) -> WalkForwardStrategyResult:
        self._validate_case(case)
        folds = self._walk_forward.split_indices(len(case.steps))
        fold_reports: list[WalkForwardFoldReport] = []
        oos_account = account.model_copy(deep=True)
        oos_engine = BacktestEngine(self._planner_builder(case.profile), self._costs)
        in_sample_reports: list[BacktestReport] = []
        out_of_sample_reports: list[BacktestReport] = []

        for fold in folds:
            train_steps = list(case.steps[fold.train_slice])
            test_steps = list(case.steps[fold.test_slice])
            in_sample = BacktestEngine(
                self._planner_builder(case.profile),
                self._costs,
            ).run(train_steps, account.model_copy(deep=True), risk)
            out_of_sample = oos_engine.run(test_steps, oos_account, risk)
            oos_account = out_of_sample.ending_state or oos_account
            fold_reports.append(
                WalkForwardFoldReport(
                    fold=fold,
                    in_sample=in_sample,
                    out_of_sample=out_of_sample,
                )
            )
            in_sample_reports.append(in_sample)
            out_of_sample_reports.append(out_of_sample)

        return WalkForwardStrategyResult(
            strategy_name=case.profile.name.value,
            description=case.profile.description,
            folds=tuple(fold_reports),
            in_sample_report=_merge_reports(in_sample_reports, account.capital0),
            out_of_sample_report=_merge_reports(out_of_sample_reports, account.capital0),
        )

    @staticmethod
    def _validate_case(case: StrategyBacktestCase) -> None:
        if not case.steps:
            raise ValueError(f"strategy '{case.profile.name.value}' has no backtest steps")
        horizon = case.profile.decision_horizon_bars
        for step in case.steps:
            try:
                step.prediction.horizon(horizon)
            except KeyError as exc:
                raise ValueError(
                    f"strategy '{case.profile.name.value}' requires horizon {horizon}, "
                    f"but step at {step.prediction.ts} does not provide it"
                ) from exc


def _merge_reports(reports: Sequence[BacktestReport], capital0: float) -> BacktestReport:
    if not reports:
        raise ValueError("expected at least one backtest report to merge")

    decisions: list[FinalDecision] = []
    step_returns: list[float] = []
    trade_pnls: list[float] = []
    trade_returns: list[float] = []
    market_returns: list[float] = []
    step_turnover: list[float] = []
    step_cost_returns: list[float] = []
    step_exposure: list[float] = []
    timestamps: list = []
    regime: dict[str, dict[str, float]] = {}
    gross_total = cost_total = 0.0
    turnover_total = 0.0
    n_steps = n_trades = interventions = 0

    for i, report in enumerate(reports):
        decisions.extend(report.decisions)
        step_returns.extend(report.step_returns)
        trade_pnls.extend(report.trade_pnls)
        trade_returns.extend(report.trade_returns)
        market_returns.extend(report.market_returns)
        step_turnover.extend(report.step_turnover)
        step_cost_returns.extend(report.step_cost_returns)
        step_exposure.extend(report.step_exposure)
        gross_total += report.gross_pnl
        cost_total += report.total_cost
        turnover_total += report.turnover
        n_steps += report.n_steps
        n_trades += report.n_trades
        interventions += report.risk_shield_interventions
        if report.timestamps:
            timestamps.extend(report.timestamps if i == 0 else report.timestamps[1:])
        for name, bucket in report.regime_breakdown.items():
            agg = regime.setdefault(name, {"n": 0.0, "net_pnl": 0.0})
            agg["n"] += bucket.get("n", 0.0)
            agg["net_pnl"] += bucket.get("net_pnl", 0.0)

    equity = [capital0]
    capital = capital0
    for step_return in step_returns:
        capital *= 1.0 + step_return
        equity.append(capital)
    net_total = equity[-1] - capital0
    gross_total = net_total + cost_total
    annualized_return = tm.annualized_return(equity, timestamps)
    max_drawdown = tm.max_drawdown(equity)
    return BacktestReport(
        n_steps=n_steps,
        n_trades=n_trades,
        no_trade_fraction=(
            sum(1 for decision in decisions if decision.final_action.action_type is ActionType.NO_TRADE)
            / n_steps
            if n_steps
            else 0.0
        ),
        gross_pnl=gross_total,
        net_pnl=net_total,
        total_cost=cost_total,
        final_capital=equity[-1],
        total_return=(equity[-1] / capital0) - 1.0 if capital0 else 0.0,
        annualized_return=annualized_return,
        sharpe=tm.sharpe(step_returns, periods_per_year=tm.infer_periods_per_year(timestamps)),
        sortino=tm.sortino(step_returns, periods_per_year=tm.infer_periods_per_year(timestamps)),
        max_drawdown=max_drawdown,
        drawdown_duration=tm.drawdown_duration(equity),
        calmar=tm.calmar(annualized_return, max_drawdown),
        hit_rate=tm.hit_rate(trade_pnls),
        profit_factor=tm.profit_factor(trade_pnls),
        payoff_ratio=tm.payoff_ratio(trade_pnls),
        expectancy=tm.expectancy(trade_returns),
        turnover=turnover_total,
        capital_utilization=float(sum(step_exposure) / len(step_exposure)) if step_exposure else 0.0,
        risk_shield_interventions=interventions,
        regime_breakdown=regime,
        decisions=decisions,
        step_returns=step_returns,
        trade_pnls=trade_pnls,
        trade_returns=trade_returns,
        market_returns=market_returns,
        step_turnover=step_turnover,
        step_cost_returns=step_cost_returns,
        step_exposure=step_exposure,
        equity_curve=equity,
        timestamps=timestamps,
        ending_state=reports[-1].ending_state,
    )


__all__ = [
    "WalkForwardComparisonReport",
    "WalkForwardFoldReport",
    "WalkForwardStrategyResult",
    "WalkForwardStrategyRunner",
]
