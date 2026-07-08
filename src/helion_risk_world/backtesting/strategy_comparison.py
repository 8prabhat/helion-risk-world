"""Multi-strategy backtest comparison.

Each strategy may require its own horizon-specific step stream, so comparison is expressed as
multiple backtest cases that reuse the same engine/reporting stack instead of forcing one shared
realized-return series across incompatible trading cadences.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import inf

from helion_risk_world.backtesting.backtest_engine import BacktestEngine, BacktestReport, BacktestStep
from helion_risk_world.backtesting.strategy_diagnostics import evaluate_backtest_report
from helion_risk_world.backtesting.transaction_costs import TransactionCosts
from helion_risk_world.config.execution_config import CostModelConfig
from helion_risk_world.config.risk_config import RiskShieldConfig
from helion_risk_world.planner.protocols import PlannerProtocol
from helion_risk_world.schemas.portfolio_schema import PortfolioState, RiskProfile
from helion_risk_world.strategy import StrategyPlanner, StrategyProfile


@dataclass(frozen=True)
class StrategyBacktestCase:
    """One strategy-specific backtest input set."""

    profile: StrategyProfile
    steps: Sequence[BacktestStep]


@dataclass(frozen=True)
class StrategyComparisonResult:
    """One completed strategy backtest."""

    strategy_name: str
    description: str
    report: BacktestReport

    def summary(self) -> dict[str, float | str]:
        row: dict[str, float | str] = {"strategy": self.strategy_name}
        row.update(self.report.summary())
        return row

    def to_dict(
        self,
        *,
        n_trials: int = 1,
        n_bootstrap: int = 1000,
        random_trials: int = 32,
    ) -> dict[str, object]:
        payload: dict[str, object] = {"strategy": self.strategy_name, "description": self.description}
        payload["summary"] = self.report.summary()
        payload["diagnostics"] = evaluate_backtest_report(
            self.report,
            n_trials=n_trials,
            n_bootstrap=n_bootstrap,
            random_trials=random_trials,
        )
        return payload


@dataclass(frozen=True)
class StrategyComparisonReport:
    """Comparable set of strategy results ranked by one metric."""

    results: tuple[StrategyComparisonResult, ...]
    ranking_metric: str = "sharpe"

    def ranked(self, metric: str | None = None) -> list[StrategyComparisonResult]:
        metric_name = metric or self.ranking_metric
        return sorted(
            self.results,
            key=lambda result: self._metric_value(result.report, metric_name),
            reverse=True,
        )

    def best(self, metric: str | None = None) -> StrategyComparisonResult | None:
        ranked = self.ranked(metric)
        return ranked[0] if ranked else None

    def summary_rows(self, metric: str | None = None) -> list[dict[str, float | str]]:
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
                result.to_dict(
                    n_trials=len(self.results),
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


class StrategyComparisonRunner:
    """Run multiple strategy backtests against their own validated step streams."""

    def __init__(
        self,
        costs: TransactionCosts,
        planner_builder: Callable[[StrategyProfile], PlannerProtocol],
    ) -> None:
        self._costs = costs
        self._planner_builder = planner_builder

    @classmethod
    def default(
        cls,
        *,
        cost_cfg: CostModelConfig | None = None,
        risk_cfg: RiskShieldConfig | None = None,
    ) -> StrategyComparisonRunner:
        cfg = cost_cfg or CostModelConfig()
        return cls(
            costs=TransactionCosts(cfg),
            planner_builder=lambda profile: StrategyPlanner.default(
                profile,
                risk_cfg=risk_cfg,
                cost_cfg=cfg,
            ),
        )

    def run(
        self,
        cases: Sequence[StrategyBacktestCase],
        account: PortfolioState,
        risk: RiskProfile,
        *,
        ranking_metric: str = "sharpe",
    ) -> StrategyComparisonReport:
        results: list[StrategyComparisonResult] = []
        for case in cases:
            self._validate_case(case)
            report = BacktestEngine(
                self._planner_builder(case.profile),
                self._costs,
            ).run(case.steps, account.model_copy(deep=True), risk)
            results.append(
                StrategyComparisonResult(
                    strategy_name=case.profile.name.value,
                    description=case.profile.description,
                    report=report,
                )
            )
        return StrategyComparisonReport(tuple(results), ranking_metric=ranking_metric)

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


__all__ = [
    "StrategyBacktestCase",
    "StrategyComparisonReport",
    "StrategyComparisonResult",
    "StrategyComparisonRunner",
]
