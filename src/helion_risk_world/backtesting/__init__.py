"""Walk-forward, purged, leakage-checked backtesting (SPEC.md §23). Lazy import."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

_LAZY: dict[str, str] = {
    "BacktestEngine": "backtest_engine",
    "BacktestStep": "backtest_engine",
    "BacktestReport": "backtest_engine",
    "EventStressTest": "event_stress_test",
    "LeakageReport": "leakage_report",
    "evaluate_backtest_report": "strategy_diagnostics",
    "StrategyBacktestCase": "strategy_comparison",
    "StrategyComparisonReport": "strategy_comparison",
    "StrategyComparisonResult": "strategy_comparison",
    "StrategyComparisonRunner": "strategy_comparison",
    "WalkForwardComparisonReport": "walk_forward_evaluation",
    "WalkForwardFold": "walk_forward",
    "WalkForwardFoldReport": "walk_forward_evaluation",
    "WalkForwardStrategyResult": "walk_forward_evaluation",
    "WalkForwardStrategyRunner": "walk_forward_evaluation",
    "TransactionCosts": "transaction_costs",
    "WalkForward": "walk_forward",
}

__all__ = [
    "BacktestEngine",
    "BacktestStep",
    "BacktestReport",
    "EventStressTest",
    "LeakageReport",
    "evaluate_backtest_report",
    "StrategyBacktestCase",
    "StrategyComparisonReport",
    "StrategyComparisonResult",
    "StrategyComparisonRunner",
    "WalkForwardComparisonReport",
    "WalkForwardFold",
    "WalkForwardFoldReport",
    "WalkForwardStrategyResult",
    "WalkForwardStrategyRunner",
    "TransactionCosts",
    "WalkForward",
]


def __getattr__(name: str) -> Any:  # PEP 562 lazy attribute access
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = importlib.import_module(f"{__name__}.{module}")
    return getattr(mod, name)


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # eager imports for type checkers / IDEs only
    from helion_risk_world.backtesting.backtest_engine import (
        BacktestEngine,
        BacktestReport,
        BacktestStep,
    )
    from helion_risk_world.backtesting.event_stress_test import EventStressTest
    from helion_risk_world.backtesting.leakage_report import LeakageReport
    from helion_risk_world.backtesting.strategy_diagnostics import evaluate_backtest_report
    from helion_risk_world.backtesting.strategy_comparison import (
        StrategyBacktestCase,
        StrategyComparisonReport,
        StrategyComparisonResult,
        StrategyComparisonRunner,
    )
    from helion_risk_world.backtesting.transaction_costs import TransactionCosts
    from helion_risk_world.backtesting.walk_forward import WalkForward, WalkForwardFold
    from helion_risk_world.backtesting.walk_forward_evaluation import (
        WalkForwardComparisonReport,
        WalkForwardFoldReport,
        WalkForwardStrategyResult,
        WalkForwardStrategyRunner,
    )
