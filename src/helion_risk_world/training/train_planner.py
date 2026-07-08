from __future__ import annotations

from dataclasses import asdict
from typing import Any

from helion_risk_world.backtesting.backtest_engine import BacktestEngine
from helion_risk_world.evaluation import no_trade_metrics, regime_analysis, risk_metrics


class PlannerEvaluator:
    """Stage 7: conservative MPC evaluation (not RL) (SPEC.md §20)."""

    def __init__(self, engine: BacktestEngine) -> None:
        self._engine = engine

    def evaluate(self, steps: Any, account: Any, risk: Any) -> dict[str, Any]:
        report = self._engine.run(steps, account, risk)
        actions = [d.final_action.action_type.value for d in report.decisions]
        realized_market_returns = [step.realized_return for step in steps]
        return {
            "report": report,
            "summary": report.summary(),
            "risk": risk_metrics.compute(
                step_returns=report.step_returns,
                equity=report.equity_curve,
                risk_shield_interventions=report.risk_shield_interventions,
            ),
            "no_trade": no_trade_metrics.compute(
                actions=actions,
                realized_returns=realized_market_returns,
            ),
            "regimes": regime_analysis.compute(
                regimes=[d.latent_regime for d in report.decisions],
                values=report.step_returns,
            ),
        }

    def run(self, cfg: Any) -> Any:
        return self.evaluate(cfg["steps"], cfg["account"], cfg["risk"])
