"""Backtest engine (SPEC.md §23, Appendix A, Day 7).

Drives the full decision loop over a sequence of point-in-time steps and produces an auditable,
cost-aware ``BacktestReport``. Each step the planner decides (three planes -> Risk Shield -> final
action), the engine charges the SAME transaction-cost model used live (DRY), settles the REALIZED
return, advances the paper account, and records the ``FinalDecision``.

The engine is model-agnostic by design (DIP): it consumes pre-built ``BacktestStep``s (prediction +
market + realized return), so the predictor — heuristic or trained ``HRWForecaster`` — is decoupled.
The report separates gross vs net PnL, no-trade quality, risk-shield interventions and regime-wise
performance (SPEC.md §23). SRP: orchestration + reporting only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

from helion_risk_world.backtesting.transaction_costs import TransactionCosts
from helion_risk_world.execution.cost_model import overnight_financing_cost
from helion_risk_world.execution.feasibility import assert_entry_feasible
from helion_risk_world.execution.instrument_specs import resolve_instrument_spec
from helion_risk_world.execution.order_builder import build_candidate_order
from helion_risk_world.evaluation import trading_metrics as tm
from helion_risk_world.planner.protocols import PlannerProtocol
from helion_risk_world.schemas.action_schema import ActionType, FinalDecision
from helion_risk_world.schemas.execution_schema import ExecutionState
from helion_risk_world.schemas.portfolio_schema import PortfolioState, PositionSide, RiskProfile
from helion_risk_world.schemas.prediction_schema import ModelPrediction
from helion_risk_world.worlds.portfolio_world import PortfolioWorld
from helion_risk_world.worlds.position_math import resolve_executable_position

_EPS = 1e-9


@dataclass(frozen=True)
class BacktestStep:
    """One point-in-time decision step. ``realized_return`` is the future label (post-bar)."""

    prediction: ModelPrediction
    market: ExecutionState
    realized_return: float
    execution_market: ExecutionState | None = None
    label_realized_at: datetime | None = None
    """Timestamp the realized_return label actually became known (review finding
    H8) — lets LeakageReport.run() verify label_realized_at > ts for real,
    instead of the check being silently skipped."""


@dataclass
class BacktestReport:
    """Aggregated, decomposed backtest result (SPEC.md §23)."""

    n_steps: int
    n_trades: int
    no_trade_fraction: float
    gross_pnl: float
    net_pnl: float
    total_cost: float
    final_capital: float
    total_return: float
    annualized_return: float
    sharpe: float
    sortino: float
    max_drawdown: float
    drawdown_duration: float
    calmar: float
    hit_rate: float
    profit_factor: float
    payoff_ratio: float
    expectancy: float
    turnover: float
    capital_utilization: float
    risk_shield_interventions: int
    regime_breakdown: dict[str, dict[str, float]]
    decisions: list[FinalDecision] = field(default_factory=list)
    step_returns: list[float] = field(default_factory=list)
    trade_pnls: list[float] = field(default_factory=list)
    trade_returns: list[float] = field(default_factory=list)
    market_returns: list[float] = field(default_factory=list)
    step_turnover: list[float] = field(default_factory=list)
    step_cost_returns: list[float] = field(default_factory=list)
    step_exposure: list[float] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)
    ending_state: PortfolioState | None = None

    def summary(self) -> dict[str, float]:
        """Compact numeric summary (no per-decision detail)."""
        return {
            "n_steps": self.n_steps,
            "n_trades": self.n_trades,
            "no_trade_fraction": round(self.no_trade_fraction, 4),
            "gross_pnl": round(self.gross_pnl, 2),
            "net_pnl": round(self.net_pnl, 2),
            "total_cost": round(self.total_cost, 2),
            "total_return": round(self.total_return, 5),
            "annualized_return": round(self.annualized_return, 5),
            "sharpe": round(self.sharpe, 3),
            "sortino": round(self.sortino, 3) if self.sortino != float("inf") else float("inf"),
            "max_drawdown": round(self.max_drawdown, 4),
            "drawdown_duration": round(self.drawdown_duration, 2),
            "calmar": round(self.calmar, 3) if self.calmar != float("inf") else float("inf"),
            "hit_rate": round(self.hit_rate, 3),
            "profit_factor": round(self.profit_factor, 3),
            "payoff_ratio": round(self.payoff_ratio, 3),
            "expectancy": round(self.expectancy, 6),
            "turnover": round(self.turnover, 4),
            "capital_utilization": round(self.capital_utilization, 4),
            "risk_shield_interventions": self.risk_shield_interventions,
        }


class BacktestEngine:
    """Walk-forward-ready, leakage-checked backtest loop (SPEC.md §23)."""

    def __init__(self, planner: PlannerProtocol, costs: TransactionCosts) -> None:
        self._planner = planner
        self._costs = costs

    def run(
        self, steps: Sequence[BacktestStep], account: PortfolioState, risk: RiskProfile
    ) -> BacktestReport:
        """Run the decision loop over ``steps`` starting from ``account``; deterministic."""
        effective_risk = self._planner.adapt_risk(risk)
        assert_entry_feasible(
            [step.execution_market or step.market for step in steps],
            capital=account.capital,
            max_exposure=effective_risk.max_exposure,
            cost_cfg=self._costs.config,
        )
        state = account
        equity = [state.capital]
        step_returns: list[float] = []
        trade_pnls: list[float] = []
        trade_returns: list[float] = []
        turnover_fractions: list[float] = []
        cost_return_fractions: list[float] = []
        market_returns: list[float] = []
        exposures: list[float] = []
        decisions: list[FinalDecision] = []
        gross_total = net_total = cost_total = 0.0
        interventions = 0
        regime: dict[str, dict[str, float]] = {}
        timestamps = [state.ts]
        no_trade_steps = 0
        trade_open = False
        open_trade_pnl = 0.0
        open_trade_growth = 1.0
        last_session_date = state.ts.date()

        for step in steps:
            decision = self._planner.plan(step.prediction, state, risk, step.market)
            final = decision.final_action
            execution_market = step.execution_market or step.market
            resolved = resolve_executable_position(
                state,
                final,
                effective_risk.max_exposure,
                market=execution_market,
                execution_cfg=self._costs.config,
            )
            order = build_candidate_order(
                final,
                state,
                execution_market,
                max_exposure=effective_risk.max_exposure,
                cost_cfg=self._costs.config,
            )
            cost = self._costs.apply(
                final,
                resolved.traded_notional,
                execution_market,
                order=order,
            )
            # Overnight NRML-carry financing (feature/label overhaul Phase 4a): the
            # ~192-bar management horizon routinely holds a position across a session
            # close, which this cost model previously never charged for (only
            # per-trade statutory/spread/slippage). If a position was ALREADY open
            # entering this bar and this bar's session date has advanced, one night's
            # financing accrued on the held notional since the last bar.
            # PortfolioState.entry_price is never populated anywhere in this codebase
            # (schema-only field), so notional is derived from margin_used / the
            # instrument's margin_fraction (the same lookup position_math.py already
            # uses) rather than qty * entry_price.
            held_open = state.position is not PositionSide.FLAT and state.exposure > _EPS
            session_date = execution_market.ts.date()
            if held_open and session_date != last_session_date:
                spec = resolve_instrument_spec(execution_market.symbol, self._costs.config)
                margin_fraction = spec.margin_fraction if spec is not None else 1.0
                held_notional = state.margin_used / max(margin_fraction, _EPS)
                cost += overnight_financing_cost(
                    self._costs.config, notional=held_notional, nights_held=1
                )
            last_session_date = session_date
            prev_capital = state.capital
            prev_state = state
            state = PortfolioWorld.apply_fill(
                state,
                final,
                step.realized_return,
                cost,
                effective_risk.max_exposure,
                market=execution_market,
            )
            net_step = state.capital - prev_capital
            gross_step = net_step + cost
            step_return = net_step / max(prev_capital, _EPS)
            decision = decision.model_copy(update={"paper_result": step_return})
            old_open = prev_state.position is not PositionSide.FLAT and prev_state.exposure > _EPS
            new_open = state.position is not PositionSide.FLAT and state.exposure > _EPS
            side_flip = old_open and new_open and prev_state.position is not state.position
            if side_flip and trade_open:
                trade_pnls.append(open_trade_pnl)
                trade_returns.append(open_trade_growth - 1.0)
                trade_open = False
                open_trade_pnl = 0.0
                open_trade_growth = 1.0
                old_open = False
            if old_open or new_open:
                if not trade_open:
                    trade_open = True
                    open_trade_pnl = 0.0
                    open_trade_growth = 1.0
                open_trade_pnl += net_step
                open_trade_growth *= 1.0 + step_return
            if trade_open and old_open and not new_open:
                trade_pnls.append(open_trade_pnl)
                trade_returns.append(open_trade_growth - 1.0)
                trade_open = False
                open_trade_pnl = 0.0
                open_trade_growth = 1.0

            equity.append(state.capital)
            timestamps.append(execution_market.ts)
            step_returns.append(step_return)
            turnover_fractions.append(resolved.traded_margin / max(prev_capital, _EPS))
            cost_return_fractions.append(cost / max(prev_capital, _EPS))
            market_returns.append(step.realized_return)
            exposures.append(state.exposure)
            gross_total += gross_step
            net_total += net_step
            cost_total += cost
            if not decision.risk_shield.allowed:
                interventions += 1
            if final.action_type is ActionType.NO_TRADE:
                no_trade_steps += 1
            bucket = regime.setdefault(decision.latent_regime, {"n": 0.0, "net_pnl": 0.0})
            bucket["n"] += 1
            bucket["net_pnl"] += net_step
            decisions.append(decision)

        if trade_open:
            trade_pnls.append(open_trade_pnl)
            trade_returns.append(open_trade_growth - 1.0)

        n = len(steps)
        annualized_return = tm.annualized_return(equity, timestamps)
        max_drawdown = tm.max_drawdown(equity)
        return BacktestReport(
            n_steps=n,
            n_trades=len(trade_pnls),
            no_trade_fraction=(no_trade_steps / n) if n else 0.0,
            gross_pnl=gross_total,
            net_pnl=net_total,
            total_cost=cost_total,
            final_capital=state.capital,
            total_return=(state.capital / max(account.capital, _EPS)) - 1.0,
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
            turnover=tm.turnover(turnover_fractions),
            capital_utilization=float(sum(exposures) / len(exposures)) if exposures else 0.0,
            risk_shield_interventions=interventions,
            regime_breakdown=regime,
            decisions=decisions,
            step_returns=step_returns,
            trade_pnls=trade_pnls,
            trade_returns=trade_returns,
            market_returns=market_returns,
            step_turnover=turnover_fractions,
            step_cost_returns=cost_return_fractions,
            step_exposure=exposures,
            equity_curve=equity,
            timestamps=timestamps,
            ending_state=state,
        )
