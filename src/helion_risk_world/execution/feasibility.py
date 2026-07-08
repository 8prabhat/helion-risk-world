"""Contract-aware account feasibility checks for discrete futures execution."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Sequence

from helion_risk_world.config.execution_config import CostModelConfig
from helion_risk_world.execution.order_builder import build_candidate_order
from helion_risk_world.schemas.action_schema import ActionType, CandidateAction
from helion_risk_world.schemas.execution_schema import ExecutionState
from helion_risk_world.schemas.portfolio_schema import PortfolioState
from helion_risk_world.worlds.position_math import minimum_contract_margin_fraction


@dataclass(frozen=True)
class EntryFeasibilityReport:
    symbol: str | None
    feasible_any: bool
    checked_discrete_contract: bool
    required_exposure_min: float | None = None
    required_exposure_median: float | None = None
    required_exposure_max: float | None = None


def analyze_entry_feasibility(
    markets: Sequence[ExecutionState],
    *,
    capital: float,
    max_exposure: float,
    cost_cfg: CostModelConfig,
) -> EntryFeasibilityReport:
    if not markets:
        return EntryFeasibilityReport(
            symbol=None,
            feasible_any=True,
            checked_discrete_contract=False,
        )

    account = PortfolioState(
        ts=markets[0].ts,
        capital0=capital,
        capital=capital,
        cash=capital,
        free_margin=capital,
    )
    probe = CandidateAction(action_type=ActionType.ENTER_LONG, size_fraction=1.0)
    required_fractions: list[float] = []
    feasible_any = False
    for market in markets:
        required = minimum_contract_margin_fraction(capital, market, cost_cfg)
        if required is not None:
            required_fractions.append(required)
        if build_candidate_order(
            probe,
            account,
            market,
            max_exposure=max_exposure,
            cost_cfg=cost_cfg,
        ) is not None:
            feasible_any = True
            break

    if not required_fractions:
        return EntryFeasibilityReport(
            symbol=markets[0].symbol,
            feasible_any=True,
            checked_discrete_contract=False,
        )
    return EntryFeasibilityReport(
        symbol=markets[0].symbol,
        feasible_any=feasible_any,
        checked_discrete_contract=True,
        required_exposure_min=min(required_fractions),
        required_exposure_median=median(required_fractions),
        required_exposure_max=max(required_fractions),
    )


def assert_entry_feasible(
    markets: Sequence[ExecutionState],
    *,
    capital: float,
    max_exposure: float,
    cost_cfg: CostModelConfig,
) -> EntryFeasibilityReport:
    report = analyze_entry_feasibility(
        markets,
        capital=capital,
        max_exposure=max_exposure,
        cost_cfg=cost_cfg,
    )
    if report.feasible_any or not report.checked_discrete_contract:
        return report
    assert report.symbol is not None
    assert report.required_exposure_min is not None
    assert report.required_exposure_median is not None
    assert report.required_exposure_max is not None
    raise ValueError(
        "account/risk profile cannot open one contract on any tested market step; "
        f"symbol={report.symbol}, capital={capital:.2f}, max_exposure={max_exposure:.3f}, "
        f"required_exposure_min={report.required_exposure_min:.3f}, "
        f"required_exposure_median={report.required_exposure_median:.3f}, "
        f"required_exposure_max={report.required_exposure_max:.3f}. "
        "Increase account capital or max_exposure, or choose a smaller contract."
    )


__all__ = [
    "EntryFeasibilityReport",
    "analyze_entry_feasibility",
    "assert_entry_feasible",
]
