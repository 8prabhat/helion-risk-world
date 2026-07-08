"""Transaction-cost application for the backtest (SPEC.md §23, Day 7).

Charges the SAME conservative Indian cost model the Execution Reality Layer uses (DRY) — statutory
fees + spread + slippage on the traded notional — so backtest and live costs cannot drift. NO_TRADE
and zero-notional actions cost nothing. SRP: cost application only.
"""

from __future__ import annotations

from datetime import datetime

from helion_risk_world.config.execution_config import CostModelConfig
from helion_risk_world.execution.cost_model import ConservativeIndianCostModel
from helion_risk_world.execution.slippage_model import SlippageModel
from helion_risk_world.schemas.action_schema import ActionType, CandidateAction
from helion_risk_world.schemas.execution_schema import CandidateOrder, ExecutionState

_EPS = 1e-9
_EPOCH = datetime(1970, 1, 1)


class TransactionCosts:
    """Apply realistic Indian transaction costs in the backtest (shared with live; DRY)."""

    def __init__(self, cfg: CostModelConfig) -> None:
        self._cfg = cfg
        self._cost = ConservativeIndianCostModel(cfg)
        self._slip = SlippageModel(cfg)

    @property
    def config(self) -> CostModelConfig:
        return self._cfg

    def apply(
        self,
        action: CandidateAction,
        notional: float,
        market: ExecutionState | None = None,
        *,
        order: CandidateOrder | None = None,
    ) -> float:
        """Total cost (INR) for trading ``notional``. Zero for NO_TRADE / zero-notional."""
        if action.action_type is ActionType.NO_TRADE or abs(notional) <= _EPS:
            return 0.0
        candidate = order or CandidateOrder(
            symbol=market.symbol if market else "UNKNOWN",
            side="buy",
            qty=abs(notional),
            notional=abs(notional),
        )
        mkt = market or ExecutionState(symbol=candidate.symbol, ts=_EPOCH, available_at=_EPOCH)
        spread = self._cost.spread_cost(candidate, mkt)
        statutory = self._cost.statutory(candidate)
        slippage = self._slip.estimate(candidate, mkt)
        return spread + statutory + slippage
