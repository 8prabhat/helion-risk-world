"""Execution Reality Layer (SPEC.md §15, Day 5).

Composes the cost / slippage / liquidity / latency / fill models into a single ``CostEstimate`` with
an execution **realism** score (high/medium/low):

  * microstructure realism (always available): driven by fill probability + liquidity;
  * edge-aware realism (when ``expected_edge`` is supplied by the planner): total cost as a fraction
    of the expected edge, compared against the config bands — this is the "does the edge survive
    costs?" test from SPEC §15. The edge↔cost comparison itself also lives in the planner/Risk
    Shield (``expected_return < total_cost -> no trade``); here it only colours the realism label.

DIP: depends on ``CostModelProtocol``, never a concrete broker. SRP: orchestration only.
"""

from __future__ import annotations

from helion_risk_world.config.execution_config import CostModelConfig
from helion_risk_world.execution.cost_model import ConservativeIndianCostModel, CostModelProtocol
from helion_risk_world.execution.fill_simulator import FillSimulator
from helion_risk_world.execution.latency_model import LatencyModel
from helion_risk_world.execution.liquidity_model import LiquidityModel
from helion_risk_world.execution.slippage_model import SlippageModel
from helion_risk_world.schemas.execution_schema import (
    CandidateOrder,
    CostEstimate,
    ExecutionRealism,
    ExecutionState,
)


class ExecutionReality:
    """Execution Reality Layer: 'can this trade be executed profitably?' (SPEC.md §15)."""

    def __init__(self, cfg: CostModelConfig, cost_model: CostModelProtocol | None = None) -> None:
        self._cfg = cfg
        self._cost = cost_model or ConservativeIndianCostModel(cfg)
        self._slip = SlippageModel(cfg)
        self._liq = LiquidityModel()
        self._lat = LatencyModel()
        self._fill = FillSimulator(cfg)

    @property
    def config(self) -> CostModelConfig:
        return self._cfg

    def estimate(
        self,
        order: CandidateOrder,
        market: ExecutionState,
        *,
        expected_edge: float | None = None,
    ) -> CostEstimate:
        """Estimate cost + fill + realism for one candidate order (SPEC.md Appendix A)."""
        spread_cost = self._cost.spread_cost(order, market)
        statutory = self._cost.statutory(order)
        slippage = self._slip.estimate(order, market)
        total_cost = spread_cost + statutory + slippage

        liquidity = self._liq.condition(order, market)
        probs = self._fill.probabilities(order, market, liquidity)
        latency = self._lat.impact(order, market)

        realism = self._score_realism(total_cost, probs.fill, liquidity, expected_edge)
        return CostEstimate(
            spread_cost=spread_cost,
            statutory_fees=statutory,
            slippage=slippage,
            total_cost=total_cost,
            fill_prob=probs.fill,
            partial_fill_prob=probs.partial,
            reject_prob=probs.reject,
            latency_ms=latency,
            realism=realism,
        )

    def _score_realism(
        self, total_cost: float, fill_prob: float, liquidity: float, expected_edge: float | None
    ) -> ExecutionRealism:
        cfg = self._cfg
        # Hard microstructure veto: an unreliable fill is never "high realism".
        if fill_prob < 0.6 or liquidity < 0.3:
            return ExecutionRealism.LOW
        if expected_edge is not None and expected_edge > 0:
            burden = total_cost / expected_edge
            if burden >= cfg.realism_low_cost_frac:
                return ExecutionRealism.LOW
            if burden <= cfg.realism_high_cost_frac and fill_prob >= 0.9:
                return ExecutionRealism.HIGH
            return ExecutionRealism.MEDIUM
        # No edge supplied -> microstructure-only judgement.
        if fill_prob >= 0.9 and liquidity >= 0.7:
            return ExecutionRealism.HIGH
        return ExecutionRealism.MEDIUM
