"""Conservative MPC planner — mean–CVaR objective (SPEC.md §19, Appendix A).

For each admissible candidate (incl. NO_TRADE): simulate the Portfolio World consequence
analytically from the heads, estimate Execution Reality cost, and score:
  U(a) = E[ΔW] − λ · CVaR_α[ΔW] − Cost(a)

Pick the argmax, then hand to the deterministic Risk Shield which may override.

Management loop: when IN a position the planner checks whether early exit is warranted
(via ManagementLoop); otherwise the position is held to its triple-barrier exit.  A new
ENTRY is evaluated only when flat.

DIP: depends on Protocol abstractions, never concrete brokers or cost implementations.
SRP: rank candidates + delegate to Risk Shield.
"""

from __future__ import annotations

from helion_risk_world.config.execution_config import CostModelConfig
from helion_risk_world.config.planner_config import PlannerConfig
from helion_risk_world.config.risk_config import RiskShieldConfig
from helion_risk_world.execution.execution_reality import ExecutionReality
from helion_risk_world.execution.order_builder import build_candidate_order
from helion_risk_world.planner.action_auditor import ActionAuditor
from helion_risk_world.planner.action_sampler import ActionSampler
from helion_risk_world.planner.management_loop import ManagementLoop
from helion_risk_world.planner.position_sizer import PositionSizer
from helion_risk_world.planner.reward_scorer import RewardScorer
from helion_risk_world.risk.risk_shield import RiskShield
from helion_risk_world.schemas.action_schema import (
    ActionType,
    CandidateAction,
    FinalDecision,
    ScoredCandidate,
)
from helion_risk_world.schemas.execution_schema import (
    CandidateOrder,
    CostEstimate,
    ExecutionRealism,
    ExecutionState,
)
from helion_risk_world.schemas.portfolio_schema import Consequence, PortfolioState, RiskProfile
from helion_risk_world.schemas.prediction_schema import ModelPrediction
from helion_risk_world.worlds.portfolio_world import PortfolioWorld

_EPS = 1e-9

_ZERO_COST = CostEstimate(
    spread_cost=0.0, statutory_fees=0.0, slippage=0.0, total_cost=0.0,
    fill_prob=1.0, partial_fill_prob=0.0, reject_prob=0.0, latency_ms=0.0,
    realism=ExecutionRealism.HIGH,
)
_ZERO_CONSEQUENCE = Consequence(
    exp_dW=0.0, cvar_dW=0.0, p_drawdown_breach=0.0, d_margin=0.0, d_exposure=0.0,
)


class MPCPlanner:
    """Mean–CVaR MPC planner with barrier-managed position cadence (SPEC.md §19)."""

    def __init__(
        self,
        sampler: ActionSampler,
        portfolio_world: PortfolioWorld,
        execution_reality: ExecutionReality,
        scorer: RewardScorer,
        auditor: ActionAuditor,
        risk_shield: RiskShield,
        management_loop: ManagementLoop,
        position_sizer: PositionSizer | None = None,
        confidence_scale: float = 1.0,
        strategy_name: str | None = None,
        decision_horizon_bars: int | None = None,
    ) -> None:
        self._sampler = sampler
        self._portfolio_world = portfolio_world
        self._execution_reality = execution_reality
        self._scorer = scorer
        self._auditor = auditor
        self._risk_shield = risk_shield
        self._management = management_loop
        self._sizer = position_sizer or PositionSizer()
        self._confidence_scale = confidence_scale
        self._strategy_name = strategy_name
        self._decision_horizon_bars = decision_horizon_bars
        self._bars_in_position: dict[str, int] = {}  # symbol -> bar count

    @classmethod
    def default(
        cls,
        *,
        planner_cfg: PlannerConfig | None = None,
        risk_cfg: RiskShieldConfig | None = None,
        cost_cfg: CostModelConfig | None = None,
        confidence_scale: float = 1.0,
        management_loop: ManagementLoop | None = None,
        strategy_name: str | None = None,
        decision_horizon_bars: int | None = None,
    ) -> MPCPlanner:
        planner_cfg = planner_cfg or PlannerConfig()
        risk_cfg = risk_cfg or RiskShieldConfig()
        return cls(
            sampler=ActionSampler(planner_cfg.sizes),
            portfolio_world=PortfolioWorld(
                cost_rate=0.0,
                n_samples=planner_cfg.n_outcome_samples,
                cvar_alpha=planner_cfg.cvar_alpha,
            ),
            execution_reality=ExecutionReality(cost_cfg or CostModelConfig()),
            scorer=RewardScorer(planner_cfg, risk_cfg),
            auditor=ActionAuditor(),
            risk_shield=RiskShield(risk_cfg),
            management_loop=management_loop or ManagementLoop(),
            position_sizer=PositionSizer(),
            confidence_scale=confidence_scale,
            strategy_name=strategy_name,
            decision_horizon_bars=decision_horizon_bars,
        )

    def _decision_prediction(self, prediction: ModelPrediction) -> ModelPrediction:
        """Project the model output onto the strategy-specific decision horizon."""
        if self._decision_horizon_bars is None:
            return prediction
        try:
            horizon = prediction.horizon(self._decision_horizon_bars)
        except KeyError:
            return prediction
        return prediction.model_copy(
            update={
                "horizon_preds": [horizon],
                "sigma_H": horizon.volatility,
            }
        )

    @staticmethod
    def adapt_risk(risk: RiskProfile) -> RiskProfile:
        """Base planner leaves account risk unchanged."""
        return risk

    def plan(
        self,
        prediction: ModelPrediction,
        state: PortfolioState,
        risk: RiskProfile,
        market: ExecutionState | None = None,
    ) -> FinalDecision:
        prediction = self._decision_prediction(prediction)
        market = market or ExecutionState(
            symbol=prediction.symbol, ts=prediction.ts, available_at=prediction.ts
        )
        key = prediction.symbol
        # Track bars-in-position for the management loop
        from helion_risk_world.schemas.portfolio_schema import PositionSide
        if state.position is PositionSide.FLAT:
            self._bars_in_position[key] = 0
        else:
            self._bars_in_position[key] = self._bars_in_position.get(key, 0) + 1

        bars_in = self._bars_in_position.get(key, 0)
        exit_early, _ = self._management.should_exit_early(state, prediction, bars_in)

        # If in position and no early exit signal: HOLD without re-scoring
        if state.position is not PositionSide.FLAT and not exit_early:
            hold = CandidateAction(action_type=ActionType.NO_TRADE, size_fraction=0.0)
            held_sc = ScoredCandidate(action=hold, score=0.0, components={"hold": 0.0})
            risk_decision = self._risk_shield.validate(hold, state, risk, prediction)
            final_hold = hold if risk_decision.allowed else risk_decision.final_action
            return self._auditor.record(
                risk_decision, [held_sc],
                prediction=prediction, state=state, chosen=held_sc,
                final_action=final_hold, cost=_ZERO_COST, consequence=_ZERO_CONSEQUENCE,
                strategy_name=self._strategy_name,
            )
        if state.position is not PositionSide.FLAT and exit_early:
            exit_action = CandidateAction(action_type=ActionType.EXIT, size_fraction=0.0)
            exit_sc = ScoredCandidate(action=exit_action, score=0.0, components={"exit_early": 1.0})
            risk_decision = self._risk_shield.validate(exit_action, state, risk, prediction)
            final_exit = exit_action if risk_decision.allowed else risk_decision.final_action
            self._bars_in_position[key] = 0
            return self._auditor.record(
                risk_decision,
                [exit_sc],
                prediction=prediction,
                state=state,
                chosen=exit_sc,
                final_action=final_exit,
                cost=_ZERO_COST,
                consequence=_ZERO_CONSEQUENCE,
                strategy_name=self._strategy_name,
            )

        max_size = self._sizer.adjust(1.0, prediction, self._confidence_scale)
        candidates = self._sampler.enumerate(state, risk, max_size=max_size)
        common_noise = self._portfolio_world.sample_noise()

        evaluated: list[ScoredCandidate] = []
        by_action: dict[CandidateAction, tuple[CostEstimate, Consequence]] = {}

        for action in candidates:
            _, consequence = self._portfolio_world.step(
                state, action, prediction, risk, market=market, common_noise=common_noise
            )
            order = self._to_order(action, state, market, risk.max_exposure)
            if action.action_type is not ActionType.NO_TRADE and order is None:
                cost = _ZERO_COST
                score = -1e9
                components = {
                    "exp_dW": 0.0,
                    "cvar_term": 0.0,
                    "cost_term": 0.0,
                    "utility": score,
                    "unexecutable": 1.0,
                }
            elif order is not None:
                cost = self._execution_reality.estimate(
                    order, market, expected_edge=consequence.exp_dW * state.capital
                )
                score, components = self._scorer.score(
                    action, prediction, consequence, cost, risk, state
                )
            else:
                cost = _ZERO_COST
                score, components = self._scorer.score(
                    action, prediction, consequence, cost, risk, state
                )
            sc = ScoredCandidate(action=action, score=score, components=components)
            evaluated.append(sc)
            by_action[action] = (cost, consequence)

        chosen = max(evaluated, key=self._candidate_rank)
        chosen_cost, chosen_cons = by_action[chosen.action]
        risk_decision = self._risk_shield.validate(chosen.action, state, risk, prediction)
        final_action = chosen.action if risk_decision.allowed else risk_decision.final_action

        if final_action.action_type is ActionType.EXIT:
            self._bars_in_position[key] = 0

        return self._auditor.record(
            risk_decision, evaluated,
            prediction=prediction, state=state, chosen=chosen,
            final_action=final_action, cost=chosen_cost, consequence=chosen_cons,
            strategy_name=self._strategy_name,
        )

    @staticmethod
    def _candidate_rank(candidate: ScoredCandidate) -> tuple[float, float, float, float]:
        """Prefer the highest score, then executable candidates, then the best raw utility."""
        components = candidate.components
        return (
            candidate.score,
            -float(components.get("unexecutable", 0.0)),
            float(components.get("raw_utility", components.get("utility", candidate.score))),
            float(components.get("exp_dW", 0.0)),
        )

    def _to_order(
        self,
        action: CandidateAction,
        state: PortfolioState,
        market: ExecutionState,
        max_exposure: float,
    ) -> CandidateOrder | None:
        return build_candidate_order(
            action,
            state,
            market,
            max_exposure=max_exposure,
            cost_cfg=self._execution_reality.config,
        )

    def set_confidence_scale(self, confidence_scale: float) -> None:
        self._confidence_scale = float(max(0.0, min(1.0, confidence_scale)))
