"""Portfolio World — action-conditioned account simulator (SPEC.md §17, Appendix A).

``step`` answers: *if this account takes this action under the predicted market distribution,
what happens to capital / margin / exposure / drawdown?*

ΔW is computed ANALYTICALLY from the decode heads — there is no price-path simulation:
  - P(stop) × explicit stop return (or legacy ``−d · sigma_H`` fallback)
  - P(target) × explicit target return (or legacy ``+u · sigma_H`` fallback)
  - P(timeout) × timeout_returns   timeout returns from the return-quantile head
These are mixed into a ΔW outcome distribution; CVaR is the positive shortfall.

CVaR sign convention (SPEC.md §17, §19): ``cvar_dW`` = positive shortfall ≥ 0.
Sampling is deterministic by default and supports common-random-number candidate comparisons.

CONSUMES portfolio variables; NEVER exposes them to market encoders (SPEC.md §6).
SRP: portfolio transition + consequence summary.
"""

from __future__ import annotations

import numpy as np

from helion_risk_world.schemas.action_schema import ActionType, CandidateAction
from helion_risk_world.schemas.execution_schema import ExecutionState
from helion_risk_world.schemas.portfolio_schema import (
    Consequence,
    PortfolioState,
    PositionSide,
    RiskProfile,
)
from helion_risk_world.schemas.prediction_schema import ModelPrediction
from helion_risk_world.worlds.position_math import resolve_executable_position

_EPS = 1e-9


def _build_dW_distribution(
    prediction: ModelPrediction,
    signed_notional: float,
    cost: float,
    u: float,
    d: float,
    n_samples: int,
    event_uniforms: np.ndarray,
    return_uniforms: np.ndarray,
) -> np.ndarray:
    """Build a ΔW outcome array from head-implied distribution (analytic, no path sim).

    Outcomes:
      stop    → explicit stop return (or legacy fallback)
      target  → explicit target return (or legacy fallback)
      timeout → sampled from return-quantile distribution at H=12
    All converted to fraction-of-capital ΔW.
    """
    side = "short" if signed_notional < 0.0 else "long"
    p = prediction.barrier_for_side(side)
    stop_ret = prediction.resolved_stop_return_for_side(side, fallback_mult=d)
    tgt_ret = prediction.resolved_target_return_for_side(side, fallback_mult=u)

    p_stop = float(p.stop)
    p_tgt = float(p.target)
    stop_cut = p_stop
    target_cut = p_stop + p_tgt

    # Timeout returns: sample from piecewise-linear inverse-CDF of return-quantile head at H.
    hp = prediction.longest_horizon
    items = sorted(hp.return_quantiles.items())
    levels = np.array([l for l, _ in items], dtype=float)
    values = np.array([v for _, v in items], dtype=float)
    timeout_rets = np.interp(return_uniforms, levels, values)

    raw_returns = np.where(
        event_uniforms < stop_cut,
        stop_ret,
        np.where(event_uniforms < target_cut, tgt_ret, timeout_rets),
    )
    return signed_notional * raw_returns - cost  # [N] ΔW values


class PortfolioWorld:
    """Analytic portfolio consequence simulator (SPEC.md §17)."""

    def __init__(
        self,
        cost_rate: float = 0.0,
        u: float = 2.0,
        d: float = 2.0,
        n_samples: int = 1000,
        cvar_alpha: float = 0.05,
        seed: int = 7,
    ) -> None:
        if cost_rate < 0:
            raise ValueError("cost_rate must be non-negative")
        self._cost_rate = cost_rate
        self._u = u
        self._d = d
        self._n_samples = n_samples
        self._cvar_alpha = cvar_alpha
        self._rng = np.random.default_rng(seed)

    def sample_noise(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate common random numbers for one decision step."""
        return (
            self._rng.uniform(0.0, 1.0, self._n_samples),
            self._rng.uniform(0.0, 1.0, self._n_samples),
        )

    def step(
        self,
        state: PortfolioState,
        action: CandidateAction,
        prediction: ModelPrediction,
        risk_profile: RiskProfile,
        *,
        market: ExecutionState | None = None,
        common_noise: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[PortfolioState, Consequence]:
        """Simulate the consequence of ``action`` under the head-implied ΔW distribution.
        """
        resolved = resolve_executable_position(
            state,
            action,
            risk_profile.max_exposure,
            market=market,
        )
        cost = self._cost_rate * resolved.traded_margin
        cap = max(state.capital, _EPS)
        event_uniforms, return_uniforms = common_noise or self.sample_noise()

        dW_vec = _build_dW_distribution(
            prediction=prediction,
            signed_notional=resolved.new_signed_notional,
            cost=cost,
            u=self._u,
            d=self._d,
            n_samples=self._n_samples,
            event_uniforms=event_uniforms,
            return_uniforms=return_uniforms,
        )

        exp_dW = float(dW_vec.mean()) / cap
        # CVaR as POSITIVE shortfall: expected worst-α loss, reported ≥ 0
        k = max(1, int(np.ceil(self._cvar_alpha * dW_vec.size)))
        cvar_dW = float(-np.sort(dW_vec)[:k].mean()) / cap
        cvar_dW = max(cvar_dW, 0.0)

        projected_dd = state.drawdown + np.maximum(-dW_vec / cap, 0.0)
        p_dd_breach = float((projected_dd > risk_profile.max_drawdown).mean())

        consequence = Consequence(
            exp_dW=exp_dW,
            cvar_dW=cvar_dW,
            p_drawdown_breach=p_dd_breach,
            d_margin=resolved.new_margin_used - state.margin_used,
            d_exposure=abs(resolved.new_signed_fraction) - state.exposure,
        )
        next_state = self._next_state(
            state,
            action,
            resolved.new_signed_fraction,
            resolved.new_signed_notional,
            resolved.new_margin_used,
            resolved.position_qty,
            exp_dW * cap,
            resolved.traded_margin,
        )
        return next_state, consequence

    @staticmethod
    def apply_fill(
        state: PortfolioState,
        action: CandidateAction,
        realized_return: float,
        cost: float,
        max_exposure: float,
        market: ExecutionState | None = None,
    ) -> PortfolioState:
        """Settle an executed action with the REALIZED return (backtest/paper fill)."""
        resolved = resolve_executable_position(
            state,
            action,
            max_exposure,
            market=market,
        )
        realized_pnl = resolved.new_signed_notional * realized_return - cost
        return PortfolioWorld._next_state(
            state,
            action,
            resolved.new_signed_fraction,
            resolved.new_signed_notional,
            resolved.new_margin_used,
            resolved.position_qty,
            realized_pnl,
            resolved.traded_margin,
        )

    @staticmethod
    def _next_state(
        state: PortfolioState,
        action: CandidateAction,
        new_frac: float,
        new_notional: float,
        new_margin_used: float,
        position_qty: float,
        exp_pnl: float,
        traded_budget: float,
    ) -> PortfolioState:
        new_capital = state.capital + exp_pnl
        traded_frac = traded_budget / max(state.capital, _EPS)
        if new_frac > _EPS:
            side = PositionSide.LONG
        elif new_frac < -_EPS:
            side = PositionSide.SHORT
        else:
            side = PositionSide.FLAT
        margin_used = max(0.0, new_margin_used)
        traded = traded_budget > _EPS
        realized = state.realized_pnl + (exp_pnl if action.action_type == ActionType.EXIT else 0.0)
        unrealized = 0.0 if side is PositionSide.FLAT else exp_pnl
        drawdown = max(0.0, (state.capital0 - new_capital) / max(state.capital0, _EPS))
        return state.model_copy(
            update={
                "capital": new_capital,
                "cash": new_capital - margin_used,
                "position": side,
                "position_qty": position_qty,
                "exposure": abs(new_frac),
                "margin_used": margin_used,
                "free_margin": max(0.0, new_capital - margin_used),
                "realized_pnl": realized,
                "unrealized_pnl": unrealized,
                "daily_pnl": state.daily_pnl + exp_pnl,
                "drawdown": drawdown,
                "trades_today": state.trades_today + (1 if traded else 0),
                "consecutive_losses": (state.consecutive_losses + 1) if exp_pnl < 0 else 0,
                "risk_budget_used": state.risk_budget_used + traded_frac,
            }
        )
