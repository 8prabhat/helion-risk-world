"""Management loop — barrier-managed position lifecycle (SPEC.md §19).

Once a position is entered, the planner does NOT re-evaluate an entry every bar.
The position is managed to its triple-barrier exit.  This module decides whether
a bar-update should trigger an early exit (discretionary close before barrier) or
hold to the next bar.

A new ENTRY is evaluated only when the account is flat.  This keeps the managed
trade coherent with the single-H barrier label (SPEC.md §11.5, §19 cadence).

SRP: hold/exit signal only — action scoring lives in reward_scorer.py.
"""

from __future__ import annotations

import math

from helion_risk_world.schemas.portfolio_schema import PortfolioState, RiskProfile
from helion_risk_world.schemas.prediction_schema import ModelPrediction


def _is_unsafe_reading(value: float) -> bool:
    """True for NaN/Inf uncertainty readings (review finding H11) — a non-finite
    reading never satisfies a `> threshold` comparison, so treat it as maximally
    unsafe (trigger the exit) rather than silently passing through as "fine"."""
    return not math.isfinite(value)


class ManagementLoop:
    """Decide whether the current position should be held or exited early (SPEC.md §19).

    Exit conditions (any → signal exit):
    1. OOD score above the configured threshold.
    2. Epistemic uncertainty above threshold.
    3. Barrier probabilities flip: P(stop) > P(target) when long (or vice versa for short).
    4. Maximum holding duration reached (H bars from entry).

    NOTE (review finding H9): condition 2 is inert whenever
    ``prediction.epistemic_calibrated`` is False — ``ForecasterPredictor`` (the
    default, non-world-model predictor) always emits ``epistemic=0.0`` with no
    real ensemble behind it. Use ``model_kind='world_model'`` for this exit
    condition to be meaningful.
    """

    def __init__(
        self,
        ood_threshold: float = 0.9,
        epistemic_threshold: float = 0.5,
        max_hold_bars: int = 12,
    ) -> None:
        self._ood_thr = ood_threshold
        self._epistemic_thr = epistemic_threshold
        self._max_hold = max_hold_bars

    def should_exit_early(
        self,
        state: PortfolioState,
        prediction: ModelPrediction,
        bars_in_position: int,
    ) -> tuple[bool, str]:
        """Return (exit_signal, reason_code).

        Called only when the account is in a position.  If True, the planner
        will include EXIT as the dominant candidate (overriding HOLD).
        """
        if bars_in_position >= self._max_hold:
            return True, "max_hold_reached"

        if _is_unsafe_reading(prediction.ood_score) or prediction.ood_score > self._ood_thr:
            return True, "ood_above_threshold"

        # Epistemic uncertainty — use the first (shortest) horizon's value
        if _is_unsafe_reading(prediction.epistemic) or prediction.epistemic > self._epistemic_thr:
            return True, "epistemic_above_threshold"

        from helion_risk_world.schemas.portfolio_schema import PositionSide

        if (
            state.position is PositionSide.LONG
            and prediction.barrier_for_side("long").stop > prediction.barrier_for_side("long").target
        ):
            return True, "barrier_flip_against_long"
        if (
            state.position is PositionSide.SHORT
            and prediction.barrier_for_side("short").stop > prediction.barrier_for_side("short").target
        ):
            return True, "barrier_flip_against_short"

        return False, ""

    def is_new_entry_allowed(self, state: PortfolioState) -> bool:
        """A new entry is only evaluated when the account is flat."""
        from helion_risk_world.schemas.portfolio_schema import PositionSide
        return state.position is PositionSide.FLAT


__all__ = ["ManagementLoop"]
