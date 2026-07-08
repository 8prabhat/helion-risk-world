"""Candidate action enumeration (SPEC.md §18, §21, Day 6).

NO_TRADE is ALWAYS the first candidate — it is a first-class action, not the absence of one. The
admissible set depends on the current position: a flat account can enter long/short; an open
position can exit/reduce/increase. Sizes are capped by ``max_size`` (the uncertainty-gated cap from
the PositionSizer). SRP: enumeration only — no scoring, no risk checks.
"""

from __future__ import annotations

from collections.abc import Sequence

from helion_risk_world.schemas.action_schema import ActionType, CandidateAction
from helion_risk_world.schemas.portfolio_schema import PortfolioState, PositionSide, RiskProfile


class ActionSampler:
    """Enumerate admissible candidate actions. NO_TRADE is ALWAYS included (SPEC.md §18, §21)."""

    def __init__(self, sizes: Sequence[float]) -> None:
        if 0.0 not in tuple(sizes):
            raise ValueError("size grid must include 0.0 (NO_TRADE)")
        self._sizes = tuple(sorted(set(sizes)))

    def enumerate(
        self, state: PortfolioState, risk: RiskProfile, max_size: float = 1.0
    ) -> list[CandidateAction]:
        """Admissible candidates. NO_TRADE first; positive sizes capped by ``max_size``."""
        candidates = [CandidateAction(action_type=ActionType.NO_TRADE, size_fraction=0.0)]
        positive = [s for s in self._sizes if 0.0 < s <= max_size + 1e-9]

        if state.position is PositionSide.FLAT:
            for s in positive:
                candidates.append(
                    CandidateAction(action_type=ActionType.ENTER_LONG, size_fraction=s)
                )
                candidates.append(
                    CandidateAction(action_type=ActionType.ENTER_SHORT, size_fraction=s)
                )
        else:
            # V1: single-position, no pyramiding (SPEC.md §19).
            # INCREASE and fresh ENTER_* are NOT available mid-position.
            candidates.append(CandidateAction(action_type=ActionType.EXIT, size_fraction=0.0))
            for s in positive:
                candidates.append(CandidateAction(action_type=ActionType.REDUCE, size_fraction=s))
        return candidates
