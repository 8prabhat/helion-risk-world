"""Prediction schemas — Market World output. Distributions, not point estimates (SPEC.md §15-16).

Key design:
- No ``direction_probs`` — direction is dropped as a primary target (SPEC.md §3, §15).
- Per-horizon return/vol heads for h ∈ {3, 6, 12} (term structure for sizing/uncertainty).
- Single-H barrier and MAE heads at H=12 (the actual managed-trade horizon, SPEC.md §11.5).
- ``epistemic/aleatoric/ood_score`` derived from the RSSM's own distributions (§16), not heads.
- ``sigma_H`` is the vol-head output at H=12; explicit stop/target widths are carried separately
  when available so downstream risk and execution logic uses decision-time barrier geometry.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from helion_risk_world.schemas.market_schema import Regime

QUANTILE_LEVELS: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)


class BarrierProbabilities(BaseModel):
    """Predicted probabilities for the three barrier outcomes at H=12 (SPEC.md §15)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stop: float = Field(ge=0.0, le=1.0, description="P(stop hit first).")
    target: float = Field(ge=0.0, le=1.0, description="P(target hit first).")
    timeout: float = Field(ge=0.0, le=1.0, description="P(no barrier hit within H).")

    @model_validator(mode="after")
    def _sums_to_one(self) -> BarrierProbabilities:
        total = self.stop + self.target + self.timeout
        if abs(total - 1.0) > 1e-3:
            raise ValueError(f"barrier probabilities sum to {total:.4f}, expected ~1.0")
        return self


class HorizonPrediction(BaseModel):
    """Return-quantile + volatility prediction for one horizon h (SPEC.md §15, §11.5)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    horizon_bars: int = Field(gt=0)
    return_quantiles: dict[float, float] = Field(
        description=(
            "tau -> predicted quantile of futures return over [t, t+h]. "
            "Round-trips correctly through pydantic v2 in Python, but JSON object "
            "keys are always strings, so model_dump_json()/.json() emits tau as a "
            "string (e.g. \"0.5\"); non-Python consumers must parse it back to float "
            "themselves (review finding, docs/review_2026-07-01.md)."
        )
    )
    volatility: float = Field(ge=0.0, description="Predicted realised vol over [t, t+h].")

    @model_validator(mode="after")
    def _quantile_monotone(self) -> HorizonPrediction:
        if not self.return_quantiles:
            raise ValueError("return_quantiles must be non-empty")
        qs = [self.return_quantiles[k] for k in sorted(self.return_quantiles)]
        if any(b < a for a, b in zip(qs, qs[1:], strict=False)):
            raise ValueError("return quantiles must be non-decreasing across levels")
        return self


class ModelPrediction(BaseModel):
    """Full Market World output: per-horizon term structure + barrier/excursions + uncertainty/OOD.

    SPEC.md §15 layout:
      horizon_preds  — return-quantile + vol per h ∈ {3,6,12}  (term structure)
      barrier        — P(stop/target/timeout) at single H=12   (managed-trade life)
      mae            — expected max adverse excursion at H=12
      mfe            — expected max favourable excursion at H=12
      sigma_H        — vol head output at H=12
      stop_return    — optional explicit stop-barrier return from entry
      target_return  — optional explicit target-barrier return from entry
      regime_probs   — optional regime probability vector (state-derived, §11.4)
      epistemic      — RSSM prior-rollout spread across ensemble members (§16)
      aleatoric      — per-horizon quantile width / prior σ_p (§16)
      ood_score      — −log p_θ(z_t | h_t), higher = more OOD (§16)
      epistemic_calibrated — False when ``epistemic`` is a hardcoded placeholder rather
                     than a real RSSM ensemble-spread estimate (review finding H9): the
                     plain ``ForecasterPredictor`` has no ensemble to derive spread from
                     and always emits ``epistemic=0.0``. Consumers gating on epistemic
                     uncertainty (ManagementLoop, EpistemicRiskBlock, PositionSizer)
                     should treat that gate as inert, not "confirmed safe", when this
                     is False.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    ts: datetime
    horizon_preds: list[HorizonPrediction] = Field(
        description="Per-horizon predictions sorted ascending by horizon_bars."
    )
    barrier: BarrierProbabilities
    mae: float = Field(ge=0.0, description="Predicted max adverse excursion at H=12.")
    mfe: float = Field(default=0.0, ge=0.0, description="Predicted max favourable excursion at H=12.")
    sigma_H: float = Field(ge=0.0, description="Predicted vol at H=12.")
    stop_return: float | None = Field(
        default=None,
        le=0.0,
        description="Explicit stop-barrier return from entry when known at decision time.",
    )
    target_return: float | None = Field(
        default=None,
        ge=0.0,
        description="Explicit target-barrier return from entry when known at decision time.",
    )
    regime_probs: dict[Regime, float] | None = Field(
        default=None,
        description=(
            "Regime -> probability. Like return_quantiles, the enum key serializes to "
            "its string value in JSON output; non-Python consumers must map it back "
            "to the Regime enum themselves."
        ),
    )
    epistemic: float = Field(ge=0.0, description="Epistemic uncertainty from RSSM rollout spread.")
    aleatoric: float = Field(ge=0.0, description="Aleatoric uncertainty from quantile width.")
    ood_score: float = Field(ge=0.0, description="OOD = −log p_θ(z_t|h_t); higher = more OOD.")
    epistemic_calibrated: bool = Field(
        default=True,
        description=(
            "False when epistemic is a hardcoded placeholder (ForecasterPredictor has no "
            "ensemble), not a real RSSM rollout-spread estimate. See class docstring."
        ),
    )

    @property
    def longest_horizon(self) -> HorizonPrediction:
        """Shortcut to the H=max prediction (barrier/MAE target horizon)."""
        return max(self.horizon_preds, key=lambda h: h.horizon_bars)

    def horizon(self, h: int) -> HorizonPrediction:
        """Return the prediction for a specific horizon_bars value."""
        for hp in self.horizon_preds:
            if hp.horizon_bars == h:
                return hp
        raise KeyError(f"no prediction for horizon_bars={h}")

    def resolved_stop_return(self, *, fallback_mult: float = 1.0) -> float:
        """Explicit stop return when present; otherwise a legacy sigma-based fallback."""
        if self.stop_return is not None:
            return float(self.stop_return)
        return -float(fallback_mult) * max(float(self.sigma_H), 1e-6)

    def resolved_target_return(self, *, fallback_mult: float = 1.0) -> float:
        """Explicit target return when present; otherwise a legacy sigma-based fallback."""
        if self.target_return is not None:
            return float(self.target_return)
        return float(fallback_mult) * max(float(self.sigma_H), 1e-6)

    @staticmethod
    def _normalize_side(side: str) -> str:
        normalized = str(side).strip().lower()
        if normalized not in {"long", "short"}:
            raise ValueError(f"unsupported side: {side!r}")
        return normalized

    def barrier_for_side(self, side: str) -> BarrierProbabilities:
        """Return barrier probabilities interpreted for one trade side.

        Stored barrier probabilities are long-style in underlying-return space:
          stop   -> lower barrier touched first
          target -> upper barrier touched first

        For short trades the adverse/favourable interpretation swaps, so the planner and
        portfolio world must see stop/target reversed.
        """
        if self._normalize_side(side) == "long":
            return self.barrier
        return BarrierProbabilities(
            stop=float(self.barrier.target),
            target=float(self.barrier.stop),
            timeout=float(self.barrier.timeout),
        )

    def resolved_stop_return_for_side(
        self,
        side: str,
        *,
        fallback_mult: float = 1.0,
    ) -> float:
        """Return the adverse underlying move for one trade side."""
        long_stop = self.resolved_stop_return(fallback_mult=fallback_mult)
        long_target = self.resolved_target_return(fallback_mult=fallback_mult)
        if self._normalize_side(side) == "long":
            return long_stop
        return abs(long_target)

    def resolved_target_return_for_side(
        self,
        side: str,
        *,
        fallback_mult: float = 1.0,
    ) -> float:
        """Return the favourable underlying move for one trade side."""
        long_stop = self.resolved_stop_return(fallback_mult=fallback_mult)
        long_target = self.resolved_target_return(fallback_mult=fallback_mult)
        if self._normalize_side(side) == "long":
            return long_target
        return -abs(long_stop)


__all__ = [
    "QUANTILE_LEVELS",
    "BarrierProbabilities",
    "HorizonPrediction",
    "ModelPrediction",
]
