"""Inference bridges: model tensors -> ModelPrediction schema (SPEC.md §17, §18).

``ForecasterPredictor`` wraps ``HRWForecaster`` (single-step, single-horizon).
``WorldModelPredictor`` wraps ``HRWWorldModel`` (RSSM, multi-horizon).

Schema (SPEC.md §11, §15, §19):
  * ``horizon_preds``  — per-horizon HorizonPrediction (horizon_bars, return_quantiles, volatility)
  * ``barrier``        — BarrierProbabilities at H=max (management horizon)
  * ``sigma_H``        — volatility at H=max (used by PortfolioWorld)
  * ``mae``            — learned adverse-excursion prediction at H=max
  * ``regime_probs``   — at top level of ModelPrediction (NOT per HorizonPrediction)
  * NO direction_probs — direction inferred from return-quantile asymmetry in the planner

V1 futures path: pass ``futures: Tensor | None`` ([B, T, F] or [T, F] for predict_one)
to incorporate futures microstructure signals.

SRP: tensor → schema translation only.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from helion_risk_world.barrier_context import BarrierContext
from helion_risk_world.data.option_surface_builder import featurize_surface
from helion_risk_world.data.regime_builder import featurize_regime
from helion_risk_world.encoders.option_surface_encoder import SurfaceTensors
from helion_risk_world.heads.regime_head import REGIME_CLASSES
from helion_risk_world.heads.return_head import DEFAULT_QUANTILES
from helion_risk_world.model import HRWForecaster, HRWWorldModel
from helion_risk_world.schemas.market_schema import EventContext, RegimeContext
from helion_risk_world.schemas.option_chain_schema import OptionSurfaceSnapshot
from helion_risk_world.schemas.prediction_schema import (
    BarrierProbabilities,
    HorizonPrediction,
    ModelPrediction,
)
from helion_risk_world.worlds.rssm import RSSMState

# A regime/event context for one decision step.
RegimeInput = tuple[RegimeContext, EventContext]

_EPS = 1e-9

_log = logging.getLogger(__name__)
_warned_uncalibrated_epistemic = False


def _warn_uncalibrated_epistemic_once() -> None:
    """Review finding H9: ForecasterPredictor has no ensemble, so epistemic is always
    0.0 — this silently disables ManagementLoop's/EpistemicRiskBlock's/PositionSizer's
    epistemic-gated safety checks for any run using model_kind='forecaster' (the
    default). Warn loudly once per process instead of failing silently."""
    global _warned_uncalibrated_epistemic
    if not _warned_uncalibrated_epistemic:
        _log.warning(
            "ForecasterPredictor emits epistemic=0.0 (uncalibrated placeholder, no "
            "RSSM ensemble available). Epistemic-gated safety checks in "
            "ManagementLoop/EpistemicRiskBlock/PositionSizer are inert for this "
            "model_kind. Use model_kind='world_model' if those checks must be live."
        )
        _warned_uncalibrated_epistemic = True


def _batch_regime_tensor(regimes: Sequence[RegimeInput], device: torch.device) -> Tensor:
    rows = [featurize_regime(reg, evt) for reg, evt in regimes]
    return torch.tensor(np.stack(rows), device=device)


def _batch_surface_tensors(
    surfaces: Sequence[OptionSurfaceSnapshot | None], device: torch.device
) -> SurfaceTensors | None:
    """Batch ``OptionSurfaceSnapshot``s into model-ready ``SurfaceTensors`` (feature-onboarding
    pass). A ``None`` entry (no chain available at that row) becomes a zero-filled,
    all-masked row -- same "no signal" convention as ``FeatureBuilder.build_surface_history``,
    not an excluded row."""
    if all(s is None for s in surfaces):
        return None
    grids, masks, contexts = [], [], []
    for surf in surfaces:
        if surf is None:
            grids.append(None)
            masks.append(None)
            contexts.append(None)
            continue
        g, m, c = featurize_surface(surf)
        grids.append(g)
        masks.append(m)
        contexts.append(c)
    shape_g = next(g.shape for g in grids if g is not None)
    shape_m = next(m.shape for m in masks if m is not None)
    shape_c = next(c.shape for c in contexts if c is not None)
    grids = [g if g is not None else np.zeros(shape_g, dtype=np.float32) for g in grids]
    masks = [m if m is not None else np.zeros(shape_m, dtype=np.float32) for m in masks]
    contexts = [c if c is not None else np.zeros(shape_c, dtype=np.float32) for c in contexts]
    return SurfaceTensors(
        grid=torch.tensor(np.stack(grids), device=device),
        mask=torch.tensor(np.stack(masks), device=device),
        context=torch.tensor(np.stack(contexts), device=device),
    )


def _build_horizon_prediction(
    quant: dict[float, float],
    vol: float,
    horizon_bars: int,
) -> HorizonPrediction:
    """Assemble a HorizonPrediction from decoded model outputs (shared; DRY)."""
    return HorizonPrediction(
        horizon_bars=horizon_bars,
        return_quantiles=quant,
        volatility=vol,
    )


def _derived_mae_from_quantiles(hp: HorizonPrediction) -> float:
    """Legacy MAE proxy from the management-horizon quantile spread."""
    q = hp.return_quantiles
    q10 = q.get(min(q.keys()), 0.0)
    q50 = q.get(0.5, 0.0)
    return abs(q50 - q10)


def _batch_barrier_context_tensor(
    contexts: Sequence[BarrierContext | None] | None,
    device: torch.device,
) -> Tensor | None:
    if contexts is None:
        return None
    rows = []
    for context in contexts:
        if context is None:
            rows.append((0.0, 0.0, 0.0))
        else:
            rows.append((context.sigma, context.stop_return, context.target_return))
    return torch.tensor(rows, dtype=torch.float32, device=device)


class ForecasterPredictor:
    """Wrap a trained ``HRWForecaster`` and emit ``ModelPrediction``s (SPEC.md §17)."""

    def __init__(
        self,
        model: HRWForecaster,
        *,
        device: torch.device | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        horizon_bars: int = 12,
        use_predicted_mae: bool = True,
        use_barrier_geometry: bool = True,
    ) -> None:
        if 0.5 not in quantiles:
            raise ValueError("quantiles must include the 0.5 median")
        self._model = model.eval()
        self._device = device or next(model.parameters()).device
        self._quantiles = tuple(quantiles)
        self._horizon_bars = horizon_bars
        self._use_predicted_mae = use_predicted_mae
        self._use_barrier_geometry = use_barrier_geometry

    @torch.no_grad()
    def predict_batch(
        self,
        features: Tensor,
        symbol: str,
        timestamps: Sequence[datetime],
        futures: Tensor | None = None,
        regimes: Sequence[RegimeInput] | None = None,
        barrier_contexts: Sequence[BarrierContext | None] | None = None,
        surfaces: Sequence[OptionSurfaceSnapshot | None] | None = None,
    ) -> list[ModelPrediction]:
        """Features [B, A, L, F] (+ optional futures [B, T, F] / regimes / option surfaces) ->
        one prediction per row.

        futures: [B, T, FUTURES_FEATURE_DIM] — per-row futures microstructure windows.
        """
        if features.ndim != 4:
            raise ValueError(f"features must be [B, A, L, F]; got {tuple(features.shape)}")
        if features.shape[0] != len(timestamps):
            raise ValueError("timestamps length must match the batch size")
        if futures is not None and futures.shape[0] != features.shape[0]:
            raise ValueError("futures batch size must match features batch size")
        if barrier_contexts is not None and len(barrier_contexts) != features.shape[0]:
            raise ValueError("barrier_contexts length must match the batch size")
        if surfaces is not None and len(surfaces) != features.shape[0]:
            raise ValueError("surfaces length must match the batch size")
        regime_tensor = None
        if regimes is not None:
            if len(regimes) != features.shape[0]:
                raise ValueError("regimes length must match the batch size")
            regime_tensor = _batch_regime_tensor(regimes, self._device)
        fut = futures.to(self._device) if futures is not None else None
        barrier_tensor = (
            _batch_barrier_context_tensor(barrier_contexts, self._device)
            if self._use_barrier_geometry
            else None
        )
        surface_tensor = _batch_surface_tensors(surfaces, self._device) if surfaces is not None else None
        out = self._model(features.to(self._device), fut, regime_tensor, barrier_tensor, surface=surface_tensor)
        rq = out["return_quantiles"].cpu()                              # [B, Q]
        vol = out["volatility"].reshape(-1).cpu()                       # [B]
        mae_pred = out["mae"].reshape(-1).cpu()                         # [B]
        mfe_pred = out["mfe"].reshape(-1).cpu()                         # [B]
        barrier = F.softmax(out["barrier_logits"], dim=-1).cpu()        # [B, 3]
        unc = out["uncertainty"].reshape(-1).cpu()                      # [B]
        ood = out["ood_score"].reshape(-1).cpu()                        # [B]
        regime_probs_t = F.softmax(out["regime_logits"], dim=-1).cpu() # [B, R]
        primary_side_t = out["primary_side"].reshape(-1).cpu()          # [B] in {-1,0,1}
        meta_prob_t = torch.sigmoid(out["meta_label_logit"]).reshape(-1).cpu()  # [B]
        return [
            self._to_prediction(
                rq[i], float(vol[i]), float(mae_pred[i]), float(mfe_pred[i]), barrier[i], float(unc[i]), float(ood[i]), regime_probs_t[i],
                symbol, timestamps[i], barrier_contexts[i] if barrier_contexts is not None else None,
                int(primary_side_t[i]), float(meta_prob_t[i]),
            )
            for i in range(features.shape[0])
        ]

    def predict_one(
        self,
        features: Tensor,
        symbol: str,
        ts: datetime,
        futures: Tensor | None = None,
        regime: RegimeInput | None = None,
        barrier_context: BarrierContext | None = None,
        surface: OptionSurfaceSnapshot | None = None,
    ) -> ModelPrediction:
        """Features [A, L, F] (+ optional futures [T, F] / regime / option surface) ->
        one ModelPrediction."""
        fut_batch = futures.unsqueeze(0) if futures is not None else None
        regimes = [regime] if regime is not None else None
        barrier_contexts = [barrier_context] if barrier_context is not None else None
        surfaces = [surface] if surface is not None else None
        return self.predict_batch(
            features.unsqueeze(0),
            symbol,
            [ts],
            fut_batch,
            regimes,
            barrier_contexts,
            surfaces,
        )[0]

    def _to_prediction(
        self,
        rq: Tensor,
        vol: float,
        mae_pred: float,
        mfe_pred: float,
        barrier: Tensor,
        aleatoric: float,
        ood: float,
        regime_probs_t: Tensor,
        symbol: str,
        ts: datetime,
        barrier_context: BarrierContext | None,
        primary_side: int = 0,
        meta_label_prob: float | None = None,
    ) -> ModelPrediction:
        quant = {lvl: float(rq[k]) for k, lvl in enumerate(self._quantiles)}
        hp = _build_horizon_prediction(quant, vol, self._horizon_bars)
        mae = float(mae_pred if self._use_predicted_mae else _derived_mae_from_quantiles(hp))
        sigma_H = hp.volatility
        bp = BarrierProbabilities(
            stop=float(barrier[0]),
            target=float(barrier[1]),
            timeout=float(barrier[2]),
        )
        regime_probs = {r: float(regime_probs_t[k]) for k, r in enumerate(REGIME_CLASSES)}
        _warn_uncalibrated_epistemic_once()
        return ModelPrediction(
            symbol=symbol, ts=ts,
            horizon_preds=[hp],
            barrier=bp, mae=mae, mfe=float(max(mfe_pred, 0.0)), sigma_H=sigma_H,
            stop_return=barrier_context.stop_return if barrier_context is not None else None,
            target_return=barrier_context.target_return if barrier_context is not None else None,
            primary_side=primary_side,
            meta_label_prob=meta_label_prob if primary_side != 0 else None,
            epistemic=0.0,
            aleatoric=aleatoric,
            ood_score=ood,
            regime_probs=regime_probs,
            epistemic_calibrated=False,
        )


class WorldModelPredictor:
    """Wrap an ``HRWWorldModel`` -> a multi-horizon ``ModelPrediction`` (SPEC.md §13, §18).

    ``epistemic`` per horizon is the RSSM ensemble spread (calibrated).
    ``barrier`` + ``regime_probs`` are at the management horizon H=max(horizons).

    Persisted RSSM state (review finding H1): ``HRWWorldModel.forward()`` treats a
    single call as a length-1 window rolled from whatever state is passed in. Training
    always rolls a full multi-step sequence from a zero state, but calling ``predict_one``
    repeatedly (the live/paper-trading pattern — one bar at a time) with no persisted
    state means every call resets h_t to zero, discarding real bar-to-bar history the
    RSSM was trained to use. When ``persist_state=True`` (the default), this predictor
    carries the returned RSSMState forward into the next ``predict_one`` call on ascending
    timestamps, and resets it whenever a new trading day begins (``ts.date()`` changes) —
    a persisted belief should not span an overnight gap. Set ``persist_state=False`` to
    recover the original reset-every-call behavior for A/B comparison; also see
    ``reset_state()`` for callers that detect a data discontinuity themselves.
    """

    def __init__(
        self,
        model: HRWWorldModel,
        *,
        device: torch.device | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        use_predicted_mae: bool = True,
        use_barrier_geometry: bool = True,
        persist_state: bool = True,
        deterministic: bool = False,
    ) -> None:
        if 0.5 not in quantiles:
            raise ValueError("quantiles must include the 0.5 median")
        self._model = model.eval()
        self._device = device or next(model.parameters()).device
        self._quantiles = tuple(quantiles)
        self._use_predicted_mae = use_predicted_mae
        self._use_barrier_geometry = use_barrier_geometry
        self._persist_state = persist_state
        # Review finding M3: use RSSM means instead of samples throughout, so
        # repeated calls with identical inputs give identical predictions —
        # useful for reproducible eval/backtest runs. Off by default (unchanged
        # stochastic behavior); note this also collapses epistemic (ensemble
        # spread) toward 0, since every ensemble member becomes identical.
        self._deterministic = deterministic
        self._state: RSSMState | None = None
        self._last_ts: datetime | None = None

    @property
    def state(self) -> RSSMState | None:
        """The RSSM belief state carried from the most recent predict_one() call."""
        return self._state

    def reset_state(self) -> None:
        """Clear the persisted RSSM belief state.

        Call this at a known data discontinuity (e.g. a detected feed gap or roll)
        so the next predict_one() starts fresh instead of carrying stale belief
        across it. Also happens automatically at a trading-day boundary.
        """
        self._state = None
        self._last_ts = None

    @torch.no_grad()
    def predict_one(
        self,
        features: Tensor,
        symbol: str,
        ts: datetime,
        futures: Tensor | None = None,
        regime: RegimeInput | None = None,
        barrier_context: BarrierContext | None = None,
        n_samples: int | None = None,
        surface: OptionSurfaceSnapshot | None = None,
    ) -> ModelPrediction:
        """Features [A, L, F] (+ optional futures [T, F] / regime / option surface) -> a
        multi-horizon ModelPrediction."""
        if self._persist_state and self._last_ts is not None and ts.date() != self._last_ts.date():
            self._state = None  # new trading day: don't carry belief across the overnight gap
        reg = _batch_regime_tensor([regime], self._device) if regime is not None else None
        fut = futures.unsqueeze(0).to(self._device) if futures is not None else None
        barrier_tensor = None
        if self._use_barrier_geometry and barrier_context is not None:
            barrier_tensor = _batch_barrier_context_tensor([barrier_context], self._device)
        surface_tensor = _batch_surface_tensors([surface], self._device) if surface is not None else None
        out = self._model(
            features.unsqueeze(0).to(self._device),
            fut,
            reg,
            barrier_tensor,
            n_samples=n_samples,
            state=self._state if self._persist_state else None,
            surface=surface_tensor,
            deterministic=self._deterministic,
        )
        if self._persist_state:
            self._state = out["state"]
        self._last_ts = ts

        rq = out["return_quantiles"][0].cpu()          # [|H|, Q]
        vl = out["volatility"][0].cpu()                # [|H|]
        epi = out["epistemic"][0].cpu()                # [|H|]
        aleatoric = out["aleatoric"][0].cpu()          # [|H|]
        mae_pred = out["mae"][0].cpu()                 # [|H|]
        mfe_pred = out["mfe"][0].cpu()                 # [|H|]

        barrier_probs_t = out["barrier_probs"][0].cpu()              # [3]
        regime_probs_t = F.softmax(out["regime_logits"][0], dim=-1).cpu()  # [R]
        horizon_steps = list(out["horizons"])

        horizon_preds = [
            _build_horizon_prediction(
                {lvl: float(rq[hi, k]) for k, lvl in enumerate(self._quantiles)},
                float(vl[hi]),
                step,
            )
            for hi, step in enumerate(horizon_steps)
        ]
        management_hp = horizon_preds[-1]   # management horizon = last (largest)
        mae = float(mae_pred[-1]) if self._use_predicted_mae else _derived_mae_from_quantiles(management_hp)
        mfe = float(max(mfe_pred[-1], 0.0))
        sigma_H = management_hp.volatility

        bp = BarrierProbabilities(
            stop=float(barrier_probs_t[0]),
            target=float(barrier_probs_t[1]),
            timeout=float(barrier_probs_t[2]),
        )
        regime_probs = {r: float(regime_probs_t[k]) for k, r in enumerate(REGIME_CLASSES)}

        return ModelPrediction(
            symbol=symbol, ts=ts,
            horizon_preds=horizon_preds,
            barrier=bp, mae=mae, mfe=mfe, sigma_H=sigma_H,
            stop_return=barrier_context.stop_return if barrier_context is not None else None,
            target_return=barrier_context.target_return if barrier_context is not None else None,
            epistemic=float(epi[-1]),      # management horizon spread
            aleatoric=float(aleatoric[-1]),
            ood_score=float(out["ood_score"][0, 0]),
            regime_probs=regime_probs,
        )


__all__ = ["ForecasterPredictor", "WorldModelPredictor"]
