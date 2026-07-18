"""Assembled HRW models (SPEC.md §15, §34).

``HRWForecaster`` — single-step encoder → distribution heads (no direction head per SPEC.md §15).
``HRWWorldModel`` — encoder → RSSM world model → per-horizon distributions.

Market plane only — no portfolio fields enter.  Composed from substitutable parts (DIP).

V1 futures path: pass ``futures: Tensor | None`` (shape [B, T, FUTURES_FEATURE_DIM]) to
``encode()`` / ``forward()`` to incorporate futures microstructure signals (basis, OI, calendar
spread, OI-flow).  When ``futures=None`` the model runs on OHLCV candles only.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from helion_risk_world.config.model_config import ModelConfig
from helion_risk_world.data.option_surface_builder import SURFACE_CONTEXT_FEATURES, SURFACE_STRIKE_CHANNELS
from helion_risk_world.data.regime_builder import REGIME_CONTEXT_FEATURES
from helion_risk_world.encoders.cross_asset_encoder import CrossAssetEncoder
from helion_risk_world.encoders.fusion_encoder import FusionEncoder
from helion_risk_world.encoders.futures_encoder import FUTURES_FEATURE_DIM, FuturesEncoder
from helion_risk_world.encoders.option_surface_encoder import OptionSurfaceEncoder, SurfaceTensors
from helion_risk_world.encoders.regime_encoder import RegimeEncoder
from helion_risk_world.encoders.temporal_encoder import TemporalEncoder
from helion_risk_world.heads.barrier_head import BarrierHead
from helion_risk_world.heads.direction_head import DirectionHead
from helion_risk_world.heads.excursion_barrier_head import ExcursionBarrierHead
from helion_risk_world.heads.excursion_head import ExcursionHead
from helion_risk_world.heads.meta_label_head import MetaLabelHead, primary_side_from_candle_features
from helion_risk_world.heads.ood_head import OODHead
from helion_risk_world.heads.regime_head import RegimeHead
from helion_risk_world.heads.return_head import ReturnQuantileHead
from helion_risk_world.heads.touch_head import TouchHead
from helion_risk_world.heads.uncertainty_head import UncertaintyHead
from helion_risk_world.heads.volatility_head import VolatilityHead
from helion_risk_world.worlds.market_world import MarketWorld
from helion_risk_world.worlds.rssm import RSSM, RSSMState


def _barrier_features(
    mae: Tensor,
    mfe: Tensor,
    volatility: Tensor,
    barrier_context: Tensor | None,
) -> Tensor:
    if barrier_context is None:
        return torch.stack([mae, mfe, volatility], dim=-1)
    if barrier_context.ndim != 2 or barrier_context.shape != (mae.shape[0], 3):
        raise ValueError(f"barrier_context must be [B, 3]; got {tuple(barrier_context.shape)}")
    sigma_t = barrier_context[:, 0].to(device=mae.device, dtype=mae.dtype).clamp_min(1e-6)
    stop_t = barrier_context[:, 1].to(device=mae.device, dtype=mae.dtype).abs().clamp_min(1e-6)
    target_t = barrier_context[:, 2].to(device=mae.device, dtype=mae.dtype).clamp_min(1e-6)
    return torch.stack([mae / stop_t, mfe / target_t, volatility / sigma_t], dim=-1)


def _use_derived_barrier(mode: str, barrier_context: Tensor | None) -> bool:
    return mode == "derived" and barrier_context is not None


class HRWForecaster(nn.Module):
    """Encoders → z_t → distribution heads (SPEC.md §17, Day 4 + regime/OOD).

    forward input:  candle features [B, A, L, F]  (market plane only)
                    futures [B, T, FUTURES_FEATURE_DIM] (optional V1 microstructure)
    forward output: dict with ``z`` [B, d], ``return_quantiles`` [B, Q],
                    ``volatility`` [B], ``barrier_logits`` [B, 3],
                    ``uncertainty`` [B], ``regime_logits`` [B, 6], ``ood_score`` [B, 1]

    Direction head is REMOVED (SPEC.md §15): side is inferred from return-quantile
    asymmetry + barrier probabilities in the planner (SPEC.md §19).

    OOD head must be fitted with ``fit_ood`` after training.
    """

    def __init__(self, n_features: int, cfg: ModelConfig | None = None,
                 n_quantiles: int = 5, meta_label_lookback: int = 12) -> None:
        super().__init__()
        cfg = cfg or ModelConfig()
        d = cfg.latent_dim
        self._meta_label_lookback = meta_label_lookback
        self.temporal = TemporalEncoder(n_features, latent_dim=d, layers=cfg.temporal_layers,
                                        dropout=cfg.dropout)
        self.cross_asset = CrossAssetEncoder(
            n_features, latent_dim=d, n_heads=cfg.cross_asset_heads
        )
        self.futures_encoder = FuturesEncoder(
            FUTURES_FEATURE_DIM,
            out_dim=d,
            hidden_dim=d,
            layers=cfg.futures_conv_layers,
        )
        self.option_surface_encoder = OptionSurfaceEncoder(
            n_channels=len(SURFACE_STRIKE_CHANNELS),
            n_context=len(SURFACE_CONTEXT_FEATURES),
            latent_dim=d,
        )
        self.regime_encoder = RegimeEncoder(len(REGIME_CONTEXT_FEATURES), latent_dim=d)
        self.fusion = FusionEncoder(latent_dim=d, method=cfg.fusion)
        self.return_head = ReturnQuantileHead(latent_dim=d, n_quantiles=n_quantiles)
        self.volatility_head = VolatilityHead(latent_dim=d)
        self.mae_head = ExcursionHead(latent_dim=d)
        self.mfe_head = ExcursionHead(latent_dim=d)
        self.barrier_head = BarrierHead(latent_dim=d, context_dim=3)
        self.excursion_barrier_head = ExcursionBarrierHead()
        # Decomposed barrier architecture (2026-07-13, see heads/direction_head.py's
        # docstring for the full rationale): TouchHead answers "will either barrier be hit at
        # all" (magnitude/volatility question, over the fused z); DirectionHead answers "if
        # so, which way" (direction question) with a skip connection straight to the
        # option-surface embedding, bypassing the shared fusion bottleneck that was measured
        # losing that signal. Selected via barrier_mode="decomposed".
        self.touch_head = TouchHead(latent_dim=d)
        self.direction_head = DirectionHead(latent_dim=d, surface_dim=d)
        self.uncertainty_head = UncertaintyHead(latent_dim=d)
        self.regime_head = RegimeHead(latent_dim=d)
        self.ood_head = OODHead(latent_dim=d)
        self.meta_label_head = MetaLabelHead(latent_dim=d)
        self.latent_dim = d
        self._barrier_mode = "legacy"

    @property
    def barrier_mode(self) -> str:
        return self._barrier_mode

    def set_barrier_mode(self, mode: str) -> None:
        if mode not in {"legacy", "derived", "decomposed"}:
            raise ValueError(f"unsupported barrier mode: {mode!r}")
        self._barrier_mode = mode

    def encode(
        self,
        features: Tensor,
        futures: Tensor | None = None,
        regime: Tensor | None = None,
        surface: SurfaceTensors | None = None,
    ) -> Tensor:
        """Market features [B, A, L, F] (+ optional futures/regime/option-surface) -> z_t [B, d]."""
        temporal = self.temporal(features)
        cross = self.cross_asset(features)
        futures_emb = self.futures_encoder(futures) if futures is not None else None
        surface_emb = self.option_surface_encoder(surface) if surface is not None else None
        regime_emb = self.regime_encoder(regime) if regime is not None else None
        return self.fusion(
            temporal, cross=cross, futures=futures_emb, option_surface=surface_emb, regime=regime_emb
        )

    @torch.no_grad()
    def fit_ood(
        self,
        features: Tensor,
        futures: Tensor | None = None,
        regime: Tensor | None = None,
        surface: SurfaceTensors | None = None,
    ) -> None:
        """Fit the OOD detector on the latents of ``features`` (call once after training)."""
        dev = next(self.parameters()).device
        features = features.to(dev)
        if futures is not None:
            futures = futures.to(dev)
        if regime is not None:
            regime = regime.to(dev)
        if surface is not None:
            surface = SurfaceTensors(*(t.to(dev) for t in surface))
        was_training = self.training
        self.eval()
        self.ood_head.fit(self.encode(features, futures, regime, surface))
        if was_training:
            self.train()

    def forward(
        self,
        features: Tensor,
        futures: Tensor | None = None,
        regime: Tensor | None = None,
        barrier_context: Tensor | None = None,
        surface: SurfaceTensors | None = None,
        primary_side: Tensor | None = None,
    ) -> dict[str, Tensor]:
        # Inlined (not a call to self.encode()) so `surface_emb` is available for
        # DirectionHead's skip connection without changing encode()'s existing return
        # contract (many callers -- fit_ood, WorldModelTrainer -- depend on it returning
        # just z).
        temporal = self.temporal(features)
        cross = self.cross_asset(features)
        futures_emb = self.futures_encoder(futures) if futures is not None else None
        surface_emb = self.option_surface_encoder(surface) if surface is not None else None
        regime_emb = self.regime_encoder(regime) if regime is not None else None
        z = self.fusion(
            temporal, cross=cross, futures=futures_emb, option_surface=surface_emb, regime=regime_emb
        )
        volatility = self.volatility_head(z)
        mae = self.mae_head(z)
        mfe = self.mfe_head(z)
        barrier_features = _barrier_features(mae, mfe, volatility, barrier_context)
        touch_logit: Tensor | None = None
        direction_logit: Tensor | None = None
        if self._barrier_mode == "decomposed":
            touch_logit = self.touch_head(z)
            surface_for_direction = (
                surface_emb
                if surface_emb is not None
                else torch.zeros(z.shape[0], self.latent_dim, device=z.device, dtype=z.dtype)
            )
            direction_logit = self.direction_head(z, surface_for_direction)
            p_touch = torch.sigmoid(touch_logit)
            p_up_given_touch = torch.sigmoid(direction_logit)
            p_target = p_touch * p_up_given_touch
            p_stop = p_touch * (1.0 - p_up_given_touch)
            p_timeout = 1.0 - p_touch
            probs = torch.stack([p_stop, p_target, p_timeout], dim=-1).clamp_min(1e-8)
            barrier_logits = torch.log(probs)
        elif _use_derived_barrier(self._barrier_mode, barrier_context):
            barrier_logits = self.excursion_barrier_head(barrier_features)
        else:
            barrier_logits = self.barrier_head(z, context=barrier_features)
        if primary_side is None:
            primary_side = primary_side_from_candle_features(
                features, lookback=self._meta_label_lookback
            )
        out = {
            "z": z,
            "return_quantiles": self.return_head(z),      # [B, Q]
            "volatility": volatility,                     # [B]
            "mae": mae,                                   # [B]
            "mfe": mfe,                                   # [B]
            "barrier_logits": barrier_logits,             # [B, 3]
            "uncertainty": self.uncertainty_head(z),       # [B]
            "regime_logits": self.regime_head(z),          # [B, 6]
            "ood_score": self.ood_head(z),                 # [B, 1]
            "primary_side": primary_side,                  # [B]
            "meta_label_logit": self.meta_label_head(z, primary_side),  # [B]
        }
        if touch_logit is not None:
            out["touch_logit"] = touch_logit               # [B]
            out["direction_logit"] = direction_logit        # [B]
        return out


class HRWWorldModel(nn.Module):
    """Encoders → RSSM world model → per-horizon distributions (SPEC.md §13, §14).

    The world-model counterpart to ``HRWForecaster``: the RSSM rolls the trained prior
    forward to produce calibrated epistemic uncertainty from the ensemble spread.

    forward output keys: ``z`` [B, d], ``horizons``, ``return_quantiles`` [B, |H|, Q],
    ``barrier_probs`` [B, 3], ``regime_logits`` [B, R], ``volatility`` [B, |H|],
    ``epistemic`` [B, |H|], ``aleatoric`` [B, |H|], ``ood_score`` [B, 1].

    V1 futures path: pass ``futures: [B, T, FUTURES_FEATURE_DIM]`` to encode().
    Inference uses T=1 window (TemporalEncoder captures lookback; RSSM imagines forward).
    RSSM training uses T>1 via WorldModelTrainer.encode_sequence().
    """

    def __init__(
        self,
        n_features: int,
        cfg: ModelConfig | None = None,
        horizons: tuple[int, ...] = (1, 3, 6),
        n_samples: int = 16,
        n_quantiles: int = 5,
    ) -> None:
        super().__init__()
        cfg = cfg or ModelConfig()
        d = cfg.latent_dim
        self.temporal = TemporalEncoder(n_features, latent_dim=d, layers=cfg.temporal_layers,
                                        dropout=cfg.dropout)
        self.cross_asset = CrossAssetEncoder(
            n_features, latent_dim=d, n_heads=cfg.cross_asset_heads
        )
        self.futures_encoder = FuturesEncoder(
            FUTURES_FEATURE_DIM,
            out_dim=d,
            hidden_dim=d,
            layers=cfg.futures_conv_layers,
        )
        self.option_surface_encoder = OptionSurfaceEncoder(
            n_channels=len(SURFACE_STRIKE_CHANNELS),
            n_context=len(SURFACE_CONTEXT_FEATURES),
            latent_dim=d,
        )
        self.regime_encoder = RegimeEncoder(len(REGIME_CONTEXT_FEATURES), latent_dim=d)
        self.fusion = FusionEncoder(latent_dim=d, method=cfg.fusion)
        # RSSM: embed_dim matches encoder output d; stoch_dim = d//4 for a compact latent
        rssm = RSSM(stoch_dim=max(1, d // 4), deter_dim=d, embed_dim=d)
        self.market_world = MarketWorld(rssm, n_quantiles=n_quantiles,
                                        horizons=horizons, n_samples=n_samples)
        self.latent_dim = d
        self.horizons = tuple(sorted(set(horizons)))
        self.register_buffer("_ood_boundary", torch.tensor(1.0))
        self.register_buffer("_ood_scale", torch.tensor(1.0))
        self.register_buffer("_ood_fitted", torch.tensor(0.0))
        # Fixed, architecture-derived fallback scale for _normalize_ood's unfitted
        # path (review finding M1): the raw OOD score is a sum of stoch_dim
        # independent log-prob terms, so its typical magnitude scales with
        # stoch_dim regardless of any particular batch's contents. Previously this
        # divided by the CURRENT batch's own mean |raw| — batch-composition-
        # dependent and non-reproducible (a batch of one outlier scored completely
        # differently than the same sample scored alongside typical ones).
        self.register_buffer("_ood_unfitted_scale", torch.tensor(float(max(rssm.stoch_dim, 1))))

    def encode(
        self,
        features: Tensor,
        futures: Tensor | None = None,
        regime: Tensor | None = None,
        surface: SurfaceTensors | None = None,
    ) -> Tensor:
        """Market features (+ optional futures/regime/option-surface) -> z_t [B, d]."""
        temporal = self.temporal(features)
        cross = self.cross_asset(features)
        futures_emb = self.futures_encoder(futures) if futures is not None else None
        surface_emb = self.option_surface_encoder(surface) if surface is not None else None
        regime_emb = self.regime_encoder(regime) if regime is not None else None
        return self.fusion(
            temporal, cross=cross, futures=futures_emb, option_surface=surface_emb, regime=regime_emb
        )

    @torch.no_grad()
    def fit_ood(
        self,
        features: Tensor,
        futures: Tensor | None = None,
        regime: Tensor | None = None,
        surface: SurfaceTensors | None = None,
    ) -> None:
        """Fit the runtime OOD normalizer on RSSM prior surprise scores."""
        was_training = self.training
        self.eval()
        dev = next(self.parameters()).device
        features = features.to(dev)
        if futures is not None:
            futures = futures.to(dev)
        if regime is not None:
            regime = regime.to(dev)
        if surface is not None:
            surface = SurfaceTensors(*(t.to(dev) for t in surface))
        z = self.encode(features, futures, regime, surface)
        window_e = z.unsqueeze(0)
        state = self.market_world.filter(window_e)
        raw = -self.market_world.rssm.prior(state.h).log_prob(state.z).sum(dim=-1)
        boundary = torch.quantile(raw, 0.975)
        scale = (boundary - raw.median()).clamp_min(1e-6)
        self._ood_boundary.copy_(boundary)
        self._ood_scale.copy_(scale)
        self._ood_fitted.fill_(1.0)
        if was_training:
            self.train()

    @property
    def barrier_mode(self) -> str:
        return self.market_world.barrier_mode

    def set_barrier_mode(self, mode: str) -> None:
        self.market_world.set_barrier_mode(mode)

    def _normalize_ood(self, raw: Tensor) -> Tensor:
        if float(self._ood_fitted) < 0.5:
            return torch.sigmoid(raw / self._ood_unfitted_scale)
        return torch.sigmoid((raw - self._ood_boundary) / self._ood_scale)

    def forward(
        self,
        features: Tensor,
        futures: Tensor | None = None,
        regime: Tensor | None = None,
        barrier_context: Tensor | None = None,
        n_samples: int | None = None,
        state: RSSMState | None = None,
        surface: SurfaceTensors | None = None,
        *,
        deterministic: bool = False,
    ) -> dict[str, object]:
        """state: optional RSSMState from a previous call (review finding H1).

        When None (the training-time default), the single encoded observation is
        treated as a length-1 window rolled from a zero-initialised RSSM state —
        correct for i.i.d. training batches. Live/paper inference callers that
        want h_t to reflect genuine bar-to-bar history should pass the RSSMState
        returned in this call's output (key "state") back in on the next call.

        deterministic (review finding M3): use RSSM means instead of samples, for
        reproducible eval/backtest runs. Off by default.
        """
        z = self.encode(features, futures, regime, surface)  # [B, d]
        # Treat the single encoded observation as a window of length 1 for the RSSM.
        # TemporalEncoder captures lookback history; RSSM imagines calibrated futures.
        window_e = z.unsqueeze(0)                    # [1, B, d]
        world = self.market_world(
            window_e, barrier_context=barrier_context, n_samples=n_samples, state=state,
            deterministic=deterministic,
        )
        return {
            "z": z,
            "horizons": world["horizons"],
            "state": world["state"],                          # RSSMState s_t; thread back in next call
            "return_quantiles": world["return_quantiles"],    # [B, |H|, Q]
            "barrier_logits": world["barrier_logits"],        # [B, 3] log predictive probs at
                                                                # management horizon — bugfix:
                                                                # was silently dropped, so
                                                                # WorldModelLoss's barrier CE and
                                                                # excursion-aux terms were both
                                                                # silently no-ops in training.
            "barrier_logits_intermediate": world.get("barrier_logits_intermediate"),
                                                                # [B, |H|-1, 3] or None — deep
                                                                # supervision at the shorter
                                                                # (non-management) horizons.
            "barrier_probs": world["barrier_probs"],          # [B, 3]
            "regime_logits": world["regime_logits"],          # [B, R]
            "volatility": world["volatility"],                # [B, |H|]
            "mae": world["mae"],                              # [B, |H|]
            "mfe": world["mfe"],                              # [B, |H|]
            "epistemic": world["epistemic"],                  # [B, |H|]
            "aleatoric": world["aleatoric"],                  # [B, |H|]
            "ood_score": self._normalize_ood(world["ood_score"]).unsqueeze(-1),  # [B, 1]
        }


__all__ = ["HRWForecaster", "HRWWorldModel"]
