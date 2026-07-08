"""Market World — trained RSSM latent world model (SPEC.md §14, Appendix A).

forward() rolls the RSSM over the observed window via filter() then imagines calibrated
futures by sampling the TRAINED prior.  The direction head is REMOVED (SPEC.md §3, §15) —
direction is near-unpredictable; side is chosen from return-quantile asymmetry + barrier
probabilities in the planner.

Inputs: market-plane tensors ONLY (no portfolio fields — SPEC.md §6).
SRP: encode + RSSM forward pass + head decode; portfolio/planner/shield live elsewhere.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from helion_risk_world.heads.barrier_head import BarrierHead, BARRIER_CLASSES
from helion_risk_world.heads.excursion_barrier_head import ExcursionBarrierHead
from helion_risk_world.heads.excursion_head import ExcursionHead
from helion_risk_world.heads.regime_head import RegimeHead
from helion_risk_world.heads.return_head import DEFAULT_QUANTILES, ReturnQuantileHead
from helion_risk_world.heads.volatility_head import VolatilityHead
from helion_risk_world.worlds.rssm import RSSM, RSSMState
from helion_risk_world.worlds.rollout_engine import RolloutEngine
from helion_risk_world.worlds.ensemble_quantiles import combine_ensemble_quantiles


class MarketWorld(nn.Module):
    """Tri-encoder + RSSM world model → per-horizon distributions + barrier/MAE.

    filter()  : roll RSSM over observed window using posteriors  → s_t = (h_t, z_t)
    imagine() : sample TRAINED prior forward from s_t           → ensemble [S, B, |H|, ...]
    forward() : filter + imagine + decode heads                  → prediction dict

    The ensemble spread is CALIBRATED epistemic uncertainty because the prior is
    trained (KL to posterior) — not arbitrary randn noise.
    """

    def __init__(
        self,
        rssm: RSSM,
        n_quantiles: int = 5,
        horizons: Sequence[int] = (3, 6, 12),
        n_samples: int = 16,
        n_regime_classes: int = 6,
    ) -> None:
        super().__init__()
        self.rssm = rssm
        self._engine = RolloutEngine(rssm)
        head_in = rssm.deter_dim + rssm.stoch_dim

        # Per-horizon heads (return quantiles + vol) — SPEC.md §15
        self.return_head = ReturnQuantileHead(latent_dim=head_in, n_quantiles=n_quantiles)
        # Quantile probability levels each return_head output slot corresponds to (must
        # match ReturnQuantileHead.DEFAULT_QUANTILES / schemas.prediction_schema.QUANTILE_LEVELS)
        # — needed to combine the ensemble via mixture-quantile pooling (see
        # worlds/ensemble_quantiles.py), not just to label the output.
        self.register_buffer(
            "_quantile_levels",
            torch.tensor(DEFAULT_QUANTILES[:n_quantiles], dtype=torch.float32),
            persistent=False,  # fixed constant, not learned state — must not be required in
                                # older checkpoints' state_dict (added after they were saved)
        )
        self.vol_head = VolatilityHead(latent_dim=head_in)
        self.mae_head = ExcursionHead(latent_dim=head_in)
        self.mfe_head = ExcursionHead(latent_dim=head_in)

        # Single-H heads (barrier + regime) — SPEC.md §15, §11.5
        self.barrier_head = BarrierHead(latent_dim=head_in, context_dim=3)
        self.excursion_barrier_head = ExcursionBarrierHead()
        self.regime_head = RegimeHead(latent_dim=head_in)

        self._horizons = tuple(sorted(set(horizons)))
        self._n_samples = n_samples
        self._h_management = max(self._horizons)  # H = max(horizon_bars) for barrier/MAE
        self._barrier_mode = "legacy"

    @property
    def management_horizon(self) -> int:
        return self._h_management

    @property
    def barrier_mode(self) -> str:
        return self._barrier_mode

    def set_barrier_mode(self, mode: str) -> None:
        if mode not in {"legacy", "derived"}:
            raise ValueError(f"unsupported barrier mode: {mode!r}")
        self._barrier_mode = mode

    def filter(
        self, window_e: Tensor, state: RSSMState | None = None, *, deterministic: bool = False
    ) -> RSSMState:
        """Roll the RSSM over the observed window to infer s_t.

        window_e: [T, B, embed_dim]  (time first, from the fusion encoder)
        state: optional starting state s_{t-1} (review finding H1). When None,
               rolls from a zero-initialised state as before — correct for
               training, which always observes a full T-step window. Live/paper
               inference calls this with T=1; passing the previous call's
               returned state here lets h_t genuinely depend on the full bar
               history instead of being reset to zero on every call.
        deterministic: use posterior means instead of samples (review M3) — for
               reproducible eval/backtest runs.
        Returns s_t = (h_t, z_t) — the state at the END of the lookback window.
        """
        return self.rssm.filter(window_e, state=state, deterministic=deterministic)

    def imagine(
        self,
        state: RSSMState,
        horizons: Sequence[int] | None = None,
        n_samples: int | None = None,
        *,
        deterministic: bool = False,
    ) -> Tensor:
        """state → ensemble [S, B, |H|, deter+stoch], sampling the TRAINED prior."""
        return self._engine.rollout(
            state, horizons or self._horizons, n_samples or self._n_samples,
            deterministic=deterministic,
        )

    def forward(
        self,
        window_e: Tensor,
        horizons: Sequence[int] | None = None,
        barrier_context: Tensor | None = None,
        n_samples: int | None = None,
        state: RSSMState | None = None,
        *,
        deterministic: bool = False,
    ) -> dict[str, object]:
        """filter(window_e, state) → imagine → decode all heads.

        window_e: [T, B, embed_dim] — encoded observations from the fusion encoder.
        state: optional starting RSSMState threaded from a previous call (review
               finding H1) — see filter()'s docstring.
        deterministic: use RSSM means instead of samples throughout (review M3) —
               for reproducible eval/backtest runs. Off by default.

        Returns dict with keys:
          horizons            : sorted horizon list
          state               : RSSMState s_t
          rollout             : [S, B, |H|, d] raw latent ensemble
          return_quantiles    : [B, |H|, Q]  (ensemble-mean, non-decreasing)
          volatility          : [B, |H|]     (ensemble-mean vol per horizon)
          mae                 : [B, |H|]     (ensemble-mean max adverse excursion)
          mfe                 : [B, |H|]     (ensemble-mean max favourable excursion)
          barrier_logits      : [B, 3]       (at H=max, log predictive class-probs)
          barrier_logits_intermediate : [B, |H|-1, 3] or None (deep supervision at the
                                         non-management horizons; same head weights as
                                         barrier_logits, shorter RSSM rollout per step)
          barrier_probs       : [B, 3]       (ensemble-mean predictive class-probs)
          regime_logits       : [B, R]       (at H=max, log predictive class-probs)
          epistemic           : [B, |H|]     (ensemble spread, grows with horizon)
          aleatoric           : [B, |H|]     (mean quantile width)
          ood_score           : [B]          (−log p_prior(z_t | h_t))
        """
        hs = tuple(sorted(set(horizons))) if horizons else self._horizons
        ns = n_samples or self._n_samples

        # Filter: infer s_t from the observed window (from `state` if provided)
        state = self.filter(window_e, state=state, deterministic=deterministic)

        # OOD = negative log-likelihood under the TRAINED prior (SPEC.md §16)
        prior_dist = self.rssm.prior(state.h)
        ood_score = -prior_dist.log_prob(state.z).sum(dim=-1)  # [B]

        # Imagine: sample the trained prior forward
        roll = self.imagine(state, hs, ns, deterministic=deterministic)  # [S, B, |H|, d]
        s, b, h_len, d = roll.shape
        flat = roll.reshape(s * b * h_len, d)

        # Decode heads on all (sample, horizon) combinations
        rq = self.return_head(flat).view(s, b, h_len, -1)   # [S, B, |H|, Q]
        vl = self.vol_head(flat).view(s, b, h_len)           # [S, B, |H|]
        mae = self.mae_head(flat).view(s, b, h_len)         # [S, B, |H|]
        mfe = self.mfe_head(flat).view(s, b, h_len)         # [S, B, |H|]

        # Barrier at EVERY horizon step (deep supervision — Phase 5b), regime only at the
        # MANAGEMENT horizon (last horizon = H). Reusing the same head weights at every step
        # means gradients to barrier_head/excursion_barrier_head reach the RSSM via shorter
        # paths too (e.g. hs[0] steps), not only the full management-horizon depth. Barrier
        # width scales as sqrt(horizon_bars), so per-step context/ratios are scaled by
        # sqrt(hs[i]/hs[-1]) relative to the management horizon; at i=h_idx the scale is 1.0,
        # so the management-horizon output is numerically identical to before this change.
        h_idx = len(hs) - 1  # management horizon is the last (largest) horizon
        flat_H = roll[:, :, h_idx, :].reshape(s * b, d)
        barrier_ratios_all = None
        if barrier_context is None:
            barrier_context_all = torch.stack([mae, mfe, vl], dim=-1)  # [S, B, |H|, 3]
        else:
            if barrier_context.ndim != 2 or barrier_context.shape != (b, 3):
                raise ValueError(
                    f"barrier_context must be [B, 3]; got {tuple(barrier_context.shape)}"
                )
            sigma_t = barrier_context[:, 0].to(device=roll.device, dtype=roll.dtype).clamp_min(1e-6)
            stop_t = barrier_context[:, 1].to(device=roll.device, dtype=roll.dtype).abs().clamp_min(1e-6)
            target_t = barrier_context[:, 2].to(device=roll.device, dtype=roll.dtype).clamp_min(1e-6)
            horizon_scale = torch.tensor(
                [float(h) for h in hs], device=roll.device, dtype=roll.dtype
            )
            horizon_scale = (horizon_scale / horizon_scale[-1]).sqrt()  # [|H|]; scale[h_idx] == 1.0
            stop_scaled = stop_t.unsqueeze(-1) * horizon_scale.unsqueeze(0)      # [B, |H|]
            target_scaled = target_t.unsqueeze(-1) * horizon_scale.unsqueeze(0)  # [B, |H|]
            sigma_scaled = sigma_t.unsqueeze(-1) * horizon_scale.unsqueeze(0)    # [B, |H|]
            barrier_context_all = torch.stack(
                [
                    mae / stop_scaled.unsqueeze(0),
                    mfe / target_scaled.unsqueeze(0),
                    vl / sigma_scaled.unsqueeze(0),
                ],
                dim=-1,
            )  # [S, B, |H|, 3]
            barrier_ratios_all = barrier_context_all
        flat_ctx = barrier_context_all.reshape(s * b * h_len, 3)
        if self._barrier_mode == "derived" and barrier_ratios_all is not None:
            barrier_logits_all = self.excursion_barrier_head(flat_ctx).view(s, b, h_len, -1)
        else:
            barrier_logits_all = self.barrier_head(flat, context=flat_ctx).view(s, b, h_len, -1)
        barrier_logits_flat = barrier_logits_all[:, :, h_idx, :]  # [S, B, 3] — management horizon
        regime_logits_flat = self.regime_head(flat_H).view(s, b, -1)    # [S, B, R]

        # Ensemble mean and spread
        # Return quantiles are combined via mixture-quantile pooling, not a naive per-level
        # average — averaging same-level quantile VALUES across members that disagree about
        # central tendency estimates "the average member's quantile," not "the mixture's
        # quantile," and structurally discards epistemic (between-member) spread. Diagnostic
        # (2026-07-05): naive averaging produced ~40-58% empirical coverage at every nominal
        # level regardless of target; mixture pooling is required to recover real coverage.
        rq_mean = combine_ensemble_quantiles(rq, self._quantile_levels)  # [B, |H|, Q]
        vl_mean = vl.mean(dim=0)           # [B, |H|]
        mae_mean = mae.mean(dim=0)         # [B, |H|]
        mfe_mean = mfe.mean(dim=0)         # [B, |H|]
        # unbiased=False: with the default N/A-unbiased std, n_samples=1 gives 0/0 = NaN
        # (review finding H11) — a NaN never satisfies a `> threshold` safety check, so
        # the exact moment uncertainty estimation collapses to a single sample is the
        # moment every epistemic-gated safety mechanism silently stops firing. Biased
        # std is 0.0 at N=1 (the correct "no observed spread yet" value), not NaN.
        epistemic = roll.std(dim=0, unbiased=False).mean(dim=-1)  # [B, |H|]
        aleatoric = (rq[:, :, :, -1] - rq[:, :, :, 0]).mean(dim=0)  # [B, |H|] quantile width

        # Predictive class probabilities must be aggregated in probability space; softmax(mean(logits))
        # is systematically overconfident for an ensemble.
        barrier_probs = torch.softmax(barrier_logits_flat, dim=-1).mean(dim=0)  # [B, 3]
        barrier_logits = barrier_probs.clamp_min(1e-8).log()                    # [B, 3]
        regime_probs = torch.softmax(regime_logits_flat, dim=-1).mean(dim=0)    # [B, R]
        regime_logits = regime_probs.clamp_min(1e-8).log()                      # [B, R]

        barrier_logits_intermediate: Tensor | None = None
        if h_idx > 0:
            barrier_probs_intermediate = torch.softmax(
                barrier_logits_all[:, :, :h_idx, :], dim=-1
            ).mean(dim=0)  # [B, |H|-1, 3]
            barrier_logits_intermediate = barrier_probs_intermediate.clamp_min(1e-8).log()

        return {
            "horizons": hs,
            "state": state,
            "rollout": roll,
            "return_quantiles": rq_mean,    # [B, |H|, Q]
            "volatility": vl_mean,          # [B, |H|]
            "mae": mae_mean,                # [B, |H|]
            "mfe": mfe_mean,                # [B, |H|]
            "barrier_logits": barrier_logits,  # [B, 3] log predictive probs at management horizon
            "barrier_logits_intermediate": barrier_logits_intermediate,  # [B, |H|-1, 3] or None
            "barrier_probs": barrier_probs,    # [B, 3]; BARRIER_CLASSES order: stop,target,neither
            "regime_logits": regime_logits,    # [B, R] log predictive probs
            "epistemic": epistemic,         # [B, |H|]
            "aleatoric": aleatoric,         # [B, |H|]
            "ood_score": ood_score,         # [B]
        }


__all__ = ["MarketWorld"]
