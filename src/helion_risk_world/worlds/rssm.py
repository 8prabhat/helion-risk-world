"""Recurrent State Space Model (RSSM) — the trained latent world model (SPEC.md §14).

Architecture (PlaNet/Dreamer-style, autonomous market dynamics):
  h_t = GRU(h_{t-1}, z_{t-1})                    deterministic carry
  p_θ(z_t | h_t)     = N(μ_p(h_t),  σ_p(h_t)²)  trained PRIOR (used at inference/imagine)
  q_φ(z_t | h_t,e_t) = N(μ_q([h_t,e_t]), σ_q²)  posterior (used during training)
  D_ψ(h_t, z_t) → ê_t                            JEPA representation head

The prior is trained (via KL to posterior) so ensemble spread from sampling the prior
is CALIBRATED epistemic uncertainty, not arbitrary init noise.  This is the key
property that makes it a world model rather than a noise generator.

``filter()`` rolls the RSSM over an observed window to obtain s_t = (h_t, z_t);
``worlds.rollout_engine.RolloutEngine`` then samples the TRAINED prior forward
from s_t using this class's ``step_prior()``.
SRP: RSSM dynamics only — heads, losses, and rollout scheduling live elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal


@dataclass
class RSSMState:
    """Deterministic + stochastic latent state s_t = (h_t, z_t)."""

    h: Tensor  # [B, deter]
    z: Tensor  # [B, stoch]


class _NormalHead(nn.Module):
    """Project a vector to a Normal distribution (μ, σ = softplus(raw) + min_std)."""

    def __init__(self, in_dim: int, out_dim: int, min_std: float = 0.1) -> None:
        super().__init__()
        self.fc_mu = nn.Linear(in_dim, out_dim)
        self.fc_log = nn.Linear(in_dim, out_dim)
        self._min_std = min_std

    def forward(self, x: Tensor) -> Normal:
        mu = self.fc_mu(x)
        std = torch.nn.functional.softplus(self.fc_log(x)) + self._min_std
        return Normal(mu, std)


class RSSM(nn.Module):
    """Autonomous RSSM for market dynamics.

    ``stoch_dim``  — dimension of the stochastic latent z_t
    ``deter_dim``  — dimension of the deterministic carry h_t (GRU hidden)
    ``embed_dim``  — dimension of the encoded observation e_t from the encoders
    """

    def __init__(
        self,
        stoch_dim: int = 32,
        deter_dim: int = 128,
        embed_dim: int = 128,
        min_prior_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim
        self.embed_dim = embed_dim

        # GRU: input = z_{t-1}, hidden = h_{t-1} → h_t
        self.gru = nn.GRUCell(stoch_dim, deter_dim)

        # Prior p_θ(z_t | h_t): projects h_t
        self.prior_head = _NormalHead(deter_dim, stoch_dim, min_std=min_prior_std)

        # Posterior q_φ(z_t | h_t, e_t): projects [h_t; e_t]
        self.posterior_head = _NormalHead(deter_dim + embed_dim, stoch_dim, min_std=min_prior_std)

        # Representation head D_ψ: [h_t, z_t] → ê_t (predicts encoded state, JEPA target)
        self.repr_head = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def prior(self, h: Tensor) -> Normal:
        """Trained prior p_θ(z_t | h_t). Used during imagination (inference)."""
        return self.prior_head(h)

    def posterior(self, h: Tensor, e: Tensor) -> Normal:
        """Posterior q_φ(z_t | h_t, e_t). Used during training (observation available)."""
        return self.posterior_head(torch.cat([h, e], dim=-1))

    def decode(self, h: Tensor, z: Tensor) -> Tensor:
        """D_ψ(h, z) → ê_t  [B, embed_dim]  (predicted encoded state)."""
        return self.repr_head(torch.cat([h, z], dim=-1))

    def initial_state(self, batch_size: int, device: torch.device | str = "cpu") -> RSSMState:
        """Zero-initialised state for the start of a sequence."""
        h = torch.zeros(batch_size, self.deter_dim, device=device)
        z = torch.zeros(batch_size, self.stoch_dim, device=device)
        return RSSMState(h=h, z=z)

    def step_prior(self, state: RSSMState, *, deterministic: bool = False) -> tuple[RSSMState, Normal]:
        """One step with the prior (no observation): (h_{t-1}, z_{t-1}) → (h_t, z_t~prior).

        deterministic (review finding M3): use the prior's mean instead of an
        ``rsample()`` draw, for reproducible eval/backtest runs — two calls with
        identical inputs otherwise yield different quantiles/barrier
        probabilities/OOD scores purely from resampling noise, with no way to
        turn that off. Off by default (unchanged training/inference behavior).
        """
        h = self.gru(state.z, state.h)
        dist = self.prior(h)
        z = dist.mean if deterministic else dist.rsample()
        return RSSMState(h=h, z=z), dist

    def step_posterior(
        self, state: RSSMState, e: Tensor, *, deterministic: bool = False
    ) -> tuple[RSSMState, Normal, Normal]:
        """One step with the posterior (observation available): returns (new_state, post, prior).

        deterministic: see step_prior — uses the posterior's mean instead of rsample().
        """
        h = self.gru(state.z, state.h)
        prior_dist = self.prior(h)
        post_dist = self.posterior(h, e)
        z = post_dist.mean if deterministic else post_dist.rsample()
        return RSSMState(h=h, z=z), post_dist, prior_dist

    def step(self, state: RSSMState, e: Tensor, *, deterministic: bool = False) -> RSSMState:
        """One incremental posterior update from a caller-supplied state (review
        finding H1): thin wrapper around step_posterior that drops the
        distributions, for callers that only need the new state (e.g. live/paper
        inference threading a persisted RSSMState across bars — see
        MarketWorld.filter's ``state`` parameter)."""
        new_state, _, _ = self.step_posterior(state, e, deterministic=deterministic)
        return new_state

    def filter(
        self, window_e: Tensor, state: RSSMState | None = None, *, deterministic: bool = False
    ) -> RSSMState:
        """Roll the RSSM over an observed window using posterior updates.

        window_e: [T, B, embed_dim]  (time first)
        state: starting state s_{t-1}; defaults to zero-initialised when None
               (the original training-time behavior: roll a full T-step window
               from scratch). Callers that need to preserve recurrent history
               across successive calls (T=1 live/paper inference — review
               finding H1) pass in the previous call's returned state instead.
        deterministic: see step_prior (review finding M3).
        Returns s_t = (h_t, z_t) — the inferred state at the END of the window.

        Inference requires rolling the recurrence over the full lookback because
        h_t depends on the entire history; a single-step posterior call starting
        from a zero state is wrong unless a real prior state is threaded in.
        """
        if window_e.ndim != 3:
            raise ValueError(f"window_e must be [T, B, embed]; got {tuple(window_e.shape)}")
        T, B, _ = window_e.shape
        cur = state if state is not None else self.initial_state(B, device=window_e.device)
        for t in range(T):
            cur = self.step(cur, window_e[t], deterministic=deterministic)
        return cur

__all__ = ["RSSM", "RSSMState"]
