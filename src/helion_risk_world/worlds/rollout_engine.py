"""Stochastic latent rollout using the TRAINED RSSM prior (SPEC.md §14.3, Appendix A).

Rolls the RSSM forward ``n_samples`` times by sampling the TRAINED prior p_θ(z|h).
Because the prior is trained (KL to posterior), the ensemble spread is CALIBRATED
epistemic uncertainty — NOT arbitrary initialisation noise from torch.randn.

SRP: orchestration only — RSSM architecture lives in worlds/rssm.py.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from helion_risk_world.worlds.rssm import RSSM, RSSMState


class RolloutEngine:
    """Roll the RSSM prior forward into a calibrated ensemble of futures states."""

    def __init__(self, rssm: RSSM) -> None:
        self._rssm = rssm

    def rollout(
        self, state: RSSMState, horizons: Sequence[int], n_samples: int, *, deterministic: bool = False
    ) -> Tensor:
        """state: (h [B,deter], z [B,stoch]) → ensemble [S, B, |H|, deter+stoch].

        Samples the LEARNED prior n_samples times, collecting the full state
        (h, z concatenated) at each requested horizon.

        deterministic (review finding M3): use the prior's mean at every step
        instead of sampling — for reproducible eval/backtest runs. Note this
        collapses the ensemble spread (epistemic uncertainty) to ~0 across
        samples, since every sample becomes identical; only meaningful with
        n_samples=1, or when epistemic isn't needed for that call.
        """
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        steps = sorted(set(horizons))
        if not steps or steps[0] < 1:
            raise ValueError("horizons must be positive integers")

        B = state.h.shape[0]
        # Expand to [S*B] for vectorised sampling of S independent rollouts
        h = state.h.unsqueeze(0).expand(n_samples, B, -1).reshape(n_samples * B, -1)
        z = state.z.unsqueeze(0).expand(n_samples, B, -1).reshape(n_samples * B, -1)
        cur = RSSMState(h=h, z=z)

        collected: dict[int, Tensor] = {}
        for step in range(1, steps[-1] + 1):
            cur, _ = self._rssm.step_prior(cur, deterministic=deterministic)   # TRAINED prior, NOT randn
            if step in steps:
                # Concatenate h and z into the full state representation
                full_state = torch.cat([cur.h, cur.z], dim=-1)  # [S*B, deter+stoch]
                collected[step] = full_state.view(n_samples, B, -1)

        # [S, B, |H|, deter+stoch]
        return torch.stack([collected[h] for h in steps], dim=2)

    def rollout_decoded(
        self, state: RSSMState, horizons: Sequence[int], n_samples: int, *, deterministic: bool = False
    ) -> Tensor:
        """Like rollout but decode each step to the representation space.

        Returns [S, B, |H|, embed_dim] — the predicted encoded states ê_{t+k}.
        """
        steps = sorted(set(horizons))
        B = state.h.shape[0]
        h = state.h.unsqueeze(0).expand(n_samples, B, -1).reshape(n_samples * B, -1)
        z = state.z.unsqueeze(0).expand(n_samples, B, -1).reshape(n_samples * B, -1)
        cur = RSSMState(h=h, z=z)

        collected: dict[int, Tensor] = {}
        for step in range(1, steps[-1] + 1):
            cur, _ = self._rssm.step_prior(cur, deterministic=deterministic)
            if step in steps:
                e_hat = self._rssm.decode(cur.h, cur.z)  # [S*B, embed]
                collected[step] = e_hat.view(n_samples, B, -1)

        return torch.stack([collected[h] for h in steps], dim=2)  # [S, B, |H|, embed]


__all__ = ["RolloutEngine"]
