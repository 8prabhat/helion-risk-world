"""RSSM dynamics training loss: L_dyn + L_imag (SPEC.md §14.2, §22).

L_dyn  = α · ‖D_ψ(s_t) − sg(e_t)‖²  +  β · KL_balanced(post ‖ prior)   (per step)
L_imag = Σ_k γ^k · ‖D_ψ(roll_k(s_t)) − sg(e_{t+k})‖²                   (multi-step)

KL balancing (Dreamer v2): mix straight-through and stop-gradient sides so neither the
prior nor the posterior is pushed exclusively.  free_bits clips the KL per dimension to
avoid posterior collapse on near-zero-mean, low-SNR market returns.

SRP: loss computation only — RSSM architecture lives in worlds/rssm.py.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal, kl_divergence

from helion_risk_world.worlds.rssm import RSSM, RSSMState


def kl_balanced(
    post: Normal,
    prior: Normal,
    free_bits: float = 1.0,
    balance: float = 0.8,
) -> Tensor:
    """KL with free-bits clipping and Dreamer-style balancing.

    balance=0.8 means 80 % of the gradient goes to the prior (trains the prior harder),
    20 % to the posterior (avoids posterior collapse).
    Returns a scalar (mean over batch and stoch dimensions).
    """
    kl_post_sg_prior = kl_divergence(post, Normal(prior.loc.detach(), prior.scale.detach()))
    kl_prior_sg_post = kl_divergence(Normal(post.loc.detach(), post.scale.detach()), prior)
    # free_bits: clip per-dim KL to avoid degenerate zero-KL posteriors
    kl_post_clipped = torch.clamp(kl_post_sg_prior, min=free_bits)
    kl_prior_clipped = torch.clamp(kl_prior_sg_post, min=free_bits)
    return (balance * kl_prior_clipped + (1 - balance) * kl_post_clipped).mean()


class RSSMLoss:
    """Compute L_dyn + L_imag for one training sequence (SPEC.md §14.2).

    seq_e: [T, B, embed_dim]  — pre-computed encoded observations from the frozen/slow encoder
    Returns a dict of scalar loss tensors so callers can log components separately.
    """

    def __init__(
        self,
        rssm: RSSM,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.95,
        K: int = 12,
        free_bits: float = 1.0,
        kl_balance: float = 0.8,
    ) -> None:
        if K < 1:
            raise ValueError("K must be >= 1 (imagination depth)")
        self._rssm = rssm
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._K = K
        self._free_bits = free_bits
        self._kl_balance = kl_balance

    def __call__(self, seq_e: Tensor) -> dict[str, Tensor]:
        """seq_e: [T, B, embed_dim].  Returns {l_dyn, l_imag, loss}.

        Review finding M9: both terms are now normalized (mean, not sum) so loss
        magnitude — and therefore the effective learning-rate/grad-clip scale —
        doesn't grow with sequence length T or imagination depth K. l_imag's
        normalization additionally fixes a secondary bias: earlier positions in a
        sequence have strictly more valid (t, k) targets than later ones (fewer
        bars remain to imagine into), so an unweighted sum implicitly overweights
        early-sequence positions for longer sequences; dividing by the actual
        accumulated weight corrects this regardless of where in the sequence a
        position sits.
        """
        if seq_e.ndim != 3:
            raise ValueError(f"seq_e must be [T, B, embed]; got {tuple(seq_e.shape)}")
        T, B, _ = seq_e.shape
        rssm = self._rssm

        state = rssm.initial_state(B, device=seq_e.device)
        l_dyn = seq_e.new_zeros(1)
        states: list[tuple[RSSMState, int]] = []

        for t in range(T):
            state, post, prior = rssm.step_posterior(state, seq_e[t])
            e_hat = rssm.decode(state.h, state.z)
            l_dyn = l_dyn \
                + self._alpha * F.mse_loss(e_hat, seq_e[t].detach()) \
                + self._beta * kl_balanced(post, prior, self._free_bits, self._kl_balance)
            # detach: imagination starts from fixed starting points (Dreamer v2 convention)
            states.append((RSSMState(h=state.h.detach(), z=state.z.detach()), t))
        l_dyn = l_dyn / T

        l_imag = seq_e.new_zeros(1)
        l_imag_weight = 0.0
        for base_state, t in states:
            h, z = base_state.h, base_state.z
            cur = RSSMState(h=h, z=z)
            for k in range(1, self._K + 1):
                cur, _ = rssm.step_prior(cur)
                t_k = t + k
                if t_k < T:
                    e_hat_k = rssm.decode(cur.h, cur.z)
                    w = self._gamma ** k
                    l_imag = l_imag + w * F.mse_loss(e_hat_k, seq_e[t_k].detach())
                    l_imag_weight += w
        l_imag = l_imag / max(l_imag_weight, 1e-8)

        loss = l_dyn + l_imag
        return {"l_dyn": l_dyn, "l_imag": l_imag, "loss": loss}


__all__ = ["RSSMLoss", "kl_balanced"]
