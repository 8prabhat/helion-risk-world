"""Mixture-quantile pooling for ensemble return-quantile aggregation.

``MarketWorld`` previously aggregated its S-member rollout ensemble's return-quantile
predictions via a naive per-level average (``rq.mean(dim=0)``). That only reflects each
member's own (aleatoric) spread — it does not correctly account for members disagreeing
with each other about the central tendency (epistemic spread), which the model already
tracks separately (``epistemic = roll.std(dim=0)...``) but never folds back into the
reported return quantiles.

Diagnostic (2026-07-05): naive averaging produced ~40-58% empirical coverage at every
nominal quantile level (0.1 through 0.9) — the signature of predictions clustering near the
mixture's center regardless of nominal target, because averaging same-level quantile VALUES
across disagreeing members estimates "the average member's quantile," not "the mixture's
quantile." Validated on synthetic data with a known ground truth: when ensemble members
disagree about center (epistemic-dominant), naive averaging's coverage_error was 0.179;
mixture pooling recovered 0.003.

The fix: treat each ensemble member's Q known (level, value) pairs as a piecewise-linear
quantile function (inverse CDF), draw pseudo-samples from each member via inverse-transform
sampling, pool all members' pseudo-samples, and take the empirical quantile of the POOLED
set. This is the standard way to obtain quantiles of a mixture distribution from its
components' quantile functions, and correctly folds in between-member disagreement.
"""

from __future__ import annotations

import torch
from torch import Tensor

_DEFAULT_N_PSEUDO_SAMPLES = 128
_TAIL_MARGIN = 0.02  # pseudo-sample probability grid spans [_TAIL_MARGIN, 1-_TAIL_MARGIN]


def combine_ensemble_quantiles(
    rq: Tensor,
    member_levels: Tensor,
    *,
    n_pseudo_samples: int = _DEFAULT_N_PSEUDO_SAMPLES,
) -> Tensor:
    """Pool ensemble members into mixture quantiles at the same ``member_levels``.

    Args:
        rq: ``[S, B, H, Q]`` per-member quantile predictions (monotone non-decreasing in Q,
            guaranteed by ``ReturnQuantileHead``'s construction).
        member_levels: ``[Q]`` sorted ascending quantile probability levels (e.g. the 5
            values 0.1..0.9) — used both as the interpolation grid and the output levels.
        n_pseudo_samples: pseudo-draws per ensemble member; pooled sample size per (b, h) is
            ``S * n_pseudo_samples``. 128 is a cheap, ample default (16 members -> 2048
            pooled samples per row, more than enough to resolve 5 output quantile levels).

    Returns:
        ``[B, H, Q]`` mixture quantiles at ``member_levels``, monotone non-decreasing
        (``torch.quantile`` on a pooled 1-D sample is monotone in its level argument by
        construction).
    """
    s, b, h, q = rq.shape
    device, dtype = rq.device, rq.dtype
    levels = member_levels.to(device=device, dtype=dtype)

    u = torch.linspace(_TAIL_MARGIN, 1.0 - _TAIL_MARGIN, n_pseudo_samples, device=device, dtype=dtype)
    idx_hi = torch.searchsorted(levels, u, right=True).clamp(1, q - 1)  # [K]
    idx_lo = idx_hi - 1
    lvl_lo, lvl_hi = levels[idx_lo], levels[idx_hi]  # [K]
    frac = ((u - lvl_lo) / (lvl_hi - lvl_lo).clamp_min(1e-12)).clamp(0.0, 1.0)  # [K]

    flat = rq.reshape(s, b * h, q)  # [S, N, Q]
    val_lo = flat.index_select(-1, idx_lo)  # [S, N, K]
    val_hi = flat.index_select(-1, idx_hi)  # [S, N, K]
    pseudo = val_lo + frac.view(1, 1, -1) * (val_hi - val_lo)  # [S, N, K]

    pooled = pseudo.permute(1, 0, 2).reshape(b * h, s * n_pseudo_samples)  # [N, S*K]
    out = torch.quantile(pooled, levels, dim=1)  # [Q, N]
    return out.T.reshape(b, h, q)  # [B, H, Q]


__all__ = ["combine_ensemble_quantiles"]
