"""World-model evaluation metrics (SPEC.md §21 Stage 3, §26).

``compute()`` — rollout MAE/RMSE, latent consistency, RSSM-specific metrics
               (prior predictive coverage, KL collapse detection).

RSSM-specific kwargs:
  kl_per_step:     list of per-step KL scalars from training — tracks collapse.
  prior_samples:   [S, B] ensemble samples from the TRAINED prior.
  posterior_mean:  [B]    posterior mean latent for the same batch.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute(*args: Any, **kwargs: Any) -> dict[str, float]:
    """World-model evaluation: rollout accuracy + RSSM latent diagnostics.

    Keyword args:
        predicted:       [N] or [N, d]  model rollout predictions.
        target:          [N] or [N, d]  realized targets.
        epistemic:       [N]  RSSM ensemble spread values.
        kl_per_step:     list[float]    per-step KL values from an epoch of RSSM training.
        prior_samples:   [S, B]         ensemble draws from the TRAINED prior.
        posterior_mean:  [B]            posterior mean latent for coverage check.

    Returns dict with any subset of:
        rollout_mae, rollout_rmse, latent_consistency, epistemic_mean,
        mean_kl, kl_collapse_frac, prior_coverage.
    """
    predicted = np.asarray(kwargs.get("predicted", []), dtype=float)
    target = np.asarray(kwargs.get("target", []), dtype=float)
    out: dict[str, float] = {}

    # Rollout accuracy
    if predicted.size and target.size:
        diff = predicted - target
        out["rollout_mae"] = float(np.abs(diff).mean())
        out["rollout_rmse"] = float(np.sqrt(np.square(diff).mean()))
        out["latent_consistency"] = float(np.square(diff).mean())

    # Ensemble spread
    epistemic = kwargs.get("epistemic")
    if epistemic is not None:
        out["epistemic_mean"] = float(np.asarray(epistemic, dtype=float).mean())

    # RSSM KL diagnostics — track posterior collapse (KL → 0 → prior not learning)
    kl_vals = kwargs.get("kl_per_step")
    if kl_vals is not None:
        kl_arr = np.asarray(kl_vals, dtype=float)
        out["mean_kl"] = float(kl_arr.mean())
        # fraction of steps where KL is below 0.01 (near-collapse)
        out["kl_collapse_frac"] = float(np.mean(kl_arr < 0.01))

    # Prior predictive coverage: does the prior ensemble span the posterior mean?
    # A well-trained prior should cover posterior_mean with high probability.
    prior_samples = kwargs.get("prior_samples")   # [S, B]
    posterior_mean = kwargs.get("posterior_mean")  # [B]
    if prior_samples is not None and posterior_mean is not None:
        ps = np.asarray(prior_samples, dtype=float)
        pm = np.asarray(posterior_mean, dtype=float).reshape(-1)
        # coverage = fraction of batch elements where posterior_mean is within the prior range
        coverage = float(np.mean(
            (ps.min(axis=0) <= pm) & (pm <= ps.max(axis=0))
        ))
        out["prior_coverage"] = coverage

    return out
