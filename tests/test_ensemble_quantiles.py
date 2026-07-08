"""Mixture-quantile pooling for ensemble return-quantile aggregation (2026-07-05).

Naive per-level averaging (rq.mean(dim=0)) only reflects each ensemble member's own
(aleatoric) spread, not between-member (epistemic) disagreement about central tendency --
diagnosed from real calibration data showing ~40-58% empirical coverage at every nominal
quantile level regardless of target, the signature of members disagreeing about center.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from helion_risk_world.worlds.ensemble_quantiles import combine_ensemble_quantiles  # noqa: E402

LEVELS = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])


def test_combine_ensemble_quantiles_recovers_coverage_when_members_disagree() -> None:
    """Epistemic-dominant case: members disagree a lot about center, low own-spread.
    Naive averaging should under-cover badly (matches the real diagnostic); mixture
    pooling should recover close to nominal coverage."""
    torch.manual_seed(0)
    n_rows, s = 4000, 16
    epistemic_spread, aleatoric_spread = 0.02, 0.005

    member_centers = torch.randn(n_rows, s) * epistemic_spread  # [N, S]
    chosen = torch.randint(0, s, (n_rows,))
    true_outcome = member_centers[torch.arange(n_rows), chosen] + torch.randn(n_rows) * aleatoric_spread

    from scipy.stats import norm

    z = torch.tensor([norm.ppf(float(level)) for level in LEVELS], dtype=torch.float32)
    # [S, N, 1, Q] -> reshape to [S, N, H=1, Q]
    rq = (member_centers.T.unsqueeze(-1) + aleatoric_spread * z.view(1, 1, -1)).unsqueeze(2)
    assert rq.shape == (s, n_rows, 1, len(LEVELS))

    naive = rq.mean(dim=0).squeeze(1)  # [N, Q]
    mixture = combine_ensemble_quantiles(rq, LEVELS).squeeze(1)  # [N, Q]

    nominal = LEVELS
    naive_coverage = (true_outcome.unsqueeze(-1) <= naive).float().mean(dim=0)
    mixture_coverage = (true_outcome.unsqueeze(-1) <= mixture).float().mean(dim=0)

    naive_error = (naive_coverage - nominal).abs().mean().item()
    mixture_error = (mixture_coverage - nominal).abs().mean().item()

    assert naive_error > 0.10  # reproduces the real under-coverage failure mode
    assert mixture_error < 0.03  # mixture pooling recovers close to nominal coverage
    assert mixture_error < naive_error / 5  # a large, decisive improvement


def test_combine_ensemble_quantiles_matches_naive_when_no_epistemic_disagreement() -> None:
    """Sanity check: with no between-member disagreement, mixture pooling should not make
    things meaningfully worse than the naive average -- both estimate the same thing."""
    torch.manual_seed(1)
    n_rows, s = 4000, 16
    aleatoric_spread = 0.02

    from scipy.stats import norm

    z = torch.tensor([norm.ppf(float(level)) for level in LEVELS], dtype=torch.float32)
    true_outcome = torch.randn(n_rows) * aleatoric_spread
    rq = (aleatoric_spread * z.view(1, 1, 1, -1)).expand(s, n_rows, 1, len(LEVELS)).clone()

    naive = rq.mean(dim=0).squeeze(1)
    mixture = combine_ensemble_quantiles(rq, LEVELS).squeeze(1)

    nominal = LEVELS
    naive_coverage = (true_outcome.unsqueeze(-1) <= naive).float().mean(dim=0)
    mixture_coverage = (true_outcome.unsqueeze(-1) <= mixture).float().mean(dim=0)

    naive_error = (naive_coverage - nominal).abs().mean().item()
    mixture_error = (mixture_coverage - nominal).abs().mean().item()
    assert abs(naive_error - mixture_error) < 0.03


def test_combine_ensemble_quantiles_output_is_monotone_and_shape_preserving() -> None:
    torch.manual_seed(2)
    s, b, h, q = 8, 3, 6, 5
    rq = torch.cumsum(torch.rand(s, b, h, q).abs() + 0.01, dim=-1)  # monotone per member
    out = combine_ensemble_quantiles(rq, LEVELS)
    assert out.shape == (b, h, q)
    assert bool((out[..., 1:] - out[..., :-1] >= -1e-5).all())
