"""RSSMLoss normalization (review finding M9)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch.distributions import Normal  # noqa: E402

from helion_risk_world.losses.rssm_loss import RSSMLoss  # noqa: E402
from helion_risk_world.worlds.rssm import RSSM  # noqa: E402

STOCH, DETER, EMBED = 4, 8, 6


def _rssm() -> RSSM:
    return RSSM(stoch_dim=STOCH, deter_dim=DETER, embed_dim=EMBED)


def test_rssm_loss_scale_does_not_grow_with_sequence_length(monkeypatch: pytest.MonkeyPatch) -> None:
    """Before this fix, l_dyn/l_imag were unnormalized sums over T and (t, k)
    respectively, so loss magnitude (and therefore the effective learning-rate
    scale) grew with sequence length and imagination depth. A 4x longer
    sequence must not produce ~4x the loss."""
    monkeypatch.setattr(Normal, "rsample", lambda self: self.mean)  # deterministic
    torch.manual_seed(0)
    rssm = _rssm()
    loss_fn = RSSMLoss(rssm, K=3)

    torch.manual_seed(0)
    out_short = loss_fn(torch.randn(5, 2, EMBED))

    torch.manual_seed(0)
    out_long = loss_fn(torch.randn(20, 2, EMBED))

    assert out_long["l_dyn"].item() < out_short["l_dyn"].item() * 3
    assert out_long["l_imag"].item() < out_short["l_imag"].item() * 3


def test_l_imag_is_a_weighted_average_not_a_sum(monkeypatch: pytest.MonkeyPatch) -> None:
    """With K=1 every valid (t, k=1) term shares the same weight (gamma**1), so
    the weighted average degenerates to the plain mean of per-position MSE
    terms — an exact, verifiable property distinct from an unnormalized sum
    (which would instead scale with the number of valid terms, T-1)."""
    monkeypatch.setattr(Normal, "rsample", lambda self: self.mean)
    torch.manual_seed(0)
    rssm = _rssm()
    loss_fn_k1 = RSSMLoss(rssm, K=1)

    torch.manual_seed(0)
    seq_e = torch.randn(6, 2, EMBED)
    out = loss_fn_k1(seq_e)

    # A larger T with the same per-step reconstruction difficulty must not
    # inflate l_imag if it's truly an average (K=1 keeps this comparison clean:
    # each additional t just contributes one more equally-weighted term).
    torch.manual_seed(0)
    rssm2 = _rssm()
    loss_fn_k1_b = RSSMLoss(rssm2, K=1)
    torch.manual_seed(0)
    seq_e_long = torch.randn(12, 2, EMBED)
    out_long = loss_fn_k1_b(seq_e_long)

    assert out_long["l_imag"].item() == pytest.approx(out["l_imag"].item(), rel=0.5)
