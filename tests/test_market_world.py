"""Market World: RSSM filter/imagine + head decoding (SPEC.md §14, §27).

MarketWorld takes a TRAINED RSSM. For unit tests we use a small, untrained RSSM.
forward() input is window_e: [T, B, embed_dim] (not raw latents [B, d]).
Direction head is REMOVED; no direction_logits in output.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from helion_risk_world.worlds.market_world import MarketWorld  # noqa: E402
from helion_risk_world.worlds.rollout_engine import RolloutEngine  # noqa: E402
from helion_risk_world.worlds.rssm import RSSM, RSSMState  # noqa: E402

STOCH, DETER, EMBED = 8, 16, 16   # small dims for fast tests


def _rssm() -> RSSM:
    return RSSM(stoch_dim=STOCH, deter_dim=DETER, embed_dim=EMBED)


def _market_world(horizons: tuple[int, ...] = (1, 3, 6)) -> MarketWorld:
    return MarketWorld(_rssm(), n_quantiles=5, horizons=horizons, n_samples=8)


def test_rssm_step_prior_and_posterior_shapes() -> None:
    rssm = _rssm()
    state = rssm.initial_state(4)
    # Step prior
    new_state, prior_dist = rssm.step_prior(state)
    assert new_state.h.shape == (4, DETER) and new_state.z.shape == (4, STOCH)
    assert prior_dist.loc.shape == (4, STOCH)
    # Step posterior
    e = torch.randn(4, EMBED)
    new_state2, post_dist, prior_dist2 = rssm.step_posterior(state, e)
    assert new_state2.h.shape == (4, DETER)


def test_rssm_deterministic_mode_matches_dist_mean_and_is_reproducible() -> None:
    """Review finding M3: deterministic=True must use the distribution mean (not
    an rsample() draw) so repeated calls with identical inputs give identical
    results — useful for reproducible eval/backtest runs."""
    rssm = _rssm()
    state = rssm.initial_state(4)

    new_state_a, prior_dist = rssm.step_prior(state, deterministic=True)
    new_state_b, _ = rssm.step_prior(state, deterministic=True)
    assert torch.equal(new_state_a.z, prior_dist.mean)
    assert torch.equal(new_state_a.z, new_state_b.z)

    e = torch.randn(4, EMBED)
    post_state_a, post_dist, _ = rssm.step_posterior(state, e, deterministic=True)
    post_state_b, _, _ = rssm.step_posterior(state, e, deterministic=True)
    assert torch.equal(post_state_a.z, post_dist.mean)
    assert torch.equal(post_state_a.z, post_state_b.z)


def test_rssm_filter_rolls_window() -> None:
    rssm = _rssm()
    T, B = 10, 3
    window_e = torch.randn(T, B, EMBED)
    state = rssm.filter(window_e)
    assert state.h.shape == (B, DETER) and state.z.shape == (B, STOCH)


def test_rollout_shape_and_spread() -> None:
    rssm = _rssm()
    B = 3
    state = RSSMState(h=torch.zeros(B, DETER), z=torch.zeros(B, STOCH))
    engine = RolloutEngine(rssm)
    roll = engine.rollout(state, horizons=[1, 3, 6], n_samples=8)
    assert roll.shape == (8, 3, 3, DETER + STOCH)  # [S, B, |H|, deter+stoch]
    assert roll.std(dim=0).mean() > 0               # ensemble has real spread


def test_market_world_output_shapes() -> None:
    mw = _market_world(horizons=(1, 3, 6))
    window_e = torch.randn(5, 4, EMBED)             # [T=5, B=4, embed]
    out = mw(window_e, n_samples=12)
    assert out["return_quantiles"].shape == (4, 3, 5)   # [B, |H|, Q]
    assert out["volatility"].shape == (4, 3)            # [B, |H|]
    assert out["mae"].shape == (4, 3)                   # [B, |H|]
    assert out["mfe"].shape == (4, 3)                   # [B, |H|]
    assert out["barrier_probs"].shape == (4, 3)         # [B, 3]  (at H=max)
    assert out["regime_logits"].shape[0] == 4           # [B, R]
    epi = out["epistemic"]
    assert epi.shape == (4, 3) and bool((epi > 0).all())  # calibrated spread
    # No direction_logits in new MarketWorld
    assert "direction_logits" not in out


def test_market_world_epistemic_finite_at_n_samples_1() -> None:
    """Review finding H11: unbiased std at n_samples=1 is 0/0 = NaN, and a NaN never
    satisfies a `> threshold` safety check downstream — so epistemic must stay finite
    (and specifically 0.0, i.e. "no observed spread yet") even with a single sample."""
    mw = _market_world(horizons=(1, 3, 6))
    window_e = torch.randn(5, 4, EMBED)
    out = mw(window_e, n_samples=1)
    epi = out["epistemic"]
    assert bool(torch.isfinite(epi).all())
    assert bool((epi == 0).all())


def test_market_world_quantiles_non_decreasing() -> None:
    mw = _market_world(horizons=(1, 3))
    out = mw(torch.randn(3, 2, EMBED), n_samples=4)
    q = out["return_quantiles"]
    assert bool((q[..., 1:] - q[..., :-1] >= -1e-5).all())


def test_market_world_ood_from_prior_nll() -> None:
    """OOD score is the prior NLL (negative log-prob), a real scalar per batch element."""
    mw = _market_world(horizons=(3, 6))
    out = mw(torch.randn(2, 4, EMBED), n_samples=4)
    ood = out["ood_score"]
    assert ood.shape == (4,) and bool(torch.isfinite(ood).all())


def test_market_world_gradients_flow_through_filter() -> None:
    """filter() (posterior path) must be differentiable — used during RSSM training."""
    rssm = _rssm()
    mw = MarketWorld(rssm, horizons=(1,), n_samples=2)
    window_e = torch.randn(3, 2, EMBED, requires_grad=True)   # T=3, B=2
    state = mw.filter(window_e)
    state.z.sum().backward()
    assert window_e.grad is not None   # gradient flows from z back through filter
    assert mw.rssm.gru.weight_ih.grad is not None


def test_market_world_ensemble_class_probs_are_averaged_in_probability_space() -> None:
    mw = _market_world(horizons=(1,))
    mw.set_barrier_mode("legacy")
    head_dim = mw.rssm.deter_dim + mw.rssm.stoch_dim

    class _BarrierStub(torch.nn.Module):
        def forward(self, z, context=None):
            return z[:, :3]

    class _RegimeStub(torch.nn.Module):
        def forward(self, z):
            return z[:, :2]

    mw.barrier_head = _BarrierStub()
    mw.regime_head = _RegimeStub()
    mw.filter = lambda window_e, state=None, **kw: RSSMState(  # type: ignore[method-assign]
        h=torch.zeros(1, mw.rssm.deter_dim),
        z=torch.zeros(1, mw.rssm.stoch_dim),
    )

    roll = torch.zeros(2, 1, 1, head_dim)
    roll[0, 0, 0, :3] = torch.tensor([4.0, 0.0, 0.0])
    roll[1, 0, 0, :3] = torch.tensor([0.0, 4.0, 0.0])
    roll[0, 0, 0, 3:5] = torch.tensor([3.0, 0.0])
    roll[1, 0, 0, 3:5] = torch.tensor([0.0, 3.0])
    mw.imagine = lambda state, horizons=None, n_samples=None, **kw: roll  # type: ignore[method-assign]

    out = mw(torch.randn(2, 1, EMBED), n_samples=2)
    expected_barrier = 0.5 * (
        torch.softmax(torch.tensor([4.0, 0.0, 0.0]), dim=0)
        + torch.softmax(torch.tensor([0.0, 4.0, 0.0]), dim=0)
    )

    assert torch.allclose(out["barrier_probs"][0], expected_barrier, atol=1e-6)
    assert torch.allclose(torch.softmax(out["barrier_logits"], dim=-1)[0], expected_barrier, atol=1e-6)


def test_market_world_barrier_logits_intermediate_shape_multi_horizon() -> None:
    """Phase 5b: deep supervision — barrier logits at every non-management horizon too."""
    mw = _market_world(horizons=(1, 3, 6))
    window_e = torch.randn(5, 4, EMBED)
    out = mw(window_e, n_samples=8)
    intermediate = out["barrier_logits_intermediate"]
    assert intermediate is not None
    assert intermediate.shape == (4, 2, 3)  # [B, |H|-1, 3]


def test_market_world_barrier_logits_intermediate_none_for_single_horizon() -> None:
    """Single-horizon configs (used elsewhere in this file) must keep working unchanged."""
    mw = _market_world(horizons=(1,))
    out = mw(torch.randn(3, 2, EMBED), n_samples=4)
    assert out["barrier_logits_intermediate"] is None


def test_market_world_barrier_logits_intermediate_uses_scaled_barrier_context() -> None:
    """With a real barrier_context, intermediate-horizon ratios must differ from the raw
    (unscaled) ensemble-mean mae/mfe/vol — i.e. the sqrt(h_i/H) scaling actually applies,
    not just a slice of the same numbers reused across horizons."""
    mw = _market_world(horizons=(1, 3, 6))
    window_e = torch.randn(5, 4, EMBED)
    barrier_context = torch.tensor([[0.01, -0.02, 0.02]] * 4, dtype=torch.float32)
    out = mw(window_e, barrier_context=barrier_context, n_samples=8)
    intermediate = out["barrier_logits_intermediate"]
    assert intermediate is not None
    assert intermediate.shape == (4, 2, 3)
    assert bool(torch.isfinite(intermediate).all())
    # The two intermediate horizons (1 and 3 bars) must get different scaling than each
    # other (sqrt(1/6) != sqrt(3/6)), so their logits should not be identical.
    assert not torch.allclose(intermediate[:, 0, :], intermediate[:, 1, :])
