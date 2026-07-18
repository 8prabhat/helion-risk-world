"""round_trip_cost_frac (feature/label overhaul Phase 1): derives a barrier-labeling
cost floor from this project's OWN documented cost assumptions, not a borrowed constant.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from helion_risk_world.config.execution_config import CostModelConfig
from helion_risk_world.execution.cost_model import overnight_financing_cost, round_trip_cost_frac


def test_default_cost_frac_in_sane_band() -> None:
    frac = round_trip_cost_frac(CostModelConfig())
    # A round-trip cost in the tens-of-bps range is plausible for Indian futures;
    # anything near zero or near 1.0 would indicate a formula error.
    assert 0.0005 < frac < 0.02


def test_scales_with_gst_rate() -> None:
    base = round_trip_cost_frac(CostModelConfig())
    higher_gst = round_trip_cost_frac(replace(CostModelConfig(), gst_rate=0.36))
    assert higher_gst > base


def test_scales_with_half_spread() -> None:
    base = round_trip_cost_frac(CostModelConfig())
    wider_spread = round_trip_cost_frac(replace(CostModelConfig(), half_spread_bps=0.003))
    assert wider_spread > base


def test_stt_and_stamp_duty_are_one_sided_not_doubled() -> None:
    """Cost-model audit (2026-07-18): STT (sell-side) and stamp duty (buy-side) are each
    real one-time charges per round trip, not symmetric per-leg charges like brokerage/
    exchange/GST/SEBI/spread/slippage. Removing them entirely must drop the total by
    exactly their combined rate (added once), not twice."""
    cfg = CostModelConfig()
    frac = round_trip_cost_frac(cfg, reference_notional=1_500_000.0)
    without_stt_stamp = round_trip_cost_frac(
        replace(cfg, stt_rate=0.0, stamp_duty_rate=0.0), reference_notional=1_500_000.0
    )
    assert frac - without_stt_stamp == pytest.approx(cfg.stt_rate + cfg.stamp_duty_rate)


def test_reference_notional_has_minor_sensitivity() -> None:
    """Brokerage is a small flat fee; changing the reference notional by 2x should
    change the total by much less than 2x (i.e. the rate-based terms dominate)."""
    small_notional = round_trip_cost_frac(CostModelConfig(), reference_notional=750_000.0)
    large_notional = round_trip_cost_frac(CostModelConfig(), reference_notional=1_500_000.0)
    assert small_notional > large_notional  # smaller notional -> brokerage is a bigger fraction
    ratio = small_notional / large_notional
    assert ratio < 1.2  # not anywhere close to the 2x notional change


def test_overnight_financing_cost_zero_for_intraday() -> None:
    cfg = CostModelConfig()
    assert overnight_financing_cost(cfg, notional=1_500_000.0, nights_held=0) == 0.0


def test_overnight_financing_cost_scales_linearly_with_nights() -> None:
    cfg = CostModelConfig()
    one_night = overnight_financing_cost(cfg, notional=1_500_000.0, nights_held=1)
    three_nights = overnight_financing_cost(cfg, notional=1_500_000.0, nights_held=3)
    assert one_night > 0.0
    assert three_nights == pytest.approx(3.0 * one_night)


def test_overnight_financing_cost_scales_with_notional() -> None:
    cfg = CostModelConfig()
    small = overnight_financing_cost(cfg, notional=750_000.0, nights_held=2)
    large = overnight_financing_cost(cfg, notional=1_500_000.0, nights_held=2)
    assert large == pytest.approx(2.0 * small)


def test_overnight_financing_cost_rejects_negative_nights() -> None:
    with pytest.raises(ValueError):
        overnight_financing_cost(CostModelConfig(), notional=1_500_000.0, nights_held=-1)


def test_statutory_charges_stt_only_on_sell_and_stamp_duty_only_on_buy() -> None:
    """Cost-model audit (2026-07-18): the previous implementation charged full STT and
    stamp duty on every order regardless of side, effectively double-charging both
    one-sided real-world statutory items. Both fixed rates are non-zero by default, so
    a genuine bug here would show up as buy_stat == sell_stat instead of an asymmetric
    split."""
    from helion_risk_world.execution.cost_model import ConservativeIndianCostModel
    from helion_risk_world.schemas.execution_schema import CandidateOrder

    cfg = CostModelConfig()
    model = ConservativeIndianCostModel(cfg)
    notional = 1_500_000.0
    buy = CandidateOrder(symbol="BANKNIFTY_FUT_continuous", side="buy", qty=30, notional=notional)
    sell = CandidateOrder(symbol="BANKNIFTY_FUT_continuous", side="sell", qty=30, notional=notional)

    buy_cost = model.statutory(buy)
    sell_cost = model.statutory(sell)
    assert buy_cost != sell_cost
    # Buy carries stamp duty but not STT; sell carries STT but not stamp duty.
    assert sell_cost - buy_cost == pytest.approx((cfg.stt_rate - cfg.stamp_duty_rate) * notional)
    assert (buy_cost + sell_cost) == pytest.approx(
        round_trip_cost_frac(cfg, reference_notional=notional) * notional
        - 2 * ((cfg.half_spread_bps + cfg.slippage_bps)) * notional
    )
