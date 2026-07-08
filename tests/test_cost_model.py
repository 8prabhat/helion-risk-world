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


def test_is_symmetric_double_of_one_way() -> None:
    cfg = CostModelConfig()
    frac = round_trip_cost_frac(cfg, reference_notional=1_500_000.0)
    one_way = frac / 2.0
    assert one_way > 0.0
    assert frac == 2.0 * one_way


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
