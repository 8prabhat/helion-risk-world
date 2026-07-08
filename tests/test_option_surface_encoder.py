"""Option-surface ATM alignment contract (SPEC.md §16, §27, Day 3).

Builder logic is now implemented; these tests prove ATM centering, masking of missing strikes,
fixed-width output, and derived-feature sanity.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from helion_risk_world.data.option_surface_builder import OptionSurfaceBuilder, infer_strike_step
from helion_risk_world.schemas.option_chain_schema import OptionContractSnapshot, OptionType

TS = datetime(2026, 6, 25, 11, 0)


def _contract(
    strike: float, opt: OptionType, oi: float, iv: float | None = 0.2
) -> OptionContractSnapshot:
    return OptionContractSnapshot(
        underlying="BANKNIFTY", strike=strike, opt_type=opt, ts=TS, available_at=TS,
        open=10, high=12, low=8, close=11, volume=100, oi=oi, d_oi=5, iv=iv, dte=2.0,
    )


def _full_chain(
    atm: float = 50000.0, step: float = 100.0, width: int = 4
) -> list[OptionContractSnapshot]:
    chain: list[OptionContractSnapshot] = []
    for i in range(-width, width + 1):
        k = atm + i * step
        chain.append(_contract(k, OptionType.CALL, oi=1000 + 10 * i))
        chain.append(_contract(k, OptionType.PUT, oi=1000 - 10 * i))
    return chain


def test_builder_constructs_with_n_strikes() -> None:
    assert OptionSurfaceBuilder(n_strikes=5)._n_strikes == 5


def test_infer_strike_step() -> None:
    assert infer_strike_step([49800, 49900, 50000, 50100]) == 100.0


def test_align_centers_on_atm_and_is_fixed_width() -> None:
    surf = OptionSurfaceBuilder(n_strikes=2).align_to_atm(_full_chain(), spot=50040.0, ts=TS)
    assert surf.atm_strike == 50000.0  # nearest grid strike to spot
    assert len(surf.strikes) == 5  # 2N+1
    assert [r.token for r in surf.strikes] == [-2, -1, 0, 1, 2]
    atm_row = next(r for r in surf.strikes if r.token == 0)
    assert atm_row.strike == 50000.0 and not atm_row.is_masked


def test_missing_strike_is_masked_not_dropped() -> None:
    chain = [c for c in _full_chain(width=4) if c.strike != 50100.0]  # drop both legs at ATM+1
    surf = OptionSurfaceBuilder(n_strikes=2).align_to_atm(chain, spot=50000.0, ts=TS)
    row = next(r for r in surf.strikes if r.token == 1)
    assert row.is_masked
    assert row.call_oi is None and row.put_oi is None
    assert len(surf.strikes) == 5  # still fixed width


def test_derived_features_present_and_sane() -> None:
    surf = OptionSurfaceBuilder(n_strikes=3).align_to_atm(_full_chain(), spot=50000.0, ts=TS)
    assert surf.pcr is not None and surf.pcr > 0
    assert surf.atm_iv is not None
    assert surf.max_pain_proxy in [r.strike for r in surf.strikes]
    assert 0.0 <= surf.oi_wall_strength <= 1.0


def test_point_in_time_enforced() -> None:
    # Contract is valid on its own clock (available_at == ts == 12:00) but newer than the surface
    # decision time (11:00) -> the builder's defence-in-depth PIT guard must reject it.
    later = datetime(2026, 6, 25, 12, 0)
    future = OptionContractSnapshot(
        underlying="BANKNIFTY", strike=50000, opt_type=OptionType.CALL,
        ts=later, available_at=later,
        open=1, high=2, low=0.5, close=1.5, volume=1, oi=10, dte=2.0,
    )
    with pytest.raises(ValueError):
        OptionSurfaceBuilder(n_strikes=1).align_to_atm([future], spot=50000.0, ts=TS)


def test_empty_chain_rejected() -> None:
    with pytest.raises(ValueError):
        OptionSurfaceBuilder(n_strikes=1).align_to_atm([], spot=50000.0, ts=TS)
