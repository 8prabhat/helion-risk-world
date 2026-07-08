"""Tests for ModelInputContract.assert_compatible (review finding M16).

Covers each mismatch branch individually: an artifact trained against one
config must refuse to run against a config that changed universe, interval,
lookback, or the candle feature layout, rather than silently misaligning
tensors at inference time.
"""

from __future__ import annotations

import pytest

from helion_risk_world.config.data_config import DataConfig
from helion_risk_world.data.market_window_builder import CANDLE_FEATURE_NAMES
from helion_risk_world.data.model_input_builder import ModelInputContract


def _cfg(**overrides: object) -> DataConfig:
    defaults: dict[str, object] = {
        "universe": ("BANKNIFTY",),
        "base_interval": "5min",
        "lookback_bars": 96,
    }
    defaults.update(overrides)
    return DataConfig(**defaults)


def _contract(cfg: DataConfig) -> ModelInputContract:
    return ModelInputContract.from_data_config(cfg, feature_names=CANDLE_FEATURE_NAMES)


def test_assert_compatible_passes_for_matching_config() -> None:
    cfg = _cfg()
    _contract(cfg).assert_compatible(cfg)  # no raise


def test_assert_compatible_rejects_universe_mismatch() -> None:
    contract = _contract(_cfg(universe=("BANKNIFTY",)))
    with pytest.raises(ValueError, match="universe"):
        contract.assert_compatible(_cfg(universe=("NIFTY",)))


def test_assert_compatible_rejects_base_interval_mismatch() -> None:
    contract = _contract(_cfg(base_interval="5min"))
    with pytest.raises(ValueError, match="base_interval"):
        contract.assert_compatible(_cfg(base_interval="1min"))


def test_assert_compatible_rejects_lookback_bars_mismatch() -> None:
    contract = _contract(_cfg(lookback_bars=96))
    with pytest.raises(ValueError, match="lookback_bars"):
        contract.assert_compatible(_cfg(lookback_bars=48))


def test_assert_compatible_rejects_feature_name_mismatch() -> None:
    cfg = _cfg()
    contract = ModelInputContract.from_data_config(
        cfg, feature_names=tuple(CANDLE_FEATURE_NAMES) + ("extra_feature",)
    )
    with pytest.raises(ValueError, match="feature layout"):
        contract.assert_compatible(cfg)


def test_to_metadata_from_metadata_round_trip_preserves_compatibility() -> None:
    cfg = _cfg()
    contract = ModelInputContract.from_data_config(
        cfg,
        feature_names=CANDLE_FEATURE_NAMES,
        barrier_stop_mult=1.5,
        barrier_target_mult=2.5,
        barrier_vol_span=25,
        barrier_horizon_bars=192,
        barrier_cost_floor_frac=0.001,
    )
    restored = ModelInputContract.from_metadata({"input_contract": contract.to_metadata()})
    assert restored == contract
    assert restored.barrier_horizon_bars == 192
    assert restored.barrier_cost_floor_frac == pytest.approx(0.001)
    restored.assert_compatible(cfg)


def test_from_metadata_returns_none_when_absent() -> None:
    assert ModelInputContract.from_metadata({}) is None


def test_from_legacy_metadata_defaults_missing_barrier_geometry() -> None:
    cfg = _cfg()
    payload = _contract(cfg).to_metadata()
    payload.pop("barrier_horizon_bars", None)
    payload.pop("barrier_cost_floor_frac", None)

    restored = ModelInputContract.from_metadata({"input_contract": payload})

    assert restored is not None
    assert restored.barrier_horizon_bars == 1
    assert restored.barrier_cost_floor_frac == 0.0
