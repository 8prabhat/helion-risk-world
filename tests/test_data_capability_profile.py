"""Local data capability profile tests."""

from __future__ import annotations

from pathlib import Path

from helion_risk_world.config.data_config import DataConfig
from helion_risk_world.data.capability_profile import DataCapabilityProfile


def test_capability_profile_reports_missing_assets(tmp_path: Path) -> None:
    profile = DataCapabilityProfile.from_data_dir(
        tmp_path,
        DataConfig(universe=("BANKNIFTY", "NIFTY"), lookback_bars=12),
    )

    assert profile.available_assets == ()
    assert profile.missing_assets == ("BANKNIFTY", "NIFTY")
    assert "missing_futures_continuous" in profile.critical_issues()
