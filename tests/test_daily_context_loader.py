"""DailyContextLoader: derived macro-column stabilization (feature/label overhaul Phase 0).

fii_dii_net/pc_oi_ratio -> rolling 60-day z-score; usdinr/crude -> 5-day rate-of-change
+ rolling vol. All must be trailing-only (no look-ahead) and match the existing
DailyContextLoader.get() point-in-time contract.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from helion_risk_world.data.daily_context_loader import (
    DailyContextLoader,
    _ZSCORE_MIN_PERIODS,
    _ZSCORE_WINDOW_DAYS,
)


def _write_daily_context(tmp_path: Path, df: pd.DataFrame) -> Path:
    path = tmp_path / "daily_context.parquet"
    df.to_parquet(path, index=True)
    return path


def _synthetic_frame(n: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(7)
    fii_dii_net = rng.normal(0.0, 1000.0, n).cumsum()  # trending flow series
    pc_oi_ratio = 1.0 + rng.normal(0.0, 0.05, n)
    usdinr = 82.0 + np.linspace(0.0, 13.0, n)  # steady multi-month drift
    crude = 70.0 + rng.normal(0.0, 1.0, n).cumsum() * 0.05
    return pd.DataFrame(
        {"fii_dii_net": fii_dii_net, "pc_oi_ratio": pc_oi_ratio, "usdinr": usdinr, "crude": crude},
        index=dates,
    )


def test_zscore_columns_present_and_clipped(tmp_path: Path) -> None:
    df = _synthetic_frame()
    path = _write_daily_context(tmp_path, df)
    loader = DailyContextLoader(path)
    ctx = loader.get(df.index[-1])
    assert ctx["fii_dii_net_z"] is not None
    assert -3.0 <= ctx["fii_dii_net_z"] <= 3.0
    assert ctx["pc_oi_ratio_z"] is not None
    assert -3.0 <= ctx["pc_oi_ratio_z"] <= 3.0


def test_rate_of_change_columns_present(tmp_path: Path) -> None:
    df = _synthetic_frame()
    path = _write_daily_context(tmp_path, df)
    loader = DailyContextLoader(path)
    ctx = loader.get(df.index[-1])
    assert ctx["usdinr_ret_5d"] is not None
    assert ctx["crude_ret_5d"] is not None
    assert ctx["usdinr_vol"] is not None
    assert ctx["crude_vol"] is not None


def test_zscore_is_near_zero_at_rolling_mean(tmp_path: Path) -> None:
    # A flat series after warmup: raw value sitting exactly at its own trailing mean
    # should z-score to ~0.
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    flat = pd.Series([100.0] * n, index=dates)
    df = pd.DataFrame({"fii_dii_net": flat, "pc_oi_ratio": flat, "usdinr": flat + 80, "crude": flat + 70})
    path = _write_daily_context(tmp_path, df)
    loader = DailyContextLoader(path)
    ctx = loader.get(dates[-1])
    assert ctx["fii_dii_net_z"] == pytest.approx(0.0, abs=1e-4)


def test_no_look_ahead_leak(tmp_path: Path) -> None:
    """A derived value at row i must be unchanged when rows > i are mutated."""
    df = _synthetic_frame()
    (tmp_path / "a").mkdir(exist_ok=True)
    path_a = _write_daily_context(tmp_path / "a", df)
    loader_a = DailyContextLoader(path_a)
    probe_ts = df.index[60]
    ctx_a = loader_a.get(probe_ts)

    df_b = df.copy()
    # Mutate everything strictly after the probe timestamp.
    df_b.loc[df_b.index > probe_ts, "fii_dii_net"] += 1_000_000.0
    df_b.loc[df_b.index > probe_ts, "usdinr"] += 500.0
    (tmp_path / "b").mkdir(exist_ok=True)
    path_b = _write_daily_context(tmp_path / "b", df_b)
    loader_b = DailyContextLoader(path_b)
    ctx_b = loader_b.get(probe_ts)

    assert ctx_a["fii_dii_net_z"] == pytest.approx(ctx_b["fii_dii_net_z"])
    assert ctx_a["usdinr_ret_5d"] == pytest.approx(ctx_b["usdinr_ret_5d"])


def test_expanding_window_warmup_no_nan_returned(tmp_path: Path) -> None:
    """Before _ZSCORE_MIN_PERIODS rows exist, get() must still return a usable value
    (None rather than a NaN leaking out as a float) or a valid finite float — never NaN."""
    df = _synthetic_frame(n=_ZSCORE_MIN_PERIODS + 5)
    path = _write_daily_context(tmp_path, df)
    loader = DailyContextLoader(path)
    early_ts = df.index[_ZSCORE_MIN_PERIODS - 5]
    ctx = loader.get(early_ts)
    # Either a real number or None — never NaN slipping through as a float.
    val = ctx["fii_dii_net_z"]
    assert val is None or np.isfinite(val)


def test_missing_source_columns_degrade_gracefully(tmp_path: Path) -> None:
    """If the parquet lacks a raw column entirely, its derived columns are just absent
    (None), matching the existing get() contract for missing columns."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame({"fii_dii_net": [1.0] * 30}, index=dates)
    path = _write_daily_context(tmp_path, df)
    loader = DailyContextLoader(path)
    ctx = loader.get(dates[-1])
    assert ctx["usdinr_ret_5d"] is None
    assert ctx["crude_vol"] is None
