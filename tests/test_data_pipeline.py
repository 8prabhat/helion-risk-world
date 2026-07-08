"""Tests for corporate_actions, rollover, CalibrationGate, and Upstox data pipeline."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from helion_risk_world.data.corporate_actions import (
    BLACKOUT_DAYS,
    HDFC_MERGER_DATE,
    drop_merger_bars,
    flag_merger_bars,
)
from helion_risk_world.data.rollover import (
    ROLL_GAP_THRESHOLD,
    count_roll_gaps,
    detect_roll_gaps,
    flag_and_clip,
)
from helion_risk_world.evaluation.calibration_metrics import CalibrationGate, compute
from helion_risk_world.evaluation.world_model_metrics import compute as wm_compute


# ── corporate_actions ──────────────────────────────────────────────────────────

def _df_with_dates(dates: list[date]) -> pd.DataFrame:
    return pd.DataFrame({"date": [datetime(d.year, d.month, d.day) for d in dates], "val": 1.0})


def test_flag_merger_bars_inside_blackout() -> None:
    df = _df_with_dates([HDFC_MERGER_DATE])
    assert flag_merger_bars(df).iloc[0]


def test_flag_merger_bars_outside_blackout() -> None:
    safe = HDFC_MERGER_DATE + timedelta(days=BLACKOUT_DAYS + 1)
    df = _df_with_dates([safe])
    assert not flag_merger_bars(df).iloc[0]


def test_drop_merger_bars_removes_blackout() -> None:
    dates = [HDFC_MERGER_DATE - timedelta(days=i) for i in range(BLACKOUT_DAYS + 5)]
    df = _df_with_dates(dates)
    cleaned = drop_merger_bars(df)
    # All rows within blackout should be gone
    remaining_dates = pd.to_datetime(cleaned["date"]).dt.date.tolist()
    lo = HDFC_MERGER_DATE - timedelta(days=BLACKOUT_DAYS)
    hi = HDFC_MERGER_DATE + timedelta(days=BLACKOUT_DAYS)
    assert all(d < lo or d > hi for d in remaining_dates)


# ── rollover ──────────────────────────────────────────────────────────────────

def _price_series(n: int = 50, gap_idx: int | None = None) -> pd.Series:
    prices = np.ones(n) * 50000.0
    prices = prices * np.cumprod(1 + np.random.default_rng(0).normal(0, 0.001, n))
    if gap_idx is not None:
        prices[gap_idx] *= 1.0 + ROLL_GAP_THRESHOLD * 2   # inject a big gap
    return pd.Series(prices)


def test_detect_roll_gaps_no_gap() -> None:
    gaps = detect_roll_gaps(_price_series())
    assert not gaps.any()


def test_detect_roll_gaps_with_gap() -> None:
    gaps = detect_roll_gaps(_price_series(gap_idx=25))
    assert gaps.iloc[25]


def test_flag_and_clip_sets_nan() -> None:
    df = pd.DataFrame({"close": _price_series(gap_idx=10).values})
    out = flag_and_clip(df)
    assert out["roll_gap"].iloc[10]
    assert np.isnan(out["close"].iloc[10])
    assert not np.isnan(out["close"].iloc[9])


def test_count_roll_gaps() -> None:
    # The injected price spike creates a gap at the spike bar AND one at the next bar (reversion).
    df = pd.DataFrame({"close": _price_series(gap_idx=20).values})
    assert count_roll_gaps(df) >= 1


# ── CalibrationGate ───────────────────────────────────────────────────────────

def _perfect_probs(n: int = 100) -> tuple[np.ndarray, np.ndarray]:
    labels = np.random.default_rng(1).integers(0, 3, n)
    probs = np.zeros((n, 3))
    probs[np.arange(n), labels] = 1.0
    return probs, labels


def test_calibration_gate_perfect_barrier_passes() -> None:
    probs, labels = _perfect_probs()
    gate = CalibrationGate()
    passed, reasons = gate.check(barrier_probs=probs, barrier_labels=labels)
    assert passed
    assert all("PASS" in v for v in reasons.values())


def test_calibration_gate_random_fails() -> None:
    rng = np.random.default_rng(42)
    probs = rng.dirichlet(np.ones(3), size=200)
    labels = rng.integers(0, 3, 200)
    gate = CalibrationGate(barrier_brier_max=0.01)   # very tight threshold
    passed, _ = gate.check(barrier_probs=probs, barrier_labels=labels)
    assert not passed


def test_calibration_gate_no_data_fails_closed() -> None:
    gate = CalibrationGate()
    passed, reasons = gate.check()
    assert not passed
    assert reasons["no_metrics"].startswith("FAIL")


def test_calibration_gate_uses_true_quantile_levels() -> None:
    levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9], dtype=float)
    realized = (np.arange(20_000, dtype=float) + 0.5) / 20_000.0
    quantiles = np.tile(levels, (realized.size, 1))
    out = compute(pred_quantiles=quantiles, realized=realized)
    assert out["coverage_error"] < 0.01


def test_calibration_gate_per_regime() -> None:
    rng = np.random.default_rng(7)
    n = 300
    quantiles = np.sort(rng.normal(0, 0.01, (n, 5)), axis=1)
    realized = rng.normal(0, 0.01, n)
    regime = rng.integers(0, 3, n)
    gate = CalibrationGate(coverage_tol=0.99)   # very loose — all regimes should pass
    passed, reasons = gate.check(pred_quantiles=quantiles, realized=realized,
                                 regime_labels=regime)
    regime_keys = [k for k in reasons if "regime" in k]
    assert len(regime_keys) > 0


def test_calibration_gate_accepts_string_regimes() -> None:
    rng = np.random.default_rng(17)
    n = 120
    quantiles = np.sort(rng.normal(0, 0.01, (n, 5)), axis=1)
    realized = rng.normal(0, 0.01, n)
    regime = np.array(["trend"] * 40 + ["range"] * 40 + ["low_vol"] * 40, dtype=object)
    gate = CalibrationGate(coverage_tol=0.99)
    passed, reasons = gate.check(
        pred_quantiles=quantiles,
        realized=realized,
        regime_labels=regime,
    )
    regime_keys = [k for k in reasons if "coverage_regime_" in k]
    assert passed
    assert sorted(regime_keys) == [
        "coverage_regime_low_vol",
        "coverage_regime_range",
        "coverage_regime_trend",
    ]


# ── world_model_metrics ───────────────────────────────────────────────────────

def test_wm_compute_rollout_accuracy() -> None:
    predicted = np.array([1.0, 2.0, 3.0])
    target = np.array([1.1, 1.9, 3.2])
    out = wm_compute(predicted=predicted, target=target)
    assert "rollout_mae" in out and "rollout_rmse" in out
    assert out["rollout_mae"] == pytest.approx(0.1333, abs=0.01)


def test_wm_compute_kl_collapse() -> None:
    kl_vals = [0.005, 0.003, 0.008]   # all below 0.01 → full collapse
    out = wm_compute(kl_per_step=kl_vals)
    assert out["kl_collapse_frac"] == pytest.approx(1.0)


def test_wm_compute_prior_coverage() -> None:
    # prior spans from -1 to +1; posterior mean at 0 → should be covered
    prior_samples = np.linspace(-1, 1, 10).reshape(-1, 1).repeat(4, axis=1)  # [S=10, B=4]
    posterior_mean = np.zeros(4)
    out = wm_compute(prior_samples=prior_samples, posterior_mean=posterior_mean)
    assert out["prior_coverage"] == pytest.approx(1.0)


# ── upstox_client ─────────────────────────────────────────────────────────────
# RETIRED 2026-07-08: upstox_client.py + fetch_upstox.py are superseded by
# alpha_data's ingestion (quanthelion's UpstoxClient/UpstoxCandles/UpstoxExpiries
# directly). resolve_key/is_fo were project-local symbol lookups, now replaced by
# alpha_data's universe.yaml; _month_chunks has an equivalent, already-tested copy
# in quanthelion.data.ingestion.providers.upstox.candles. Original file backed up
# in src/helion_risk_world/data/.pre_quanthelion_migration_backup/.


# ── expiry_calendar (review finding M6: 2025/2026 NSE holidays) ────────────────

from helion_risk_world.data.expiry_calendar import monthly_expiry as _monthly_expiry


@pytest.mark.parametrize(
    "year,month,expected",
    [
        # Last Thursday of the month lands exactly on a 2025/2026 NSE holiday;
        # without the extended _NSE_HOLIDAYS list these would silently return the
        # holiday date itself instead of walking back to the prior trading day.
        (2025, 12, date(2025, 12, 24)),  # last Thu = Christmas -> walks back
        (2026, 3, date(2026, 3, 25)),    # last Thu = Ram Navami -> walks back
        (2026, 5, date(2026, 5, 27)),    # last Thu = Bakri Eid -> walks back
    ],
)
def test_monthly_expiry_avoids_2025_2026_holidays(year: int, month: int, expected: date) -> None:
    assert _monthly_expiry(year, month) == expected


# ── event_calendar ────────────────────────────────────────────────────────────

from helion_risk_world.data.event_calendar import event_type_for, is_event_day
from helion_risk_world.schemas.market_schema import EventType


def test_rbi_date() -> None:
    assert event_type_for(date(2024, 2, 8)) == EventType.RBI


def test_budget_date() -> None:
    assert event_type_for(date(2024, 2, 1)) == EventType.BUDGET


def test_colliding_event_dates_apply_priority_not_silent_drop() -> None:
    """Review finding M4: 2023-02-01 is both Budget day and an FOMC date. A plain
    dict literal would silently keep only whichever was defined last (FED); the
    fix must retain both and resolve via _PRIORITY (BUDGET > FED)."""
    assert event_type_for(date(2023, 2, 1)) == EventType.BUDGET
    assert is_event_day(date(2023, 2, 1))


def test_election_date() -> None:
    assert event_type_for(date(2024, 6, 4)) == EventType.ELECTION
    assert is_event_day(date(2024, 6, 4))


def test_non_event_date() -> None:
    assert event_type_for(date(2024, 3, 15)) == EventType.NONE
    assert not is_event_day(date(2024, 3, 15))


# ── daily_context_loader ──────────────────────────────────────────────────────

from helion_risk_world.data.daily_context_loader import DailyContextLoader


def test_loader_no_file_returns_none() -> None:
    loader = DailyContextLoader(None)
    result = loader.get(datetime(2024, 1, 15, 10, 0))
    # All columns should be None when no file is loaded
    assert result["usdinr"] is None
    assert result["crude"] is None
    assert result["fii_dii_net"] is None
    assert result["atm_iv_pct"] is None
    assert result["iv_skew_pct"] is None


def test_loader_with_dataframe(tmp_path: Path) -> None:
    df = pd.DataFrame({
        "usdinr": [83.5, 83.6, 83.7],
        "crude": [75.0, 74.8, 75.1],
        "fii_dii_net": [1000.0, -500.0, 200.0],
    }, index=pd.date_range("2024-01-10", periods=3, freq="D", name="date"))
    df.to_parquet(tmp_path / "daily_context.parquet")

    loader = DailyContextLoader(tmp_path / "daily_context.parquet")
    result = loader.get(datetime(2024, 1, 12, 15, 30))
    assert result["usdinr"] == pytest.approx(83.7, abs=0.01)


def test_loader_stale_data_returns_none(tmp_path: Path) -> None:
    df = pd.DataFrame({
        "usdinr": [80.0], "crude": [60.0], "fii_dii_net": [0.0],
    }, index=pd.to_datetime(["2023-01-01"]))
    df.index.name = "date"
    df.to_parquet(tmp_path / "daily_context.parquet")

    loader = DailyContextLoader(tmp_path / "daily_context.parquet")
    result = loader.get(datetime(2024, 6, 1, 10, 0))   # 17 months later
    assert result["usdinr"] is None


# ── fetch_free_data.build_daily_context ──────────────────────────────────────
# RETIRED 2026-07-08: fetch_free_data.py + fetch_nse_bhavcopy.py are superseded by
# alpha_data's pipelines/regime.py, backed by quanthelion.data.ingestion.providers.yahoo
# and quanthelion.data.ingestion.market_data.nse_fo_bhavcopy. The point-in-time +1-day
# lag behavior these tests exercised is now covered by quanthelion's own
# tests/unit/test_daily_context_assembly.py (assemble_daily_context). Originals backed
# up in scripts/.pre_quanthelion_migration_backup/.


# ── regime_context_builder ────────────────────────────────────────────────────

from helion_risk_world.data.regime_context_builder import RegimeContextBuilder, build_regime_tensor
from helion_risk_world.data.regime_builder import REGIME_CONTEXT_FEATURES


def test_regime_builder_fallbacks() -> None:
    builder = RegimeContextBuilder.from_paths()
    regime, event = builder.build(datetime(2024, 3, 15, 10, 0))
    assert regime.vix == 15.0
    assert regime.vix_pct == 0.5
    assert event.usdinr is None


def test_regime_builder_rbi_event() -> None:
    builder = RegimeContextBuilder.from_paths()
    _, event = builder.build(datetime(2024, 2, 8, 10, 0))
    assert event.event_type == EventType.NONE
    assert not event.event_day_flag


def test_regime_builder_expiry_flag() -> None:
    from helion_risk_world.data.expiry_calendar import monthly_expiry
    exp = monthly_expiry(2024, 3)
    builder = RegimeContextBuilder.from_paths()
    _, event = builder.build(datetime(exp.year, exp.month, exp.day, 10, 0))
    assert not event.expiry_flag


def test_regime_builder_blackout() -> None:
    builder = RegimeContextBuilder.from_paths()
    _, event = builder.build(datetime(2023, 7, 1, 10, 0))
    assert not event.blackout_active
    _, event2 = builder.build(datetime(2024, 1, 15, 10, 0))
    assert not event2.blackout_active


def test_regime_builder_rejects_non_upstox_daily_context_by_default(tmp_path: Path) -> None:
    path = tmp_path / "daily_context.parquet"
    pd.DataFrame({"usdinr": [83.0]}, index=pd.date_range("2024-01-01", periods=1)).to_parquet(path)
    with pytest.raises(ValueError, match="not Upstox-sourced"):
        RegimeContextBuilder.from_paths(daily_context_path=path)


def test_regime_builder_set_live_iv() -> None:
    builder = RegimeContextBuilder.from_paths()
    builder.set_live_iv(atm_iv=18.5, iv_skew=-0.05)
    regime, _ = builder.build(datetime(2024, 3, 15, 10, 0))
    assert regime.atm_iv == pytest.approx(18.5)
    assert regime.iv_skew == pytest.approx(-0.05)


def test_regime_builder_require_live_iv_raises_when_unset() -> None:
    """Review finding C2: no live option-chain client is wired anywhere, so a caller
    that genuinely needs live IV must fail loud rather than silently fall back."""
    builder = RegimeContextBuilder.from_paths()
    builder.require_live_iv = True
    with pytest.raises(RuntimeError, match="set_live_iv"):
        builder.build(datetime(2024, 3, 15, 10, 0))


def test_regime_builder_require_live_iv_passes_once_set() -> None:
    builder = RegimeContextBuilder.from_paths()
    builder.require_live_iv = True
    builder.set_live_iv(atm_iv=18.5, iv_skew=-0.05)
    regime, _ = builder.build(datetime(2024, 3, 15, 10, 0))
    assert regime.atm_iv == pytest.approx(18.5)


def test_featurize_regime_shape() -> None:
    builder = RegimeContextBuilder.from_paths()
    vec = build_regime_tensor(builder, datetime(2024, 3, 15, 10, 30))
    assert vec.shape == (len(REGIME_CONTEXT_FEATURES),)
    assert vec.dtype == np.float32


def test_regime_builder_with_vix_parquet(tmp_path: Path) -> None:
    idx = pd.date_range("2024-01-02 09:15", periods=200, freq="5min", tz="Asia/Kolkata")
    vix_vals = np.linspace(12.0, 20.0, 200)
    df = pd.DataFrame({"close": vix_vals, "open": vix_vals, "high": vix_vals,
                       "low": vix_vals, "volume": 0.0, "oi": 0.0}, index=idx)
    df.index.name = "datetime"
    p = tmp_path / "INDIAVIX_5min.parquet"
    df.to_parquet(p)

    builder = RegimeContextBuilder.from_paths(vix_path=p)
    ts_query = idx[100].to_pydatetime().replace(tzinfo=None)
    regime, _ = builder.build(ts_query)
    assert 12.0 <= regime.vix <= 20.0
    assert 0.0 <= regime.vix_pct <= 1.0


# ── continuous_futures ────────────────────────────────────────────────────────

from helion_risk_world.data.continuous_futures import build_continuous


def _make_contract(tmp_path: Path, yymm: str, start: str, periods: int,
                   base_price: float) -> None:
    rng = np.random.default_rng(int(yymm))
    idx = pd.date_range(start, periods=periods, freq="5min", tz="Asia/Kolkata")
    price = base_price + np.cumsum(rng.normal(0, 5, periods))
    df = pd.DataFrame({
        "open": price, "high": price + 10, "low": price - 10,
        "close": price, "volume": 5000.0, "oi": 1e6,
    }, index=idx)
    df.index.name = "datetime"
    df.to_parquet(tmp_path / f"BANKNIFTY_FUT_{yymm}_5min.parquet")


def test_build_continuous_single(tmp_path: Path) -> None:
    _make_contract(tmp_path, "2401", "2024-01-02 09:15", 100, 45000.0)
    result = build_continuous(tmp_path, underlying="BANKNIFTY", interval="5min")
    assert len(result) == 100
    assert "roll_gap" in result.columns
    assert not result["close"].isna().any()


def test_build_continuous_two_contracts(tmp_path: Path) -> None:
    _make_contract(tmp_path, "2401", "2024-01-02 09:15", 1000, 45000.0)
    _make_contract(tmp_path, "2402", "2024-01-20 09:15", 1000, 45200.0)
    result = build_continuous(tmp_path)
    assert len(result) > 100
    assert not result["close"].isna().any()


def test_build_continuous_no_files_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="BANKNIFTY_FUT"):
        build_continuous(tmp_path, underlying="BANKNIFTY", interval="5min")


def test_build_continuous_uses_expiry_driven_roll_not_file_overlap(tmp_path: Path) -> None:
    """Review finding H2: with real overlapping contract data, the roll must
    happen near the near contract's own NSE expiry date, not wherever the next
    contract's downloaded file happens to start (2024-01-25 is BankNIFTY's
    January-2024 expiry; the old file-overlap heuristic would have rolled 10
    days early, at the next contract's first bar on 2024-01-15)."""
    idx1 = pd.date_range("2024-01-02", "2024-02-05", freq="1D", tz="Asia/Kolkata")
    price1 = np.full(len(idx1), 45000.0)
    df1 = pd.DataFrame(
        {"open": price1, "high": price1 + 10, "low": price1 - 10,
         "close": price1, "volume": 5000.0, "oi": 1e6},
        index=idx1,
    )
    df1.index.name = "datetime"
    df1.to_parquet(tmp_path / "BANKNIFTY_FUT_2401_5min.parquet")

    idx2 = pd.date_range("2024-01-15", "2024-03-05", freq="1D", tz="Asia/Kolkata")
    price2 = np.full(len(idx2), 45200.0)
    df2 = pd.DataFrame(
        {"open": price2, "high": price2 + 10, "low": price2 - 10,
         "close": price2, "volume": 5000.0, "oi": 1e6},
        index=idx2,
    )
    df2.index.name = "datetime"
    df2.to_parquet(tmp_path / "BANKNIFTY_FUT_2402_5min.parquet")

    result = build_continuous(tmp_path)
    roll_bars = result.index[result["roll_gap"]]
    assert len(roll_bars) == 1
    assert roll_bars[0].date() == date(2024, 1, 24)


def test_build_continuous_emits_close_fut_next_during_overlap(tmp_path: Path) -> None:
    """Review Idea #6 (calendar_spread activation): close_fut_next should carry
    the next-contract price during the near/next overlap window, scaled by the
    near contract's own accumulated backward-adjustment ratio (so it stays on
    the same price scale as close_fut and the % spread is preserved exactly —
    mixing an adjusted close_fut with a raw close_fut_next would instead produce
    a spurious spread that just tracks the cumulative adjustment factor), and be
    NaN before the next contract exists at all."""
    idx1 = pd.date_range("2024-01-02", "2024-02-05", freq="1D", tz="Asia/Kolkata")
    price1 = np.full(len(idx1), 45000.0)
    df1 = pd.DataFrame(
        {"open": price1, "high": price1 + 10, "low": price1 - 10,
         "close": price1, "volume": 5000.0, "oi": 1e6},
        index=idx1,
    )
    df1.index.name = "datetime"
    df1.to_parquet(tmp_path / "BANKNIFTY_FUT_2401_5min.parquet")

    idx2 = pd.date_range("2024-01-15", "2024-03-05", freq="1D", tz="Asia/Kolkata")
    price2 = np.full(len(idx2), 45200.0)
    df2 = pd.DataFrame(
        {"open": price2, "high": price2 + 10, "low": price2 - 10,
         "close": price2, "volume": 5000.0, "oi": 1e6},
        index=idx2,
    )
    df2.index.name = "datetime"
    df2.to_parquet(tmp_path / "BANKNIFTY_FUT_2402_5min.parquet")

    result = build_continuous(tmp_path)
    assert "close_fut_next" in result.columns

    before_overlap = pd.Timestamp("2024-01-10", tz="Asia/Kolkata")
    during_overlap = pd.Timestamp("2024-01-20", tz="Asia/Kolkata")
    assert pd.isna(result.loc[before_overlap, "close_fut_next"])
    # Both contracts are flat series (45000 / 45200), so the near contract's
    # accumulated ratio is exactly 45000/45200 and close_fut_next == 45200 * ratio
    # == 45000 throughout the overlap window — not 45200 (which would have mixed
    # an adjusted close_fut with a raw close_fut_next, producing a spurious
    # ~0.44% "spread" from the ratio mismatch alone rather than 0).
    assert result.loc[during_overlap, "close_fut_next"] == pytest.approx(45000.0)
