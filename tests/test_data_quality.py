"""Schema validation + point-in-time tests (SPEC.md §9, §27)."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from helion_risk_world.data.data_quality import DataQualityReport
from helion_risk_world.schemas import (
    LabelRecord,
    MarketCandle,
    OptionContractSnapshot,
    OptionType,
)

T = datetime(2026, 6, 25, 10, 0)


def test_market_candle_point_in_time_ok() -> None:
    c = MarketCandle(symbol="BANKNIFTY", ts=T, available_at=T,
                     open=1, high=2, low=0.5, close=1.5, volume=100)
    assert c.close == 1.5


def test_market_candle_future_availability_rejected() -> None:
    with pytest.raises(ValidationError):
        MarketCandle(symbol="X", ts=T, available_at=T + timedelta(minutes=5),
                     open=1, high=2, low=0.5, close=1.5, volume=1)


def test_label_future_only_enforced() -> None:
    from helion_risk_world.schemas.label_schema import Barrier
    LabelRecord(
        symbol="X", ts=T, label_realized_at=T + timedelta(minutes=15),
        horizon_bars=3, barrier=Barrier.TIMEOUT,
        exit_return=0.01, exit_t=3, realized_vol=0.1, mae=0.005, mfe=0.01,
    )
    with pytest.raises(ValidationError):
        LabelRecord(
            symbol="X", ts=T, label_realized_at=T,  # label_realized_at == ts → leakage
            horizon_bars=3, barrier=Barrier.TIMEOUT,
            exit_return=0.0, exit_t=3, realized_vol=0.1, mae=0.0, mfe=0.0,
        )


def test_option_snapshot_pit() -> None:
    o = OptionContractSnapshot(underlying="BANKNIFTY", strike=50000, opt_type=OptionType.CALL,
                               ts=T, available_at=T, open=1, high=2, low=0.5, close=1.5,
                               volume=10, oi=1000, dte=2.0)
    assert o.opt_type is OptionType.CALL


def test_data_quality_report_detects_duplicates_and_future_labels() -> None:
    records = [
        {"symbol": "X", "ts": T, "available_at": T, "volume": 0.0, "oi": 0.0},
        {"symbol": "X", "ts": T, "available_at": T, "volume": 0.0, "oi": 0.0},
        {
            "symbol": "X",
            "ts": T,
            "available_at": T,
            "label_realized_at": T - timedelta(minutes=5),
        },
    ]
    report = DataQualityReport().validate(records)
    assert report["passed"] is False
    assert report["duplicates"] == 2
    assert report["future_label_violations"] == 1


def test_data_quality_allows_structural_zero_volume_for_indices() -> None:
    records = [
        {
            "symbol": "BANKNIFTY",
            "ts": T,
            "available_at": T,
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": 1.0,
            "volume": 0.0,
            "oi": 0.0,
        },
        {
            "symbol": "NIFTY",
            "ts": T + timedelta(minutes=5),
            "available_at": T + timedelta(minutes=5),
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": 1.0,
            "volume": 0.0,
            "oi": 0.0,
        },
    ]
    report = DataQualityReport().validate(records)
    assert report["passed"] is True
    assert report["zero_volume_rows"] == 0
    assert report["structural_zero_volume_rows"] == 2
