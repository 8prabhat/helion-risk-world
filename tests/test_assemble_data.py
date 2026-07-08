"""scripts/assemble_data.py::assemble() (review findings H4, M7 — roll_gap preservation
and contiguity segment_id)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
_SPEC = importlib.util.spec_from_file_location(
    "assemble_data_script", _ROOT / "scripts" / "assemble_data.py"
)
assert _SPEC is not None and _SPEC.loader is not None
assemble_data_script = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = assemble_data_script
_SPEC.loader.exec_module(assemble_data_script)


def _ohlcv(index: pd.DatetimeIndex, close: list[float], *, roll_gap: list[bool] | None = None) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": [100.0] * len(close),
            "oi": [0.0] * len(close),
        },
        index=index,
    )
    df.index.name = "datetime"
    if roll_gap is not None:
        df["roll_gap"] = roll_gap
    return df


def test_assemble_preserves_contract_roll_flag_and_flags_price_jumps(tmp_path) -> None:
    """Review finding H4: the authoritative per-contract-roll marker (from
    continuous_futures.py) must survive even when a separate price-jump gap is
    also detected elsewhere in the series — previously flag_and_clip() replaced
    the whole roll_gap column, silently discarding the contract-roll marker."""
    index = pd.date_range("2024-01-10 09:15", periods=6, freq="5min")
    # Bar 2 is flagged as a contract roll (no price jump there); bar 4 has a genuine
    # >2% price jump that the price-jump detector will independently catch.
    close = [100.0, 100.0, 100.0, 100.0, 130.0, 130.0]
    roll_gap = [False, False, True, False, False, False]
    fut = _ohlcv(index, close, roll_gap=roll_gap)
    spot = _ohlcv(index, [50.0] * len(index))

    fut_path = tmp_path / "banknifty_fut_5min.parquet"
    spot_path = tmp_path / "banknifty_spot_5min.parquet"
    fut.to_parquet(fut_path)
    spot.to_parquet(spot_path)

    out_path = tmp_path / "assembled.parquet"
    merged = assemble_data_script.assemble(fut_path, spot_path, out_path, resample="5min")

    assert "roll_gap" in merged.columns
    # Row 2 (contract-roll marker) and row 4 (price-jump) must BOTH be flagged.
    roll_gap_out = merged["roll_gap"].to_numpy()
    assert bool(roll_gap_out[2]), "contract-roll marker was discarded"
    assert bool(roll_gap_out[4]), "price-jump gap was not detected"
    assert not bool(roll_gap_out[0])
    assert not bool(roll_gap_out[3])


def test_assemble_computes_segment_id_across_a_dropped_row_gap(tmp_path) -> None:
    """Review findings H3/M7: a genuine multi-day gap in the underlying data (as
    produced by a corporate-action blackout drop) must be captured in a
    segment_id column so downstream labeling can avoid bridging it."""
    day1 = pd.date_range("2024-01-10 09:15", periods=3, freq="5min")
    day2 = pd.date_range("2024-01-25 09:15", periods=3, freq="5min")  # 15-day gap
    index = day1.append(day2)
    # Prices stay continuous across the gap (no independent >2% price jump) so the
    # segment break below is attributable only to the elapsed-time gap, not an
    # incidental price-jump detection at the same boundary.
    close = [100.0, 100.1, 100.2, 100.3, 100.4, 100.5]
    fut = _ohlcv(index, close)
    spot = _ohlcv(index, [50.0] * len(index))

    fut_path = tmp_path / "banknifty_fut_5min.parquet"
    spot_path = tmp_path / "banknifty_spot_5min.parquet"
    fut.to_parquet(fut_path)
    spot.to_parquet(spot_path)

    out_path = tmp_path / "assembled.parquet"
    merged = assemble_data_script.assemble(fut_path, spot_path, out_path, resample="5min")

    assert "segment_id" in merged.columns
    seg = merged["segment_id"].to_numpy()
    assert seg.tolist() == [0, 0, 0, 1, 1, 1]
