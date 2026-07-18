"""Script-level smoke tests for scripts/label.py (Phase 2 migration).

label.py is now a thin CLI wrapper over helion_risk_world.data.alpha_labels.
build_alpha_labels, which always reads alpha_data's real continuous futures OHLCV
(--data-path is accepted but ignored, kept only for orchestration/CLI backward
compatibility -- see label.py's own docstring). Detailed synthetic-scenario regression
tests (ambiguous ties, cost-floor timeouts, session-boundary/gap exclusion) live in
tests/test_alpha_labels.py, which can inject exact synthetic OHLCV via
build_alpha_labels's raw_ohlcv test seam; this file only checks the script/CLI plumbing
itself.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
_SPEC = importlib.util.spec_from_file_location("label_script", _ROOT / "scripts" / "label.py")
assert _SPEC is not None and _SPEC.loader is not None
label_script = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = label_script
_SPEC.loader.exec_module(label_script)


def test_run_labeling_writes_output_and_ignores_data_path(tmp_path) -> None:
    """A --data-path pointing at a file that doesn't even exist must not raise --
    proves the value is genuinely unused, not just unread by coincidence."""
    out_path = tmp_path / "labels.parquet"
    bogus_data_path = tmp_path / "does_not_exist.parquet"

    labels = label_script.run_labeling(
        bogus_data_path, out_path,
        H=3, target_horizons=(3,), stop_mult=2.0, target_mult=2.0,
        session_exclude_minutes=15,
    )

    assert out_path.exists()
    assert len(labels) > 0


def test_run_labeling_dry_run_skips_write(tmp_path) -> None:
    out_path = tmp_path / "labels_dry.parquet"
    label_script.run_labeling(
        None, out_path,
        H=3, target_horizons=(3,), stop_mult=2.0, target_mult=2.0,
        session_exclude_minutes=15, dry_run=True,
    )
    assert not out_path.exists()


def test_run_labeling_output_schema_matches_helion_contract(tmp_path) -> None:
    out_path = tmp_path / "labels.parquet"
    labels = label_script.run_labeling(
        None, out_path, H=3, target_horizons=(3,), stop_mult=2.0, target_mult=2.0,
    )
    for col in (
        "symbol", "decision_ts", "label_realized_at",
        "horizon_bars", "barrier", "barrier_valid", "entry_price", "exit_price",
        "exit_return", "exit_t", "exit_bars", "realized_vol", "mae", "mfe",
        "sample_weight", "sample_weight_source", "regime", "regime_source",
        "label_schema_version",
    ):
        assert col in labels.columns, col
    assert isinstance(labels.index, pd.DatetimeIndex)
    assert labels.index.name == "ts"


def test_main_cli_smoke(tmp_path, monkeypatch) -> None:
    out_path = tmp_path / "labels_cli.parquet"
    argv = [
        "label.py",
        "--data-path", str(tmp_path / "unused.parquet"),
        "--out-path", str(out_path),
        "--H", "3",
        "--target-horizons", "3",
        "--stop-mult", "2.0",
        "--target-mult", "2.0",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    label_script.main()
    assert out_path.exists()
