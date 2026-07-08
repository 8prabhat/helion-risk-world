"""Tests for scripts/_common.py::check_calibration_gate (review finding M13)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
_SPEC = importlib.util.spec_from_file_location("common_script", _ROOT / "scripts" / "_common.py")
assert _SPEC is not None and _SPEC.loader is not None
common_script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(common_script)


def _args(*, calibration_report: str | None, allow_uncalibrated: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        calibration_report=calibration_report,
        allow_uncalibrated=allow_uncalibrated,
    )


def test_no_report_path_proceeds() -> None:
    assert common_script.check_calibration_gate(_args(calibration_report=None)) is True


def test_missing_report_file_proceeds(tmp_path) -> None:
    missing = tmp_path / "no_such_report.json"
    assert common_script.check_calibration_gate(_args(calibration_report=str(missing))) is True


def test_unreadable_report_file_proceeds(tmp_path) -> None:
    bad = tmp_path / "calibration_report.json"
    bad.write_text("not valid json{{{", encoding="utf-8")
    assert common_script.check_calibration_gate(_args(calibration_report=str(bad))) is True


def test_passed_gate_proceeds(tmp_path) -> None:
    report = tmp_path / "calibration_report.json"
    report.write_text(json.dumps({"gate": {"passed": True}}), encoding="utf-8")
    assert common_script.check_calibration_gate(_args(calibration_report=str(report))) is True


def test_failed_gate_blocks_by_default(tmp_path) -> None:
    report = tmp_path / "calibration_report.json"
    report.write_text(json.dumps({"gate": {"passed": False}}), encoding="utf-8")
    assert common_script.check_calibration_gate(_args(calibration_report=str(report))) is False


def test_failed_gate_can_be_overridden(tmp_path) -> None:
    report = tmp_path / "calibration_report.json"
    report.write_text(json.dumps({"gate": {"passed": False}}), encoding="utf-8")
    args = _args(calibration_report=str(report), allow_uncalibrated=True)
    assert common_script.check_calibration_gate(args) is True
