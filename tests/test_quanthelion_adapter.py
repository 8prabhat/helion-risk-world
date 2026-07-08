"""Quanthelion integration adapter (SPEC.md §6, §27)."""
from __future__ import annotations

from helion_risk_world.integration import quanthelion_adapter as qa


def test_get_logger_returns_a_logger() -> None:
    log = qa.get_logger("hrw.test")
    assert hasattr(log, "info")


def test_dataset_protocol_is_runtime_checkable() -> None:
    from helion_risk_world.data.dataset import HRWWindowDataset

    ds = HRWWindowDataset(index=[1, 2, 3])
    assert isinstance(ds, qa.DatasetProtocol)
    assert len(ds) == 3


def test_adapter_exposes_assumed_but_absent_symbols_locally() -> None:
    # These do NOT exist in quanthelion; the adapter provides them (SPEC.md §6.2).
    assert hasattr(qa, "TrainerAdapter")
    assert hasattr(qa, "ExperimentRunnerAdapter")
    assert hasattr(qa, "DatasetProtocol")
