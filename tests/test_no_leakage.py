"""Leakage invariants (SPEC.md §5, §27). These MUST pass — they are the project's safety spine."""

from __future__ import annotations

import pathlib

import pytest

from helion_risk_world.data.leakage_checks import (
    MARKET_FEATURE_NAMES,
    PORTFOLIO_FEATURE_NAMES,
    LeakageError,
    assert_label_in_future,
    assert_no_portfolio_in_market,
)

_PKG_ROOT = pathlib.Path(__file__).resolve().parents[1] / "src" / "helion_risk_world"


def test_market_and_portfolio_feature_sets_are_disjoint() -> None:
    assert MARKET_FEATURE_NAMES.isdisjoint(PORTFOLIO_FEATURE_NAMES)


def test_portfolio_fields_absent_from_market_encoder_inputs() -> None:
    clean = ["close", "volume", "atr", "pcr", "iv_skew", "vix"]
    assert_no_portfolio_in_market(clean)  # must not raise

    for portfolio_field in ("capital", "drawdown", "margin_used", "risk_budget_used"):
        with pytest.raises(LeakageError):
            assert_no_portfolio_in_market([*clean, portfolio_field])


def test_label_must_be_in_the_future() -> None:
    from datetime import datetime, timedelta

    t = datetime(2026, 6, 25, 10, 0)
    assert_label_in_future(t, t + timedelta(minutes=15))  # ok
    with pytest.raises(LeakageError):
        assert_label_in_future(t, t)
    with pytest.raises(LeakageError):
        assert_label_in_future(t, t - timedelta(minutes=5))


def test_no_msh_jepa_import_anywhere_in_package() -> None:
    """HRW must never import msh_jepa (SPEC.md §4)."""
    offenders: list[str] = []
    for py in _PKG_ROOT.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        allowed = "never imports msh_jepa" in text or "msh_jepa internals" in text
        if "msh_jepa" in text and not allowed:
            # allow doc mentions that explicitly say we do NOT import it
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith(("import msh_jepa", "from msh_jepa")):
                    offenders.append(f"{py}: {stripped}")
    assert not offenders, f"msh_jepa imports found: {offenders}"
