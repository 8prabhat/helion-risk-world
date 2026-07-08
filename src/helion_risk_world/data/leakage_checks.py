"""Leakage guards — hybrid shim (generic guards migrated; project fields stay local).

The generic temporal guards (``LeakageError``, ``assert_point_in_time``,
``assert_label_in_future``) are now the reusable quanthelion implementation
(re-exported here, identical signatures). ``MARKET_FEATURE_NAMES`` and
``PORTFOLIO_FEATURE_NAMES`` are this project's own field registries — they do NOT
belong in quanthelion (project-specific schema) and remain defined here.
``assert_no_portfolio_in_market`` is now a thin wrapper over quanthelion's generic
``assert_no_field_overlap``, supplying this project's ``PORTFOLIO_FEATURE_NAMES``.

Original implementation backed up in .pre_quanthelion_migration_backup/.
"""

from __future__ import annotations

from collections.abc import Iterable

from quanthelion.dataquality.leakage import (
    LeakageError,
    assert_label_in_future,
    assert_no_field_overlap,
    assert_point_in_time,
)

# Canonical field-name registries. Keep in sync with the schemas. PROJECT-SPECIFIC —
# these are helion_risk_world's own market/portfolio schema, not reusable framework data.
MARKET_FEATURE_NAMES: frozenset[str] = frozenset(
    {
        # candle / futures
        "open", "high", "low", "close", "volume", "oi", "d_oi",
        "log_return", "simple_return", "realized_vol", "atr", "volume_zscore",
        "rolling_beta", "rolling_corr", "trend_strength", "range_compression",
        "range_expansion", "gap_size", "time_of_day", "day_of_week",
        "basis", "calendar_spread", "near_next_spread", "rollover",
        # F=19 stabilized feature set
        "hl_range", "open_close_norm", "realized_vol_short", "realized_vol_long",
        "atr_pct", "bb_position", "rsi_14", "momentum_norm", "session_return",
        "high_low_pos", "oi_norm", "d_oi_pct", "tod_sin", "tod_cos",
        "dow_sin", "dow_cos", "rel_log_return",
        # feature/label overhaul Phase 2/3 candle-plane additions (F=19 -> 30)
        "adx_14", "dmi_diff_14", "variance_ratio_20", "vol_ratio_short_long",
        "opening_range_position", "first_15min_return", "breadth", "dispersion",
        "kalman_trend", "kalman_innovation_norm", "kalman_trend_uncertainty",
        # feature/label overhaul Phase 2 futures-plane addition (F=13 -> 14)
        "oi_basis_interaction",
        # regime context additions
        "pc_oi_ratio_c", "basis_daily",
        # feature/label overhaul Phase 0/2 stabilized macro derivatives
        "fii_dii_net_z", "pc_oi_ratio_z", "usdinr_ret_5d", "crude_ret_5d",
        "usdinr_vol", "crude_vol",
        # option surface (ATM-relative)
        "call_oi", "put_oi", "call_d_oi", "put_d_oi", "call_volume", "put_volume",
        "call_iv", "put_iv", "delta", "gamma", "theta", "vega", "moneyness", "dte",
        "pcr", "iv_skew", "gamma_concentration", "call_wall_strength",
        "put_wall_strength", "oi_wall_strength", "max_pain_proxy", "expiry_pressure",
        "atm_iv", "wing_iv",
        # regime / event
        "vix", "vix_pct", "expiry_flag", "event_day_flag", "blackout_active",
        "event_type", "fii_dii_net", "usdinr", "crude",
    }
)

PORTFOLIO_FEATURE_NAMES: frozenset[str] = frozenset(
    {
        "capital0", "capital", "cash", "position", "position_qty", "entry_price",
        "realized_pnl", "unrealized_pnl", "daily_pnl", "drawdown", "margin_used",
        "free_margin", "exposure", "risk_budget_used", "trades_today",
        "consecutive_losses", "net_delta", "net_gamma", "net_theta", "net_vega",
        "expiry_concentration", "strike_concentration", "max_risk_per_trade",
        "max_daily_loss", "max_weekly_loss", "max_drawdown", "max_exposure",
        "max_trades_per_day", "risk_tolerance",
    }
)


def assert_no_portfolio_in_market(feature_names: Iterable[str]) -> None:
    """Raise ``LeakageError`` if any portfolio field name appears in market encoder inputs."""
    assert_no_field_overlap(feature_names, PORTFOLIO_FEATURE_NAMES, context="Market World inputs")


__all__ = [
    "MARKET_FEATURE_NAMES",
    "PORTFOLIO_FEATURE_NAMES",
    "LeakageError",
    "assert_no_portfolio_in_market",
    "assert_point_in_time",
    "assert_label_in_future",
]
