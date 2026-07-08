"""Economically-aware sample weighting for supervised training.

The forecaster's return target remains the realized futures return. This module only changes how
much each supervised row influences optimization so the loss is not dominated by low-opportunity
bars whose realized move does not clear the current execution-cost floor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from helion_risk_world.config.execution_config import CostModelConfig
from helion_risk_world.execution.instrument_specs import resolve_instrument_spec
from helion_risk_world.schemas.label_schema import (
    horizon_mae_column,
    horizon_mfe_column,
    horizon_return_column,
)

_EPS = 1e-9
_WEIGHT_FLOOR = 0.35
_WEIGHT_CAP = 4.0


@dataclass(frozen=True)
class OpportunityWeightAudit:
    """Compact diagnostics for the computed opportunity weights."""

    mean_weight: float
    median_weight: float
    pct_upweighted: float
    pct_downweighted: float
    mean_roundtrip_cost_return: float


def compute_management_opportunity_weights(
    labels: pd.DataFrame,
    *,
    management_horizon: int,
    execution_cfg: CostModelConfig,
    weight_floor: float = _WEIGHT_FLOOR,
    weight_cap: float = _WEIGHT_CAP,
) -> tuple[np.ndarray, OpportunityWeightAudit]:
    """Return opportunity weights aligned to the management horizon.

    Weighting is based only on fields already present in the local label inventory:
      - fixed-horizon realized return
      - fixed-horizon MAE / MFE
      - decision-time entry price
      - configured futures contract metadata and cost assumptions

    The weight is higher when:
      1. the realized move is large relative to the configured round-trip cost floor
      2. the realized path is directionally efficient rather than choppy
    """
    if weight_floor <= 0.0:
        raise ValueError("weight_floor must be positive")
    if weight_cap < weight_floor:
        raise ValueError("weight_cap must be >= weight_floor")

    ret_col = horizon_return_column(management_horizon)
    mae_col = horizon_mae_column(management_horizon)
    mfe_col = horizon_mfe_column(management_horizon)
    required = {"entry_price", ret_col, mae_col, mfe_col}
    missing = sorted(name for name in required if name not in labels.columns)
    if missing:
        raise ValueError(
            "labels are missing required opportunity-weight inputs: "
            f"{', '.join(missing)}"
        )

    entry_price = labels["entry_price"].to_numpy(dtype=np.float64, copy=False)
    realized_return = labels[ret_col].to_numpy(dtype=np.float64, copy=False)
    mae = np.maximum(labels[mae_col].to_numpy(dtype=np.float64, copy=False), 0.0)
    mfe = np.maximum(labels[mfe_col].to_numpy(dtype=np.float64, copy=False), 0.0)
    if not np.all(np.isfinite(entry_price) & (entry_price > 0.0)):
        raise ValueError("entry_price must be finite and positive for opportunity weighting")
    symbols = (
        labels["symbol"].astype(str).to_numpy(copy=False)
        if "symbol" in labels.columns
        else np.repeat("BANKNIFTY_FUT_continuous", len(labels))
    )

    roundtrip_cost = estimate_roundtrip_cost_return(
        entry_price=entry_price,
        symbols=symbols,
        execution_cfg=execution_cfg,
    )
    abs_return = np.abs(realized_return)
    favorable = np.where(realized_return >= 0.0, mfe, mae)
    adverse = np.where(realized_return >= 0.0, mae, mfe)
    directional_efficiency = favorable / np.maximum(favorable + adverse, _EPS)
    path_quality = 0.5 + 0.5 * np.clip(directional_efficiency, 0.0, 1.0)
    economic_ratio = abs_return / np.maximum(roundtrip_cost, _EPS)
    weights = np.clip(
        weight_floor + economic_ratio * path_quality,
        weight_floor,
        weight_cap,
    ).astype(np.float32, copy=False)

    audit = OpportunityWeightAudit(
        mean_weight=float(weights.mean()) if len(weights) else float(weight_floor),
        median_weight=float(np.median(weights)) if len(weights) else float(weight_floor),
        pct_upweighted=float((weights > 1.0).mean()) if len(weights) else 0.0,
        pct_downweighted=float((weights < 1.0).mean()) if len(weights) else 0.0,
        mean_roundtrip_cost_return=float(roundtrip_cost.mean()) if len(roundtrip_cost) else 0.0,
    )
    return weights, audit


def estimate_roundtrip_cost_return(
    *,
    entry_price: np.ndarray,
    symbols: np.ndarray,
    execution_cfg: CostModelConfig,
) -> np.ndarray:
    """Estimate the configured round-trip cost floor as a return fraction."""
    if entry_price.shape[0] != symbols.shape[0]:
        raise ValueError("entry_price and symbols must have the same length")

    out = np.empty(entry_price.shape[0], dtype=np.float64)
    for symbol in np.unique(symbols):
        mask = symbols == symbol
        spec = resolve_instrument_spec(str(symbol), execution_cfg)
        lot_size = spec.lot_size if spec is not None else 1.0
        notional = np.maximum(entry_price[mask] * lot_size, _EPS)
        brokerage = execution_cfg.brokerage_per_order
        exchange = execution_cfg.exchange_txn_rate * notional
        statutory = (
            brokerage
            + execution_cfg.stt_rate * notional
            + exchange
            + execution_cfg.gst_rate * (brokerage + exchange)
            + execution_cfg.sebi_rate * notional
            + execution_cfg.stamp_duty_rate * notional
        )
        one_way = (
            execution_cfg.half_spread_bps * notional
            + execution_cfg.slippage_bps * notional
            + statutory
        ) / notional
        out[mask] = 2.0 * one_way
    return out


__all__ = [
    "OpportunityWeightAudit",
    "compute_management_opportunity_weights",
    "estimate_roundtrip_cost_return",
]
