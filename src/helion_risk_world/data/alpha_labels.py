"""Compatibility adapter: builds helion's exact ``labels.parquet`` schema by calling
alpha_data's real triple-barrier labeling pipeline against alpha_data's real continuous-
futures OHLCV, instead of computing labels locally via ``labeling/barrier_labeler.py`` +
``labeling/uniqueness.py`` over ``scripts/assemble_data.py``'s now-redundant local
basis/segment assembly (Phase 2 migration).

``alpha_data.pipelines.labels.build_labels`` (extended this same phase with
``entry_offset``/``horizon_targets``) already reproduces ``BarrierLabeler``'s asymmetric-
barrier/EWMA-sigma/ambiguous-tie/MAE-MFE/next-bar-open-entry scheme exactly -- this module
is a read-and-reshape shim (same pattern as ``alpha_features.py``/
``alpha_futures_features.py``), not a reimplementation. What it adds on top of
``build_labels``'s output:

- ``barrier_sigma``/``barrier_stop_return``/``barrier_target_return``: not emitted by
  ``build_labels`` itself, so computed via a direct ``barrier_context_series`` call
  (the same underlying quanthelion primitive ``BarrierLabeler`` used) with a matching
  ``BarrierSpec`` -- exact parity, not an approximation.
- ``realized_vol``: std of log-diffs over each row's own realized ``[entry_i, exit_i]``
  path (BarrierLabeler's exact formula) -- not a barrier-engine concept, recomputed here
  directly from the raw close array using ``global_exit_idx``.
- Column renaming/enum mapping (``tb_hit_barrier`` -> ``Barrier``, ``tb_*`` -> helion's
  schema names) and the same session/segment-gap exclusion filtering
  ``scripts/label.py`` applied before writing.

Sample weighting: alpha_data's ``combined_sample_weights`` (AFML uniqueness x
return-attribution x time-decay) supersedes ``labeling/uniqueness.py``'s simpler
``mean(1/concurrency)`` scheme -- an accepted, deliberate upgrade (same precedent as the
rest of this migration), not ported.

Timestamp convention note: alpha_data's continuous-futures parquet is tz-aware
Asia/Kolkata. Session-boundary exclusion and point-in-time regime/event classification
need REAL IST wall-clock hours and calendar dates, so those are computed from the
tz-aware source index directly (before any conversion -- a tz-aware Timestamp's
``.hour``/``.date()`` already reflect its own local time correctly, no manual offset
needed). The engine call and the final written index use ``_to_naive_utc`` (matching
``AlphaDataMarketWindowBuilder``/``AlphaDataFuturesWindowBuilder``'s established
convention this same phase), so ``labels.parquet``'s index lines up with the rest of the
now-migrated feature pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from quanthelion.calendars.expiry_calendar import monthly_expiry
from quanthelion.labels.barrier_context import BarrierSpec, barrier_context_series
from quanthelion.transforms.contiguity import contiguous_segment_ids

from alpha_data.io.paths import DataPaths as AlphaDataPaths
from alpha_data.pipelines.labels import LabelConfig, build_labels

from helion_risk_world.config.execution_config import CostModelConfig
from helion_risk_world.data.alpha_futures_features import _to_naive_utc
from helion_risk_world.data.event_calendar import event_type_for
from helion_risk_world.data.primitives import session_boundary_mask, state_regime_label
from helion_risk_world.execution.cost_model import round_trip_cost_frac
from helion_risk_world.integration import get_logger
from helion_risk_world.labeling.meta_labels import meta_label_for_side, primary_side_from_close
from helion_risk_world.schemas.label_schema import (
    BARRIER_COST_FLOOR_COLUMN,
    BARRIER_SIGMA_COLUMN,
    BARRIER_STOP_MULT_COLUMN,
    BARRIER_STOP_RETURN_COLUMN,
    BARRIER_TARGET_MULT_COLUMN,
    BARRIER_TARGET_RETURN_COLUMN,
    BARRIER_VOL_SPAN_COLUMN,
    META_LABEL_COLUMN,
    PRIMARY_SIDE_COLUMN,
    Barrier,
    horizon_mae_column,
    horizon_mfe_column,
    horizon_realized_at_column,
    horizon_return_column,
    horizon_volatility_column,
)
from helion_risk_world.schemas.market_schema import EventType

log = get_logger("hrw.alpha_labels")

# Bumped 7->8: labels now sourced from alpha_data's build_labels (Phase 2 migration)
# rather than local BarrierLabeler + uniqueness.py computation over
# scripts/assemble_data.py's output. sample_weight_source changed accordingly
# (combined_sample_weights supersedes the old uniqueness-only scheme).
# Bumped 8->9 (2026-07-18): added primary_side/meta_label columns (cost-aware
# meta-labeling, see labeling/meta_labels.py) -- additive, existing consumers of
# columns 1-8's schema are unaffected, but any code asserting an exact column set
# needs updating.
LABEL_SCHEMA_VERSION = 9

_HIT_BARRIER_TO_ENUM = {
    "upper": Barrier.TARGET,
    "lower": Barrier.STOP,
    "vertical": Barrier.TIMEOUT,
    "ambiguous": Barrier.AMBIGUOUS,
    "both": Barrier.AMBIGUOUS,  # unreachable in practice: ambiguous_as_distinct=True always set
}


def _point_in_time_regime(close: pd.Series, idx: int, horizon: int) -> str:
    """Identical to scripts/label.py's helper -- a pure function of price data only.

    ``close`` must be indexed by the tz-aware (Asia/Kolkata) source timestamps so
    ``ts.date()`` reflects the real IST calendar date for event/expiry lookups.
    """
    start = max(0, idx - horizon + 1)
    window = close.iloc[start : idx + 1].to_numpy(dtype=float)
    trailing_return = float(window[-1] / window[0] - 1.0) if len(window) > 1 else 0.0
    if len(window) > 2:
        trailing_vol = float(np.std(np.diff(np.log(window))))
    else:
        trailing_vol = 0.0
    ts = close.index[idx]
    dt = ts.date()
    event = event_type_for(dt) != EventType.NONE or dt == monthly_expiry(dt.year, dt.month)
    return state_regime_label(trailing_return, trailing_vol, event=event).value


def _realized_vol_over_path(
    close: np.ndarray, entry_idx: np.ndarray, exit_idx: np.ndarray
) -> np.ndarray:
    """std of log-diffs of ``close`` over ``[entry_idx[i], exit_idx[i]]`` inclusive, per
    row -- BarrierLabeler's exact realized_vol formula (0.0 if path length <= 1 bar)."""
    log_close = np.log(close)
    out = np.zeros(len(entry_idx), dtype=np.float64)
    for i in range(len(entry_idx)):
        a, b = int(entry_idx[i]), int(exit_idx[i])
        if b > a:
            out[i] = float(np.std(np.diff(log_close[a : b + 1])))
    return out


def build_alpha_labels(
    *,
    underlying: str = "BANKNIFTY",
    interval: str = "5min",
    H: int = 192,
    target_horizons: tuple[int, ...] | None = None,
    stop_mult: float = 2.0,
    target_mult: float = 2.0,
    vol_span: int = 50,
    cost_floor_frac: float | None = None,
    meta_label_lookback: int = 12,
    session_exclude_minutes: int = 15,
    symbol: str = "BANKNIFTY_FUT_continuous",
    paths: AlphaDataPaths | None = None,
    raw_ohlcv: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build helion's exact ``labels.parquet`` schema from alpha_data's real continuous
    futures OHLCV.

    Same parameters and defaults as ``scripts/label.py::run_labeling``, minus
    ``data_path``/``out_path``/``dry_run`` (this function returns a DataFrame; the
    caller decides whether/where to write it).

    ``meta_label_lookback``: bars of trailing momentum the meta-labeling primary
    signal uses (default 12 = 1 hour at 5-min bars, see labeling/meta_labels.py).
    Produces the ``primary_side``/``meta_label`` columns alongside the existing
    triple-barrier ones.

    ``raw_ohlcv``: test-only seam. When provided, skips reading alpha_data's real
    parquet and uses this frame directly (must have ``open``/``high``/``low``/``close``
    columns; index may be tz-naive, treated as already-local wall-clock the same way a
    tz-aware Asia/Kolkata index would be after conversion) -- lets tests construct exact
    synthetic scenarios (ambiguous ties, cost-floor timeouts, session-boundary/gap
    exclusion) the way ``scripts/label.py``'s pre-migration tests did via ``--data-path``.
    Production callers should never pass this.
    """
    horizons = tuple(sorted(set(int(h) for h in (target_horizons or (H,)))))
    if not horizons:
        raise ValueError("target_horizons must not be empty")
    if min(horizons) < 1:
        raise ValueError("target_horizons must be >= 1")
    if max(horizons) > H:
        raise ValueError("target_horizons cannot exceed the triple-barrier horizon H")
    resolved_cost_floor = (
        cost_floor_frac if cost_floor_frac is not None else round_trip_cost_frac(CostModelConfig())
    )

    if raw_ohlcv is not None:
        raw = raw_ohlcv.sort_index()
    else:
        paths = paths or AlphaDataPaths()
        path = paths.ohlcv / f"{underlying}_FUT_continuous_{interval}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"build_alpha_labels: {path} not found.")
        raw = pd.read_parquet(path).sort_index()

    # Real IST wall-clock, taken from the tz-aware source BEFORE any conversion --
    # session_boundary_mask/_point_in_time_regime need genuine local hours/dates.
    close_local = raw["close"]
    boundary_mask = session_boundary_mask(
        raw.index.to_pydatetime().tolist(), exclude_minutes=session_exclude_minutes
    )

    df = raw.copy()
    df.index = _to_naive_utc(df.index)

    cfg = LabelConfig(
        horizon_bars=H,
        target_vol_k=target_mult,
        stop_vol_k=stop_mult,
        sigma_mode="ewma",
        vol_window=vol_span,
        ambiguous_as_distinct=True,
        track_mae_mfe=True,
        entry_offset=1,
        cost_frac=resolved_cost_floor,
        horizon_targets=list(horizons),
        # H=192 (~2.5 sessions) spans multiple trading days -- build_labels's default
        # day-grouped barrier walk would cap every scan at whatever bars remain until
        # day-end (forcing near-universal spurious timeouts) and reset the EWMA sigma
        # estimator daily. helion's own session/segment-gap exclusion is applied
        # separately below (segment_id/boundary_mask), so day-grouping here is both
        # unnecessary and actively wrong for a multi-day horizon.
        session_scoped=False,
    )
    labeled = build_labels(df, cfg)

    spec = BarrierSpec(
        stop_mult=stop_mult, target_mult=target_mult, vol_span=vol_span,
        cost_floor_frac=resolved_cost_floor, horizon_bars=H,
    )
    barrier_rows = barrier_context_series(labeled["close"].to_numpy(), spec=spec)

    roll_gap_mask = (
        labeled["roll_gap"].to_numpy(dtype=bool) if "roll_gap" in labeled.columns else None
    )
    segment_id = contiguous_segment_ids(labeled.index, extra_gap_mask=roll_gap_mask)

    n = len(labeled)
    global_exit_idx = labeled["global_exit_idx"].to_numpy(dtype=np.int64)
    entry_idx_arr = np.arange(n) + 1
    close_arr = labeled["close"].to_numpy()
    realized_vol = _realized_vol_over_path(close_arr, entry_idx_arr, global_exit_idx)

    log.info(
        "alpha_labels.input", rows=n, H=H, target_horizons=horizons,
        stop_mult=stop_mult, target_mult=target_mult, cost_floor_frac=resolved_cost_floor,
        session_exclude_minutes=session_exclude_minutes,
    )

    rows = []
    n_skipped_bounds = 0
    n_skipped_gap = 0
    n_skipped_boundary = 0
    for idx in range(n):
        if idx + H >= n:
            n_skipped_bounds += 1
            continue
        exit_i = int(global_exit_idx[idx])
        crosses_gap = segment_id[idx] != segment_id[exit_i]
        if not crosses_gap:
            for horizon in horizons:
                if segment_id[idx] != segment_id[idx + horizon]:
                    crosses_gap = True
                    break
        if crosses_gap:
            n_skipped_gap += 1
            continue
        if boundary_mask[idx]:
            n_skipped_boundary += 1
            continue

        hit = labeled["tb_hit_barrier"].iat[idx]
        barrier_enum = _HIT_BARRIER_TO_ENUM[hit]
        entry_price = float(labeled["tb_entry_price"].iat[idx])
        exit_price = float(labeled["tb_exit_price"].iat[idx])
        exit_return = exit_price / entry_price - 1.0
        reg = _point_in_time_regime(close_local, idx, H)
        primary_side = primary_side_from_close(close_arr, idx, lookback=meta_label_lookback)
        meta_label = meta_label_for_side(primary_side, exit_return, resolved_cost_floor)
        row = {
            "ts": labeled.index[idx],
            "decision_ts": labeled.index[idx],
            "symbol": symbol,
            "label_realized_at": labeled.index[exit_i],
            "horizon_bars": H,
            "barrier": barrier_enum.value,
            "barrier_valid": bool(labeled["tb_barrier_valid"].iat[idx]),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "exit_return": exit_return,
            "exit_t": int(labeled["tb_bars_to_exit"].iat[idx]),
            "exit_bars": int(labeled["tb_bars_to_exit"].iat[idx]),
            "realized_vol": float(realized_vol[idx]),
            BARRIER_SIGMA_COLUMN: float(barrier_rows[idx, 0]),
            BARRIER_STOP_RETURN_COLUMN: float(barrier_rows[idx, 1]),
            BARRIER_TARGET_RETURN_COLUMN: float(barrier_rows[idx, 2]),
            BARRIER_STOP_MULT_COLUMN: stop_mult,
            BARRIER_TARGET_MULT_COLUMN: target_mult,
            BARRIER_VOL_SPAN_COLUMN: vol_span,
            BARRIER_COST_FLOOR_COLUMN: resolved_cost_floor,
            PRIMARY_SIDE_COLUMN: primary_side,
            META_LABEL_COLUMN: meta_label,
            "mae": float(labeled["tb_mae"].iat[idx]),
            "mfe": float(labeled["tb_mfe"].iat[idx]),
            "sample_weight": float(labeled["sample_weight"].iat[idx]),
            "sample_weight_source": "combined_sample_weights",
            "regime": reg,
            "regime_source": "point_in_time",
            "label_schema_version": LABEL_SCHEMA_VERSION,
        }
        for horizon in horizons:
            row[horizon_return_column(horizon)] = float(labeled[f"horizon_return_{horizon}"].iat[idx])
            row[horizon_volatility_column(horizon)] = float(labeled[f"horizon_vol_{horizon}"].iat[idx])
            row[horizon_mae_column(horizon)] = float(labeled[f"horizon_mae_{horizon}"].iat[idx])
            row[horizon_mfe_column(horizon)] = float(labeled[f"horizon_mfe_{horizon}"].iat[idx])
            row[horizon_realized_at_column(horizon)] = labeled[f"horizon_realized_at_{horizon}"].iat[idx]
        rows.append(row)

    if n_skipped_gap:
        log.warning("alpha_labels.gap_crossing_rows_skipped total=%s", n_skipped_gap)
    if n_skipped_boundary:
        log.warning("alpha_labels.session_boundary_rows_skipped total=%s", n_skipped_boundary)
    if not rows:
        raise ValueError("no labels survived contiguity filtering — check upstream data gaps")
    out_df = pd.DataFrame(rows).set_index("ts")
    log.info(
        "alpha_labels.summary", rows=len(out_df),
        mean_weight=float(out_df["sample_weight"].mean()),
        min_weight=float(out_df["sample_weight"].min()),
    )
    return out_df


__all__ = ["LABEL_SCHEMA_VERSION", "build_alpha_labels"]
