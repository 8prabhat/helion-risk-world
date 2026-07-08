"""Stage 1: triple-barrier labeling + uniqueness weights (SPEC.md §29).

Reads the assembled dataset produced by assemble_data.py, runs BarrierLabeler on the
futures close series, attaches uniqueness sample weights, and writes a labeled parquet
ready for Stage 2 (encoder pretraining) and Stage 4 (head training).

Usage:
    python scripts/label.py \\
        --data-path   data/processed/banknifty_5min.parquet \\
        --out-path    data/processed/labels.parquet \\
        [--H 12] [--stop-mult 2.0] [--target-mult 2.0] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _bootstrap import ensure_src_path

ensure_src_path()

import numpy as np
import pandas as pd

from helion_risk_world.config.execution_config import CostModelConfig
from helion_risk_world.data.event_calendar import event_type_for
from helion_risk_world.data.primitives import session_boundary_mask, state_regime_label
from quanthelion.calendars.expiry_calendar import monthly_expiry
from quanthelion.transforms.contiguity import contiguous_segment_ids
from helion_risk_world.execution.cost_model import round_trip_cost_frac
from helion_risk_world.integration import get_logger
from helion_risk_world.labeling.barrier_labeler import BarrierLabeler
from helion_risk_world.labeling.uniqueness import apply_uniqueness_weights
from helion_risk_world.schemas.label_schema import (
    BARRIER_COST_FLOOR_COLUMN,
    BARRIER_SIGMA_COLUMN,
    BARRIER_STOP_MULT_COLUMN,
    BARRIER_STOP_RETURN_COLUMN,
    BARRIER_TARGET_MULT_COLUMN,
    BARRIER_TARGET_RETURN_COLUMN,
    BARRIER_VOL_SPAN_COLUMN,
    Barrier,
    horizon_mae_column,
    horizon_mfe_column,
    horizon_realized_at_column,
    horizon_return_column,
    horizon_volatility_column,
)
from helion_risk_world.schemas.market_schema import EventType

log = get_logger("hrw.label")
# Bumped 6->7 for feature/label overhaul Phase 1: cost-floor barrier width +
# session-boundary exclusion change which bars get labeled.
_LABEL_SCHEMA_VERSION = 7


def _point_in_time_regime(close: pd.Series, idx: int, horizon: int) -> str:
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


def _fixed_horizon_stats(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    decision_idx: int,
    horizon: int,
) -> dict[str, object]:
    entry_idx = decision_idx + 1
    exit_idx = decision_idx + horizon
    if exit_idx >= len(close):
        raise ValueError(f"insufficient future bars for fixed horizon {horizon}")
    close_path = close.iloc[entry_idx : exit_idx + 1].to_numpy(dtype=float)
    high_path = high.iloc[entry_idx : exit_idx + 1].to_numpy(dtype=float)
    low_path = low.iloc[entry_idx : exit_idx + 1].to_numpy(dtype=float)
    entry_px = float(open_.iloc[entry_idx])
    realized_return = float(close.iloc[exit_idx] / entry_px - 1.0)
    if len(close_path) > 1:
        realized_vol = float(np.std(np.diff(np.log(close_path))))
    else:
        realized_vol = 0.0
    mae = float(max((entry_px - low_path.min()) / entry_px, 0.0)) if len(low_path) else 0.0
    mfe = float(max((high_path.max() - entry_px) / entry_px, 0.0)) if len(high_path) else 0.0
    return {
        "return": realized_return,
        "vol": max(realized_vol, 1e-6),
        "mae": mae,
        "mfe": mfe,
        "realized_at": close.index[exit_idx],
    }


def run_labeling(
    data_path: Path,
    out_path: Path,
    H: int = 12,
    target_horizons: tuple[int, ...] | None = None,
    stop_mult: float = 2.0,
    target_mult: float = 2.0,
    cost_floor_frac: float | None = None,
    session_exclude_minutes: int = 15,
    symbol: str = "BANKNIFTY_FUT_continuous",
    dry_run: bool = False,
) -> pd.DataFrame:
    """Label the assembled dataset and write results.

    Args:
        data_path:   path to assembled parquet (output of assemble_data.py)
        out_path:    destination parquet path
        H:           trade-management horizon in bars (should match max(horizon_bars))
        target_horizons:
                     fixed-horizon consequence targets to persist for training/evaluation.
                     Defaults to ``(H,)`` and each value must be ``<= H``.
        stop_mult:   stop-barrier width in σ multiples
        target_mult: target-barrier width in σ multiples
        cost_floor_frac:
                     minimum barrier half-width as a return fraction (feature/label
                     overhaul Phase 1) — floors a purely vol-scaled barrier at round-trip
                     transaction cost so sub-cost noise isn't labeled as a real
                     directional win/loss. Defaults to
                     ``round_trip_cost_frac(CostModelConfig())``, this project's own
                     documented cost assumptions; pass 0.0 to restore the old
                     purely-vol-scaled behavior.
        session_exclude_minutes:
                     exclude decision bars in the first/last N minutes of the NSE
                     session (auction overhang / end-of-day squaring) from labeling.
        symbol:      instrument name for LabelRecord
        dry_run:     if True, log the plan but do not write

    Returns:
        DataFrame of labeled records with columns:
          ts, symbol, barrier, exit_return, exit_t, realized_vol, mae, sample_weight, regime
        ``regime`` uses the heuristic from ``primitives.regime_label`` (market-plane, point-in-time).
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

    log.info("label.load", path=str(data_path))
    df = pd.read_parquet(data_path)
    df.index = pd.to_datetime(df.index)
    open_col = "open_fut" if "open_fut" in df.columns else "open"
    high_col = "high_fut" if "high_fut" in df.columns else "high"
    low_col = "low_fut" if "low_fut" in df.columns else "low"
    close_col = "close_fut" if "close_fut" in df.columns else "close"
    missing = [name for name in (open_col, high_col, low_col, close_col) if name not in df.columns]
    if missing:
        raise ValueError(f"assembled label input is missing required futures OHLC columns: {missing}")
    valid = df[[open_col, high_col, low_col, close_col]].dropna()
    if valid.empty:
        raise ValueError("assembled label input has no complete futures OHLC rows")
    open_series = valid[open_col]
    high_series = valid[high_col]
    low_series = valid[low_col]
    close_series = valid[close_col]
    close = close_series.values
    timestamps = close_series.index.to_pydatetime().tolist()

    # Contiguity segment id (review findings H3, H4, M7): dropna() above (and any
    # upstream blackout drop / roll gap from assemble_data.py) can leave
    # positionally-adjacent rows that are not actually adjacent bars. Recompute
    # fresh on the post-dropna index — the assembled parquet's own segment_id (if
    # present) is not reused as-is since dropna() here can introduce new gaps of
    # its own beyond whatever assemble_data.py already accounted for.
    roll_gap_mask = None
    if "roll_gap" in df.columns:
        roll_gap_mask = df["roll_gap"].reindex(valid.index, fill_value=False).to_numpy(dtype=bool)
    segment_id = contiguous_segment_ids(valid.index, extra_gap_mask=roll_gap_mask)
    boundary_mask = session_boundary_mask(timestamps, exclude_minutes=session_exclude_minutes)
    log.info(
        "label.input",
        rows=len(close),
        H=H,
        target_horizons=horizons,
        stop_mult=stop_mult,
        target_mult=target_mult,
        cost_floor_frac=resolved_cost_floor,
        session_exclude_minutes=session_exclude_minutes,
    )

    labeler = BarrierLabeler(
        H=H, u=target_mult, d=stop_mult, cost_floor=resolved_cost_floor, add_uniqueness=False
    )
    records = labeler.label(
        timestamps,
        close.tolist(),
        symbol=symbol,
        open_prices=open_series.to_list(),
        high_prices=high_series.to_list(),
        low_prices=low_series.to_list(),
    )
    records = apply_uniqueness_weights(records)
    log.info("label.records", total=len(records),
             n_stop=sum(1 for r in records if r.barrier == Barrier.STOP),
             n_target=sum(1 for r in records if r.barrier == Barrier.TARGET),
             n_timeout=sum(1 for r in records if r.barrier == Barrier.TIMEOUT),
             n_ambiguous=sum(1 for r in records if r.barrier == Barrier.AMBIGUOUS))

    rows = []
    n_skipped_gap = 0
    n_skipped_boundary = 0
    for idx, rec in enumerate(records):
        # Skip any record whose barrier scan or fixed-horizon window crosses a
        # contiguity gap (review findings H3, H4, M7) — e.g. a corporate-action
        # blackout drop or a futures roll — rather than silently treating a
        # positionally-adjacent-but-not-really-adjacent bar as the real entry/exit.
        # Uniqueness weights were already computed above over the full, unfiltered
        # record sequence (matching compute_uniqueness's position-based
        # concurrency assumption); only the final written rows are filtered here.
        exit_i = idx + rec.exit_t
        crosses_gap = segment_id[idx] != segment_id[exit_i]
        if not crosses_gap:
            for horizon in horizons:
                if segment_id[idx] != segment_id[idx + horizon]:
                    crosses_gap = True
                    break
        if crosses_gap:
            n_skipped_gap += 1
            continue
        # Skip decision bars in the excluded opening/closing session window (feature/
        # label overhaul Phase 1) — same additive-skip pattern as the gap check above.
        if boundary_mask[idx]:
            n_skipped_boundary += 1
            continue
        reg = _point_in_time_regime(close_series, idx, H)
        row = {
            "ts": rec.ts,
            "decision_ts": rec.ts,
            "symbol": rec.symbol,
            "label_realized_at": rec.label_realized_at,
            "horizon_bars": rec.horizon_bars,
            "barrier": rec.barrier.value,
            "barrier_valid": bool(rec.barrier_valid),
            "entry_price": rec.entry_price,
            "exit_price": rec.exit_price,
            "exit_return": rec.exit_return,
            "exit_t": rec.exit_t,
            "exit_bars": rec.exit_t,
            "realized_vol": rec.realized_vol,
            BARRIER_SIGMA_COLUMN: rec.barrier_sigma,
            BARRIER_STOP_RETURN_COLUMN: rec.barrier_stop_return,
            BARRIER_TARGET_RETURN_COLUMN: rec.barrier_target_return,
            BARRIER_STOP_MULT_COLUMN: rec.barrier_stop_mult,
            BARRIER_TARGET_MULT_COLUMN: rec.barrier_target_mult,
            BARRIER_VOL_SPAN_COLUMN: rec.barrier_vol_span,
            BARRIER_COST_FLOOR_COLUMN: rec.barrier_cost_floor_frac,
            "mae": rec.mae,
            "mfe": rec.mfe,
            "sample_weight": float(rec.uniqueness_weight or 0.0),
            "sample_weight_source": "uniqueness",
            "regime": reg,
            "regime_source": "point_in_time",
            "label_schema_version": _LABEL_SCHEMA_VERSION,
        }
        for horizon in horizons:
            stats = _fixed_horizon_stats(
                open_series,
                high_series,
                low_series,
                close_series,
                decision_idx=idx,
                horizon=horizon,
            )
            row[horizon_return_column(horizon)] = float(stats["return"])
            row[horizon_volatility_column(horizon)] = float(stats["vol"])
            row[horizon_mae_column(horizon)] = float(stats["mae"])
            row[horizon_mfe_column(horizon)] = float(stats["mfe"])
            row[horizon_realized_at_column(horizon)] = stats["realized_at"]
        rows.append(row)

    if n_skipped_gap:
        # NOTE: get_logger() returns a plain stdlib logging.Logger; .warning() is
        # enabled by default (unlike the .info() calls elsewhere in this file), so
        # this must use %-style args, not **kwargs, or it would raise TypeError.
        log.warning("label.gap_crossing_rows_skipped total=%s", n_skipped_gap)
    if n_skipped_boundary:
        log.warning("label.session_boundary_rows_skipped total=%s", n_skipped_boundary)
    if not rows:
        raise ValueError("no labels survived contiguity filtering — check upstream data gaps")
    out_df = pd.DataFrame(rows).set_index("ts")
    log.info("label.summary", rows=len(out_df),
             mean_weight=float(out_df["sample_weight"].mean()),
             min_weight=float(out_df["sample_weight"].min()))

    if dry_run:
        log.info("label.dry_run", out_path=str(out_path), note="skipping write")
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(out_path)
        log.info("label.written", path=str(out_path), rows=len(out_df))

    return out_df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", required=True, type=Path)
    p.add_argument("--out-path", required=True, type=Path)
    # Feature/label overhaul Phase 4a: H=192 (~2 trading days) is the new primary
    # management horizon a cheap linear/nonlinear diagnostic found real out-of-sample
    # directional signal at, vs. no signal at the prior 15-60 min (H=12) horizon. The
    # short horizons are kept as auxiliary target_horizons so the world model's existing
    # fixed-horizon return/direction regression heads still carry that (more robust,
    # larger-sample) signal even if the longer-horizon barrier classification is noisier.
    p.add_argument("--H", type=int, default=192)
    p.add_argument(
        "--target-horizons",
        type=int,
        nargs="+",
        default=[3, 6, 12, 48, 96, 192],
        help="Fixed consequence horizons to persist, e.g. --target-horizons 3 6 12",
    )
    p.add_argument("--stop-mult", type=float, default=2.0)
    p.add_argument("--target-mult", type=float, default=2.0)
    p.add_argument(
        "--cost-floor-frac",
        type=float,
        default=None,
        help=(
            "Minimum barrier half-width as a return fraction, floored at round-trip "
            "transaction cost. Defaults to round_trip_cost_frac(CostModelConfig()) "
            "(this project's own documented cost assumptions); pass 0.0 to restore the "
            "old purely-vol-scaled behavior."
        ),
    )
    p.add_argument(
        "--session-exclude-minutes",
        type=int,
        default=15,
        help="Exclude decision bars in the first/last N minutes of the NSE session.",
    )
    p.add_argument("--symbol", default="BANKNIFTY_FUT_continuous")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    run_labeling(
        args.data_path, args.out_path,
        H=args.H,
        target_horizons=tuple(args.target_horizons) if args.target_horizons else None,
        stop_mult=args.stop_mult,
        target_mult=args.target_mult,
        cost_floor_frac=args.cost_floor_frac,
        session_exclude_minutes=args.session_exclude_minutes,
        symbol=args.symbol, dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
