"""Stage 1: triple-barrier labeling (SPEC.md §29).

Builds labels via alpha_data's real triple-barrier pipeline against alpha_data's real
continuous-futures OHLCV (Phase 2 migration) -- see
``helion_risk_world.data.alpha_labels.build_alpha_labels`` for the engine. ``--data-path``
is accepted for backward compatibility with existing invocations/orchestration but is no
longer read: labeling no longer depends on ``scripts/assemble_data.py``'s output (that
script and its local basis/segment assembly are redundant now that alpha_data's own
futures-microstructure/basis pipelines exist and labeling reads the raw continuous
futures parquet directly).

Usage:
    python scripts/label.py \\
        --out-path    data/processed/labels.parquet \\
        [--H 192] [--stop-mult 2.0] [--target-mult 2.0] [--dry-run]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_src_path

ensure_src_path()

from helion_risk_world.data.alpha_labels import build_alpha_labels
from helion_risk_world.integration import get_logger
from helion_risk_world.schemas.label_schema import Barrier

log = get_logger("hrw.label")


def run_labeling(
    data_path: Path | None,
    out_path: Path,
    H: int = 12,
    target_horizons: tuple[int, ...] | None = None,
    stop_mult: float = 2.0,
    target_mult: float = 2.0,
    cost_floor_frac: float | None = None,
    session_exclude_minutes: int = 15,
    symbol: str = "BANKNIFTY_FUT_continuous",
    interval: str = "5min",
    dry_run: bool = False,
):
    """Label alpha_data's real continuous-futures OHLCV and write results.

    Args mirror ``build_alpha_labels`` (see there for the exact algorithm); ``data_path``
    is accepted but unused (kept for CLI/orchestration backward compatibility -- see
    module docstring).
    """
    if data_path is not None:
        log.info("label.data_path_ignored", data_path=str(data_path),
                  note="labeling reads alpha_data's continuous futures parquet directly")

    out_df = build_alpha_labels(
        H=H,
        target_horizons=target_horizons,
        stop_mult=stop_mult,
        target_mult=target_mult,
        cost_floor_frac=cost_floor_frac,
        session_exclude_minutes=session_exclude_minutes,
        symbol=symbol,
        interval=interval,
    )
    log.info(
        "label.records", total=len(out_df),
        n_stop=int((out_df["barrier"] == Barrier.STOP.value).sum()),
        n_target=int((out_df["barrier"] == Barrier.TARGET.value).sum()),
        n_timeout=int((out_df["barrier"] == Barrier.TIMEOUT.value).sum()),
        n_ambiguous=int((out_df["barrier"] == Barrier.AMBIGUOUS.value).sum()),
    )
    if "primary_side" in out_df.columns:
        with_bet = out_df[out_df["primary_side"] != 0]
        log.info(
            "label.meta_labels", total=len(out_df), n_with_bet=len(with_bet),
            pct_with_bet=round(len(with_bet) / max(len(out_df), 1), 4),
            mean_meta_label=(
                round(float(with_bet["meta_label"].mean()), 4) if len(with_bet) else None
            ),
        )

    if dry_run:
        log.info("label.dry_run", out_path=str(out_path), note="skipping write")
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(out_path)
        log.info("label.written", path=str(out_path), rows=len(out_df))

    return out_df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-path", required=False, default=None, type=Path,
        help="Deprecated/unused -- kept for backward compatibility. Labeling now reads "
             "alpha_data's continuous futures parquet directly.",
    )
    p.add_argument("--out-path", required=True, type=Path)
    # Feature/label overhaul Phase 4a: H=192 (~2 trading days) is the primary management
    # horizon a cheap linear/nonlinear diagnostic found real out-of-sample directional
    # signal at, vs. no signal at the prior 15-60 min (H=12) horizon. The short horizons
    # are kept as auxiliary target_horizons so the world model's existing fixed-horizon
    # return/direction regression heads still carry that (more robust, larger-sample)
    # signal even if the longer-horizon barrier classification is noisier.
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
    p.add_argument(
        "--interval", default="5min",
        help="Bar interval to label against, e.g. 5min or 1min. Must match a materialized "
             "{symbol}_{interval}.parquet in alpha_data's OHLCV store.",
    )
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
        symbol=args.symbol, interval=args.interval, dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
