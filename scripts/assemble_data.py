"""Stage 0: assemble raw parquets into a clean, validated dataset (SPEC.md §29).

Loads BANKNIFTY futures (continuous) and spot parquets, resamples to 5-min bars,
computes basis, flags corporate-action blackout bars and roll gaps, asserts no leakage,
and writes a merged parquet ready for Stage 1 (labeling).

Data is fetched via Upstox before running this script:
    python scripts/fetch_upstox.py --from 2023-01-01 --to 2024-12-31 --interval 5min

Usage:
    python scripts/assemble_data.py \\
        --futures-path  data/ohlcv/BANKNIFTY_FUT_continuous_5min.parquet \\
        --spot-path     data/ohlcv/BANKNIFTY_5min.parquet \\
        --out-path      data/processed/banknifty_5min.parquet \\
        [--resample 5min] [--dry-run]

If you fetched 1-min data (--interval 1min), pass --resample 5min and point to
the 1-min parquets; the script will resample automatically.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _bootstrap import ensure_src_path

ensure_src_path()

import numpy as np
import pandas as pd

from helion_risk_world.data.corporate_actions import flag_merger_bars
from quanthelion.transforms.contiguity import contiguous_segment_ids
from helion_risk_world.data.leakage_checks import assert_no_portfolio_in_market
from helion_risk_world.data.parquet_source import (
    infer_interval_from_path,
    load_ohlcv_parquet,
    prepare_ohlcv_frame,
)
from quanthelion.transforms.rollover import count_roll_gaps, flag_and_clip
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("hrw.assemble")

def _load_parquet(path: Path, label: str, target_interval: str) -> pd.DataFrame:
    source_interval = infer_interval_from_path(path, fallback=target_interval)
    log.info(f"loading {label} from {path} [{source_interval} -> {target_interval}]")
    df = load_ohlcv_parquet(path)
    log.info(f"loaded {label}: rows={len(df)}  {df.index.min()} → {df.index.max()}")
    return prepare_ohlcv_frame(
        df,
        source_interval=source_interval,
        target_interval=target_interval,
    )


def assemble(
    futures_path: Path,
    spot_path: Path,
    out_path: Path,
    resample: str = "5min",
    dry_run: bool = False,
) -> pd.DataFrame:
    """Load, validate, merge, and save the dataset."""
    # 1. Load
    fut5 = _load_parquet(futures_path, "futures", resample)
    spot5 = _load_parquet(spot_path, "spot", resample)
    log.info(f"resampled to {resample}: futures={len(fut5)} spot={len(spot5)}")

    # 2. Preserve continuous_futures.py's authoritative per-contract-roll marker
    # (review finding H4). flag_and_clip() below recomputes its OWN roll_gap from a
    # price-jump heuristic and would otherwise silently REPLACE this column outright
    # whenever it fires — which, after backward price adjustment, is rare, masking
    # the fact that the authoritative per-roll marker was wiped even when it had
    # nothing to do with whatever triggered the price-jump check.
    contract_roll_flag = (
        fut5["roll_gap"].astype(bool).copy()
        if "roll_gap" in fut5.columns
        else pd.Series(False, index=fut5.index)
    )

    # 3. Detect and remove price-jump roll gaps in futures
    n_gaps = count_roll_gaps(fut5)
    if n_gaps > 0:
        log.warning(f"roll gaps detected: {n_gaps} bars — NaN-flagged and forward-filled")
        fut5 = flag_and_clip(fut5).ffill()
        price_jump_flag = fut5["roll_gap"].astype(bool)
    else:
        log.info("no roll gaps found")
        price_jump_flag = pd.Series(False, index=fut5.index)

    # Combine: a bar is a genuine roll discontinuity if EITHER signal fired (H4) —
    # previously only the (usually-silent, post-adjustment) price-jump detector's
    # result survived, discarding the authoritative contract-roll flag entirely.
    fut5["roll_gap"] = (
        contract_roll_flag.reindex(fut5.index, fill_value=False).to_numpy()
        | price_jump_flag.reindex(fut5.index, fill_value=False).to_numpy()
    )

    # 4. Flag corporate action bars (HDFC merger 2023-07-01) — both series share the date
    fut5 = fut5.reset_index().rename(columns={"index": "ts", fut5.index.name: "ts"})
    merger_mask = flag_merger_bars(fut5, date_col="ts")
    n_merger = int(merger_mask.sum())
    if n_merger > 0:
        log.warning(f"merger blackout bars flagged: {n_merger} (HDFC merger 2023-07-01)")
        fut5 = fut5[~merger_mask]
    fut5 = fut5.set_index("ts")

    # 5. Inner-join on timestamp — keeps only bars present in both series
    merged = fut5.join(spot5, how="inner", lsuffix="_fut", rsuffix="_spot")
    log.info(f"merged rows: {len(merged)}")

    # 6. Compute basis = (futures_close - spot_close) / spot_close
    merged["basis"] = (merged["close_fut"] - merged["close_spot"]) / merged["close_spot"].replace(0, np.nan)

    # 6b. Contiguity segment id (review findings H3, H4, M7): both the merger-
    # blackout drop (step 4) and the inner join (step 5) can silently remove rows,
    # leaving positionally-adjacent rows that are NOT actually adjacent bars.
    # Persist a segment id so downstream labeling (scripts/label.py,
    # BarrierLabeler) can detect and skip any window that would otherwise silently
    # bridge one of these gaps.
    merged["segment_id"] = contiguous_segment_ids(
        merged.index, extra_gap_mask=merged["roll_gap"].to_numpy()
    )

    # 7. Leakage guard — block any accidental portfolio/account field names
    assert_no_portfolio_in_market(merged.columns)
    log.info(f"leakage check passed: {sorted(map(str, merged.columns))}")

    # 8. Write output
    if dry_run:
        log.info(f"dry-run: would write {len(merged)} rows to {out_path}")
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(out_path)
        log.info(f"written {len(merged)} rows → {out_path}")

    return merged


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--futures-path", required=True, type=Path)
    p.add_argument("--spot-path", required=True, type=Path)
    p.add_argument("--out-path", required=True, type=Path)
    p.add_argument("--resample", default="5min")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    assemble(args.futures_path, args.spot_path, args.out_path,
             resample=args.resample, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
