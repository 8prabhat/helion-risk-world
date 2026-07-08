"""Load a saved HRW model artifact and emit one prediction as JSON."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from _bootstrap import ensure_src_path

ensure_src_path()

from train import _demo_candles

from helion_risk_world.config.loaders import (
    data_config_from_mapping as data_config_from_cfg,
)
from helion_risk_world.data.feature_builder import InMemoryMarketDataSource
from helion_risk_world.data.parquet_source import ParquetMarketDataSource
from helion_risk_world.integration import load_config
from helion_risk_world.runtime import (
    build_runtime_inputs,
    load_model_runtime,
    predict_snapshot,
)


def _resolve_timestamp(raw: str | None, timestamps: list[datetime]) -> datetime:
    if not timestamps:
        raise ValueError("no timestamps available")
    if raw is None:
        return timestamps[-1]
    ts = datetime.fromisoformat(raw)
    if ts not in set(timestamps):
        raise ValueError(f"timestamp {raw} not available in the source")
    return ts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--model-path", required=True, type=Path)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--timestamp", default=None, help="ISO timestamp. Defaults to the latest bar.")
    p.add_argument("--symbol", default=None, help="Prediction symbol label. Defaults to the primary symbol.")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg = load_config(str(args.config))
    dc = data_config_from_cfg(cfg)
    if args.dry_run:
        symbol = args.symbol or dc.universe[0]
        print(
            f"would load {args.model_path} and predict {symbol} "
            f"from {'demo data' if args.demo else args.data_dir}"
        )
        return

    if args.demo:
        source = InMemoryMarketDataSource(candles=_demo_candles(dc.universe))
        timestamps = [row.ts for row in source.candles[dc.universe[0]]]
    else:
        if args.data_dir is None:
            raise ValueError("--data-dir is required unless --demo is set")
        source = ParquetMarketDataSource(
            data_dir=str(args.data_dir),
            universe=dc.universe,
            base_interval=dc.base_interval,
        )
        timestamps = source.timestamps()

    ts = _resolve_timestamp(args.timestamp, timestamps)
    runtime = load_model_runtime(args.model_path)
    symbol = args.symbol or runtime.target_symbol or dc.universe[0]
    inputs = build_runtime_inputs(
        dc,
        source,
        data_dir=args.data_dir,
        runtime=runtime,
    )
    snapshot = inputs.build(ts)
    pred = predict_snapshot(
        runtime,
        snapshot,
        symbol=symbol,
        ts=ts,
    )
    print(pred.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
