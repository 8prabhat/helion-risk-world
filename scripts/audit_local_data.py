"""Audit the local HelionRiskWorld dataset and emit a capability/coverage report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ensure_src_path

ensure_src_path()

from helion_risk_world.config.loaders import data_config_from_mapping as data_config_from_cfg
from helion_risk_world.data.capability_profile import DataCapabilityProfile
from helion_risk_world.data.parquet_source import ParquetMarketDataSource, load_ohlcv_parquet
from helion_risk_world.integration import load_config


def _frame_report(data_dir: Path, symbol: str, interval: str) -> dict[str, object]:
    path = data_dir / "ohlcv" / f"{symbol}_{interval}.parquet"
    if not path.exists():
        return {"symbol": symbol, "present": False}
    df = load_ohlcv_parquet(path)
    return {
        "symbol": symbol,
        "present": True,
        "rows": int(len(df)),
        "start": str(df.index.min()) if not df.empty else None,
        "end": str(df.index.max()) if not df.empty else None,
        "missing_close": int(df["close"].isna().sum()) if "close" in df.columns else 0,
        "zero_volume_rows": int((df.get("volume", 0.0) == 0).sum()) if "volume" in df.columns else 0,
    }


def build_report(config_path: Path, data_dir: Path) -> dict[str, object]:
    cfg = load_config(str(config_path))
    dc = data_config_from_cfg(cfg)
    profile = DataCapabilityProfile.from_data_dir(data_dir, dc)
    common_timestamps: list[object] = []
    if not profile.missing_assets:
        source = ParquetMarketDataSource(str(data_dir), dc.universe, base_interval=dc.base_interval)
        common_timestamps = source.timestamps()
    return {
        "capability_profile": profile.to_metadata(),
        "common_timestamp_count": len(common_timestamps),
        "common_start": str(common_timestamps[0]) if common_timestamps else None,
        "common_end": str(common_timestamps[-1]) if common_timestamps else None,
        "asset_reports": [
            _frame_report(data_dir, symbol, dc.base_interval) for symbol in dc.universe
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--out-path", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    report = build_report(args.config, args.data_dir)
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return
    if args.out_path is not None:
        args.out_path.parent.mkdir(parents=True, exist_ok=True)
        args.out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
