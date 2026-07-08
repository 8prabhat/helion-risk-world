"""Build and cache point-in-time market features (SPEC.md §10, Day 3).

Wires the shared FeatureBuilder + OptionSurfaceBuilder and caches a MarketBatch to disk.

Usage:
    # real run (requires a configured MarketDataSource — see TODO below):
    python scripts/build_features.py --config configs/v1.yaml

    # demo run (synthetic in-memory source so the path is runnable today):
    python scripts/build_features.py --config configs/v1.yaml --demo [--dry-run]
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from _common import log, setup

from helion_risk_world.config.data_config import DataConfig
from helion_risk_world.config.loaders import (
    data_config_from_mapping as data_config_from_cfg,
)
from helion_risk_world.data.feature_builder import FeatureBuilder, InMemoryMarketDataSource
from helion_risk_world.schemas.market_schema import MarketCandle
from helion_risk_world.schemas.option_chain_schema import OptionContractSnapshot, OptionType


def _synthetic_source(dc: DataConfig, n: int = 120) -> tuple[InMemoryMarketDataSource, datetime]:
    """Deterministic synthetic candles + a chain for the primary underlying (demo only)."""
    rng = np.random.default_rng(7)
    start = datetime(2026, 6, 25, 9, 20)
    candles: dict[str, list[MarketCandle]] = {}
    for s, base in zip(dc.universe, 100 + rng.uniform(0, 50, len(dc.universe)), strict=False):
        rows = []
        price = float(base)
        for i in range(n):
            ts = start + timedelta(minutes=5 * i)
            price = max(1.0, price * (1.0 + float(rng.normal(0, 0.002))))
            rows.append(
                MarketCandle(
                    symbol=s, ts=ts, available_at=ts, open=price,
                    high=price * 1.004, low=price * 0.996, close=price,
                    volume=1000 + i, oi=5000 + i * 5,
                )
            )
        candles[s] = rows
    ts = candles[dc.universe[0]][-1].ts
    spot = candles[dc.universe[0]][-1].close
    chain = []
    for k in range(-dc.n_strikes - 1, dc.n_strikes + 2):
        strike = round(spot) + k
        for opt in (OptionType.CALL, OptionType.PUT):
            chain.append(
                OptionContractSnapshot(
                    underlying=dc.universe[0], strike=float(strike), opt_type=opt, ts=ts,
                    available_at=ts, open=1, high=2, low=0.5, close=1.5, volume=10,
                    oi=100 + abs(k), iv=0.18 + 0.01 * abs(k), dte=2.0,
                )
            )
    return InMemoryMarketDataSource(candles=candles, chains={dc.universe[0]: chain}), ts


def main() -> None:
    args, cfg = setup(
        "Build and cache point-in-time market features (SPEC.md §10, Day 3).",
        option_groups=("demo",),
    )
    dc = data_config_from_cfg(cfg)

    if not args.demo:
        log.warning(
            "build_features.no_source note=%s",
            "No real MarketDataSource wired yet. Run with --demo to exercise the pipeline, "
            "or implement a MarketDataSource over your parquet store (SPEC.md §8).",
        )
        return

    source, ts = _synthetic_source(dc)
    batch = FeatureBuilder(dc, source).build_window(ts)
    log.info(
        "build_features.built",
        ts=ts.isoformat(),
        candle_shape=list(batch.candle_features.shape),
        n_features=len(batch.feature_names),
        has_surface=batch.surface is not None,
        atm_strike=getattr(batch.surface, "atm_strike", None),
    )

    if args.dry_run:
        log.info("build_features.dry_run", note="skipping cache write")
        return

    out_dir = Path(dc.feature_cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"features_{ts:%Y%m%d_%H%M}.npz"
    np.savez(
        out_path,
        candle_features=batch.candle_features,
        feature_names=np.array(batch.feature_names),
        symbols=np.array(batch.symbols),
    )
    log.info("build_features.cached", path=str(out_path))


if __name__ == "__main__":
    main()
