"""Typed data configuration (SPEC.md §7, §8)."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_UNIVERSE: tuple[str, ...] = (
    "BANKNIFTY", "NIFTY", "FINNIFTY",
    "HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK",
)


@dataclass(frozen=True)
class DataConfig:
    """Data sourcing and feature-window configuration."""

    universe: tuple[str, ...] = DEFAULT_UNIVERSE
    base_interval: str = "5min"
    lookback_bars: int = 96            # ~ one session of 5-min bars
    n_strikes: int = 5                 # ATM-N .. ATM+N -> 2N+1 strike tokens
    feature_cache_dir: str = "feature_cache"
    data_sources_path: str = "configs/data_sources.yaml"
    # V1 guard: must be False (no historical tick/depth dependency).
    use_historical_depth: bool = False

    def __post_init__(self) -> None:
        if self.use_historical_depth:
            raise ValueError(
                "V1 must not depend on historical tick/depth data (SPEC.md §8, §30). "
                "Set use_historical_depth=False; depth is a V2/V3 feature."
            )
        if self.n_strikes < 1:
            raise ValueError("n_strikes must be >= 1")
