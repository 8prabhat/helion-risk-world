"""Local data capability inspection for the current on-disk V1 dataset.

The goal is to make every artifact and runtime explicit about which context sources were
actually available locally, which optional feeds were absent, and which modules must stay
disabled as a result.  This is descriptive metadata only; it does not invent data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from helion_risk_world.config.data_config import DataConfig
from quanthelion.data.quality.capability_profile import check_paths_exist, glob_stems

LOCAL_CONTEXT_GROUPS: dict[str, tuple[str, ...]] = {
    "indices": ("BANKNIFTY", "NIFTY", "FINNIFTY"),
    "bank_basket": ("HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK"),
    "volatility": ("INDIAVIX",),
}


@dataclass(frozen=True)
class DataCapabilityProfile:
    """Summary of what the current local dataset can and cannot support."""

    universe: tuple[str, ...]
    available_assets: tuple[str, ...]
    missing_assets: tuple[str, ...]
    has_tradeable_futures_continuous: bool
    monthly_futures_contracts: tuple[str, ...]
    has_indiavix: bool
    has_daily_context: bool
    has_nse_daily_fo: bool
    has_nse_fii_fo: bool
    has_processed_futures_dataset: bool
    has_labels: bool
    supports_options_history: bool = False
    supports_market_depth_history: bool = False
    supports_execution_logs: bool = False

    @classmethod
    def from_data_dir(
        cls,
        data_dir: str | Path,
        cfg: DataConfig,
    ) -> DataCapabilityProfile:
        root = Path(data_dir)

        expected_assets = {
            symbol: f"ohlcv/{symbol}_{cfg.base_interval}.parquet" for symbol in cfg.universe
        }
        asset_exists = check_paths_exist(root, expected_assets)
        available_assets = tuple(s for s in cfg.universe if asset_exists[s])
        missing_assets = tuple(s for s in cfg.universe if not asset_exists[s])

        monthly = tuple(
            token.replace("_5min", "")
            for token in glob_stems(root, "ohlcv/BANKNIFTY_FUT_[0-9][0-9][0-9][0-9]_5min.parquet")
        )

        singletons = check_paths_exist(root, {
            "futures_continuous": f"ohlcv/BANKNIFTY_FUT_continuous_{cfg.base_interval}.parquet",
            "indiavix": f"ohlcv/INDIAVIX_{cfg.base_interval}.parquet",
            "daily_context": "regime/daily_context.parquet",
            "nse_daily_fo": "regime/nse_daily_fo.parquet",
            "nse_fii_fo": "regime/nse_fii_fo.parquet",
            "processed_futures": "processed/banknifty_5min.parquet",
            "labels": "processed/labels.parquet",
        })

        return cls(
            universe=tuple(cfg.universe),
            available_assets=available_assets,
            missing_assets=missing_assets,
            has_tradeable_futures_continuous=singletons["futures_continuous"],
            monthly_futures_contracts=monthly,
            has_indiavix=singletons["indiavix"],
            has_daily_context=singletons["daily_context"],
            has_nse_daily_fo=singletons["nse_daily_fo"],
            has_nse_fii_fo=singletons["nse_fii_fo"],
            has_processed_futures_dataset=singletons["processed_futures"],
            has_labels=singletons["labels"],
        )

    def enabled_context_groups(self) -> tuple[str, ...]:
        groups: list[str] = []
        for name, members in LOCAL_CONTEXT_GROUPS.items():
            if all(member in self.available_assets for member in members):
                groups.append(name)
        return tuple(groups)

    def disabled_optional_modules(self) -> tuple[str, ...]:
        disabled = ["options_history", "market_depth_history", "execution_logs"]
        if not self.has_indiavix:
            disabled.append("vix_regime_context")
        if not self.has_daily_context:
            disabled.append("daily_regime_context")
        return tuple(disabled)

    def critical_issues(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.missing_assets:
            issues.append(f"missing_assets:{','.join(self.missing_assets)}")
        if not self.has_tradeable_futures_continuous:
            issues.append("missing_futures_continuous")
        if not self.has_processed_futures_dataset:
            issues.append("missing_processed_futures_dataset")
        return tuple(issues)

    def to_metadata(self) -> dict[str, object]:
        payload = asdict(self)
        payload["enabled_context_groups"] = list(self.enabled_context_groups())
        payload["disabled_optional_modules"] = list(self.disabled_optional_modules())
        payload["critical_issues"] = list(self.critical_issues())
        return payload


__all__ = ["DataCapabilityProfile", "LOCAL_CONTEXT_GROUPS"]
