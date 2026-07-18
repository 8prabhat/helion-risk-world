"""Compatibility adapter: a ``DailyContextLoader``-shaped (``.get(ts) -> dict``)
loader backed by alpha_data's REAL Upstox-sourced macro/IV context instead of the
"legacy daily context" ``RegimeContextBuilder`` deliberately disables by default
(its own module docstring: "hard-coded macro/event calendars are not Upstox data").

This is a genuine CAPABILITY UPGRADE, not a parity port: in helion's default
Upstox-only path (``allow_non_upstox_context=False``), ``daily_ctx`` is always
``None``, so every macro field this loader provides (ATM-IV, IV-skew, FII/DII flow,
USDINR/crude returns+vol, PCR) is already always zero/inert today. alpha_data's data
IS Upstox-sourced end to end (its own options/macro backfill, see
``alpha_data/docs/DATA_CATALOG.md``), so this loader is constructed with
``allow_non_upstox_context=True`` on ``RegimeContextBuilder`` -- that flag's name
predates this adapter and really means "any daily_ctx source", not "definitely not
Upstox"; the docstring warning it guards against does not apply to this loader.

SCALE-CONVENTION CHANGE, flagged for the Phase 2 parity review (not silently decided):
helion's own ``regime_builder.featurize_regime`` divides ``atm_iv``/``iv_skew`` by
fixed constants (100.0 / 10.0), treating them as raw percentage levels -- but
alpha_data's ``macro_features.py`` deliberately does NOT stitch a raw ATM-IV/skew
level series into its stored ``macro_context.parquet`` (its own design philosophy:
"raw flow/rate LEVELS aren't comparable across regimes; a rolling z-score is" --
matching the ALREADY-z-scored convention helion's own fii_dii_net_z/pc_oi_ratio_z use).
This loader therefore feeds alpha_data's ``atm_iv_zscore``/``iv_skew_zscore``/
``pcr_oi_zscore`` into the ``atm_iv_pct``/``iv_skew_pct``/``pc_oi_ratio`` dict keys --
a switch from raw-level/fixed-divisor to z-score for these three fields specifically,
consistent with (rather than inconsistent with, as helion's current mixed convention
is) the two fields already z-scored. ``featurize_regime``'s own fixed /100, /10
divisors would need updating (or accepting the resulting scale as-is, since a z-score
divided by 10 is still a reasonable small-magnitude regime input) once this is
reviewed -- not changed unilaterally here.

``basis_daily``/raw ``usdinr``/``crude``/``fii_dii_net`` levels: alpha_data has no
daily-aggregated equivalent (its own basis feature is intraday, in
``pipelines/basis_features.py``) and helion's own ``featurize_regime`` doesn't
actually consume the raw (non-return, non-z-scored) forms of usdinr/crude/fii_dii_net
at all -- left as ``None`` (matching the current always-``None`` default behavior for
``basis_daily``, and a genuine no-op for the unused raw fields).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from quanthelion.utils.logging import get_logger

from alpha_data.io.paths import DataPaths as AlphaDataPaths
from alpha_data.pipelines.macro_features import stitch_atm_iv, stitch_iv_skew, stitch_pcr

log = get_logger(__name__)

_STALE_DAYS = 10


class AlphaDataMacroContextLoader:
    """``DailyContextLoader``-shaped loader backed by alpha_data's real macro/IV data."""

    _COLS = (
        "usdinr", "crude", "fii_dii_net",
        "atm_iv_pct", "iv_skew_pct", "pc_oi_ratio", "basis",
        "fii_dii_net_z", "pc_oi_ratio_z",
        "usdinr_ret_5d", "crude_ret_5d",
        "usdinr_vol", "crude_vol",
        "realized_vol_vix_ratio", "breadth_index_divergence",
    )

    def __init__(
        self,
        underlying: str = "BANKNIFTY",
        *,
        paths: AlphaDataPaths | None = None,
        interval: str = "5min",
    ) -> None:
        self._paths = paths or AlphaDataPaths()
        macro_path = self._paths.features / f"{underlying}_macro_context.parquet"
        self._df: pd.DataFrame | None = None
        if macro_path.exists():
            df = pd.read_parquet(macro_path)
            df.index = pd.to_datetime(df.index).normalize()
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            atm_iv = stitch_atm_iv(underlying, paths=self._paths)
            iv_skew = stitch_iv_skew(underlying, paths=self._paths)
            pcr = stitch_pcr(underlying, paths=self._paths)
            for name, series in (("atm_iv_pct", atm_iv), ("iv_skew_pct", iv_skew), ("pc_oi_ratio", pcr)):
                if series.empty:
                    continue
                idx = pd.to_datetime(series.index).normalize()
                if idx.tz is not None:
                    idx = idx.tz_localize(None)
                df[name] = pd.Series(series.to_numpy(), index=idx).reindex(df.index)
            # Realized-vol/VIX ratio (alpha_data pipelines/cross_asset_features.py::compute_vol_ratio,
            # `{underlying}_FUT_vs_INDIAVIX_{interval}.parquet`) -- daily-resampled (last value of
            # day) onto the same daily macro-context index. This is the model's own strongest known
            # signal per the walk-forward diagnostic in configs/v1.yaml (OOS corr 0.34 -> 0.65,
            # 12 -> 192 bars) -- a direct read of an alpha_data output, not a local recomputation.
            vol_ratio_path = self._paths.features / f"{underlying}_FUT_vs_INDIAVIX_{interval}.parquet"
            if vol_ratio_path.exists():
                vr = pd.read_parquet(vol_ratio_path)["realized_vol_vix_ratio"]
                idx = pd.to_datetime(vr.index)
                if idx.tz is not None:
                    idx = idx.tz_convert("UTC").tz_localize(None)
                daily = pd.Series(vr.to_numpy(), index=idx).resample("D").last()
                df["realized_vol_vix_ratio"] = daily.reindex(df.index)
            else:
                log.warning("no alpha_data vol-ratio context found", path=str(vol_ratio_path))
            # Breadth/dispersion divergence (alpha_data pipelines/breadth_features.py,
            # `{underlying}_breadth_{interval}.parquet`) -- only `breadth_index_divergence` is
            # merged here (a regime-level scalar); `pct_advancing`/`return_dispersion` feed the
            # candle-tensor breadth/dispersion columns directly in feature_builder.py instead.
            breadth_path = self._paths.features / f"{underlying}_breadth_{interval}.parquet"
            if breadth_path.exists():
                bd = pd.read_parquet(breadth_path)["breadth_index_divergence"]
                idx = pd.to_datetime(bd.index)
                if idx.tz is not None:
                    idx = idx.tz_convert("UTC").tz_localize(None)
                daily = pd.Series(bd.to_numpy(), index=idx).resample("D").last()
                df["breadth_index_divergence"] = daily.reindex(df.index)
            else:
                log.warning("no alpha_data breadth context found", path=str(breadth_path))
            # Forward-fill non-reporting days (weekends/holidays/FII-DII publication
            # gaps) to the last known value, matching DailyContextLoader._load's own
            # convention exactly -- otherwise get() on a gap day returns None even
            # though a recent, still-valid value exists a day or two prior.
            self._df = df.sort_index().ffill()
        else:
            log.warning("no alpha_data macro context found", path=str(macro_path))

    def get(self, ts: datetime) -> dict[str, float | None]:
        """Return context values for the date of ``ts``. None for unavailable/stale columns."""
        if self._df is None:
            return {c: None for c in self._COLS}

        query_date = pd.Timestamp(ts).normalize()
        if query_date.tz is not None:
            query_date = query_date.tz_localize(None)
        end = self._df.index.searchsorted(query_date, side="right")
        if end == 0:
            return {c: None for c in self._COLS}
        latest_idx = self._df.index[end - 1]
        if (query_date - latest_idx).days > _STALE_DAYS:
            return {c: None for c in self._COLS}
        row = self._df.iloc[end - 1]

        def _get(col: str) -> float | None:
            return float(row[col]) if col in row.index and pd.notna(row[col]) else None

        return {
            "usdinr": None,
            "crude": None,
            "fii_dii_net": None,
            "atm_iv_pct": _get("atm_iv_pct"),
            "iv_skew_pct": _get("iv_skew_pct"),
            "pc_oi_ratio": _get("pc_oi_ratio"),
            "basis": None,
            "fii_dii_net_z": _get("fii_dii_net_combined_zscore"),
            "pc_oi_ratio_z": _get("pcr_oi_zscore"),
            "usdinr_ret_5d": _get("usdinr_ret_5d"),
            "crude_ret_5d": _get("crude_ret_5d"),
            "usdinr_vol": _get("usdinr_vol"),
            "crude_vol": _get("crude_vol"),
            "realized_vol_vix_ratio": _get("realized_vol_vix_ratio"),
            "breadth_index_divergence": _get("breadth_index_divergence"),
        }


__all__ = ["AlphaDataMacroContextLoader"]
