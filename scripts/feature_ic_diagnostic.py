"""Per-feature, per-horizon walk-forward Information Coefficient (IC) diagnostic.

No existing tool in this repo measures per-feature predictive power (confirmed by
investigation: `evaluation/predictive_diagnostics.py` operates on already-trained model
outputs, not raw features). This script fills that gap: for every candle/futures/regime/
option-surface feature, it computes the out-of-sample Spearman rank correlation against
forward returns, volatility-ratio, and barrier outcomes at each labeled horizon, both
pooled (full history) and per chronological fold (to check sign/magnitude STABILITY across
time, since a global correlation can hide a feature that was only "predictive" in one
regime).

This is a read-only research script -- it does not train anything and does not modify the
repo's model/config. Run:
    python scripts/feature_ic_diagnostic.py --data-dir data --labels-path data/processed/labels.parquet
"""

from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from _bootstrap import ensure_src_path

ensure_src_path()

from alpha_data.io.paths import DataPaths as AlphaDataPaths  # noqa: E402
from helion_risk_world.config.data_config import DataConfig  # noqa: E402
from helion_risk_world.data.alpha_features import AlphaDataMarketWindowBuilder  # noqa: E402
from helion_risk_world.data.alpha_futures_features import AlphaDataFuturesWindowBuilder  # noqa: E402
from helion_risk_world.data.alpha_option_chain import (  # noqa: E402
    AlphaDataOptionChainSource,
    AlphaDataSurfaceStatsLoader,
    CompositeMarketDataSource,
)
from helion_risk_world.data.alpha_regime_context import AlphaDataMacroContextLoader  # noqa: E402
from helion_risk_world.data.feature_builder import FeatureBuilder  # noqa: E402
from helion_risk_world.data.market_window_builder import CANDLE_FEATURE_NAMES  # noqa: E402
from helion_risk_world.data.option_surface_builder import (  # noqa: E402
    SURFACE_CONTEXT_FEATURES,
    OptionSurfaceBuilder,
)
from helion_risk_world.data.parquet_source import ParquetMarketDataSource  # noqa: E402
from helion_risk_world.data.regime_builder import REGIME_CONTEXT_FEATURES, featurize_regime  # noqa: E402
from helion_risk_world.data.regime_context_builder import RegimeContextBuilder, _VixLoader  # noqa: E402

_HORIZONS = (3, 6, 12, 48, 96, 192)
_N_FOLDS = 5
_UNIVERSE = ("BANKNIFTY", "NIFTY", "FINNIFTY", "HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK")


_CYCLE_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})_")

# option_greeks.py's price_oi_signal is a categorical (price-direction x OI-direction
# quadrant), not numeric -- encode to a signed continuous score so Spearman IC is
# meaningful. long_buildup/short_buildup are "fresh" positioning (stronger); short_
# covering/long_unwinding are unwinds of the opposite side (same direction, weaker
# conviction) -- matches the standard options-flow interpretation of this quadrant.
_PRICE_OI_SIGNAL_SCORE: dict[str, float] = {
    "long_buildup": 1.0,
    "short_covering": 0.5,
    "flat": 0.0,
    "long_unwinding": -0.5,
    "short_buildup": -1.0,
}


def _discover_expiry_cycles(glob_pattern: str, paths: AlphaDataPaths) -> list[date]:
    """Every expiry date embedded in filenames matching ``glob_pattern`` under
    ``paths.options`` -- same convention as alpha_option_chain.py's ``_discover_cycles``
    (reimplemented locally rather than importing that module's private helper; this is a
    read-only research script, not a second production consumer of that internal)."""
    expiries: set[date] = set()
    for p in paths.options.glob(glob_pattern):
        m = _CYCLE_RE.search(p.name)
        if m:
            expiries.add(date.fromisoformat(m.group(1)))
    return sorted(expiries)


def _active_expiry(expiries: list[date], ts: pd.Timestamp) -> date | None:
    """Nearest expiry >= ts's date (the currently-listed near-month contract at ts) --
    matches alpha_option_chain.py's ``_active_cycle`` convention. Options cycles have
    OVERLAPPING date ranges in the raw files (multiple listed expiries tracked
    concurrently), so a plain chronological concat across cycle files is wrong; this
    active-cycle selection is required, not optional."""
    d = ts.date()
    upcoming = [e for e in expiries if e >= d]
    if upcoming:
        return min(upcoming)
    return max(expiries) if expiries else None


def _load_atm_series(
    underlying: str, decision_ts: pd.Series, *, glob_pattern: str, path_template: str,
    columns: tuple[str, ...], paths: AlphaDataPaths,
) -> dict[str, np.ndarray]:
    """As-of (backward) lookup of ``columns`` from a rolling-expiry-cycle file series
    (ATM greeks / ATM PCR / ATM straddle -- all one-row-per-bar-per-cycle, unlike the
    multi-strike surface case) at each of ``decision_ts``. NaN where no cycle is active
    or the column is absent from that cycle's file (matches the option-surface path's
    existing partial-coverage convention -- rows outside coverage aren't excluded,
    just NaN-filled)."""
    expiries = _discover_expiry_cycles(glob_pattern.format(underlying=underlying), paths)
    n = len(decision_ts)
    out = {c: np.full(n, np.nan, dtype=np.float64) for c in columns}
    if not expiries:
        return out
    cache: dict[date, pd.DataFrame | None] = {}
    for i, ts in enumerate(decision_ts):
        ts = pd.Timestamp(ts)
        expiry = _active_expiry(expiries, ts)
        if expiry is None:
            continue
        if expiry not in cache:
            path = paths.options / path_template.format(underlying=underlying, expiry=expiry.isoformat())
            if path.exists():
                cycle_df = pd.read_parquet(path)
                if cycle_df.index.tz is not None:
                    cycle_df.index = cycle_df.index.tz_convert("UTC").tz_localize(None)
                cache[expiry] = cycle_df
            else:
                cache[expiry] = None
        df = cache[expiry]
        if df is None or df.empty:
            continue
        pos = df.index.searchsorted(ts, side="right")
        if pos == 0:
            continue
        row = df.iloc[pos - 1]
        for c in columns:
            if c not in df.columns:
                continue
            val = row[c]
            if c == "price_oi_signal":
                val = _PRICE_OI_SIGNAL_SCORE.get(val, np.nan)
            out[c][i] = val
    return out


def _build_feature_frame(data_dir: str, labels: pd.DataFrame) -> pd.DataFrame:
    """Align every candle/futures/regime/option-surface feature to each label row's
    decision timestamp. Returns a DataFrame indexed the same as `labels`."""
    dc = DataConfig(universe=_UNIVERSE, base_interval="5min", lookback_bars=96)
    source = ParquetMarketDataSource(data_dir=data_dir, universe=_UNIVERSE, base_interval="5min")
    surface_source = CompositeMarketDataSource(source, AlphaDataOptionChainSource(interval="5min"))
    fb = FeatureBuilder(
        dc, surface_source,
        window_builder=AlphaDataMarketWindowBuilder(interval="5min"),
        surface_builder=OptionSurfaceBuilder(n_strikes=5, stats_source=AlphaDataSurfaceStatsLoader(interval="5min")),
        futures_builder=AlphaDataFuturesWindowBuilder(_UNIVERSE[0], interval="5min"),
    )

    print("building candle-feature history...")
    history = fb.build_history()  # [T, A, F]
    positions = history.index.get_indexer(pd.DatetimeIndex(labels["decision_ts"]))
    valid = positions >= 0
    candle_primary = history.candle_features[positions[valid], 0, :]  # [N, F] primary=BANKNIFTY

    print("building futures-microstructure history...")
    fut_index, fut_hist = fb.futures_builder.build_history()
    fut_pos = fut_index.get_indexer(pd.DatetimeIndex(labels["decision_ts"]))
    fut_valid = fut_pos >= 0

    print("building regime vectors (per-row loop; no vectorized alpha_data equivalent)...")
    vix_path = Path(data_dir) / "ohlcv" / "INDIAVIX_5min.parquet"
    vix_loader = _VixLoader.from_parquet(vix_path) if vix_path.exists() else None
    regime_builder = RegimeContextBuilder(
        vix_loader=vix_loader,
        daily_ctx=AlphaDataMacroContextLoader(_UNIVERSE[0], interval="5min"),
        symbol=_UNIVERSE[0],
        allow_non_upstox_context=True,
    )
    regime_rows = np.zeros((len(labels), len(REGIME_CONTEXT_FEATURES)), dtype=np.float32)
    for i, ts in enumerate(labels["decision_ts"]):
        regime, event = regime_builder.build(pd.Timestamp(ts).to_pydatetime())
        regime_rows[i] = featurize_regime(regime, event)

    print("building option-surface context (per-row loop, partial coverage expected)...")
    surface_grid, surface_mask, surface_context, surface_eligible = fb.build_surface_history(
        [pd.Timestamp(ts).to_pydatetime() for ts in labels["decision_ts"]]
    )

    print("reading price-action + candle-pattern columns (2026-07-14 expansion)...")
    alpha_paths = AlphaDataPaths()

    def _read_tz_naive(path: Path) -> pd.DataFrame:
        # alpha_data's feature parquets are tz-aware UTC; labels' decision_ts is
        # tz-naive (matching alpha_labels.py's _to_naive_utc convention) -- convert
        # before reindexing or the join silently matches nothing (same tz bug class
        # documented repeatedly in alpha_features.py's own docstring history).
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(path)
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        return df

    tech_extra = _read_tz_naive(alpha_paths.features / f"{_UNIVERSE[0]}_technical_5min.parquet")
    price_action_cols = (
        "donchian_position", "keltner_position", "consolidation_score", "price_acceleration",
        "higher_high", "lower_low", "pivot_distance_high", "pivot_distance_low",
        "rsi_price_divergence", "upper_wick_pct", "lower_wick_pct", "gap_from_prev_close",
    )
    pattern_df = _read_tz_naive(alpha_paths.features / f"{_UNIVERSE[0]}_candle_patterns_5min.parquet")
    pattern_cols = (
        "bullish_engulfing_score", "bearish_engulfing_score", "doji_score", "marubozu_score",
        "hammer_score", "shooting_star_score", "inside_bar_score", "outside_bar_score",
    )
    decision_index = pd.DatetimeIndex(labels["decision_ts"])

    print("stitching Tier-1 options-family series across expiry cycles (per-row loop)...")
    decision_ts = labels["decision_ts"]
    atm_ce = _load_atm_series(
        _UNIVERSE[0], decision_ts,
        glob_pattern="{underlying}_ATM_CE_*_5min_greeks.parquet",
        path_template="{underlying}_ATM_CE_{expiry}_5min_greeks.parquet",
        columns=("delta", "gamma", "theta", "vega", "rho", "iv", "oi_pct_change", "price_oi_signal"),
        paths=alpha_paths,
    )
    atm_pe = _load_atm_series(
        _UNIVERSE[0], decision_ts,
        glob_pattern="{underlying}_ATM_PE_*_5min_greeks.parquet",
        path_template="{underlying}_ATM_PE_{expiry}_5min_greeks.parquet",
        columns=("delta", "gamma", "theta", "vega", "rho", "iv", "oi_pct_change", "price_oi_signal"),
        paths=alpha_paths,
    )
    atm_pcr = _load_atm_series(
        _UNIVERSE[0], decision_ts,
        glob_pattern="{underlying}_ATM_PCR_*_5min.parquet",
        path_template="{underlying}_ATM_PCR_{expiry}_5min.parquet",
        columns=("pcr_oi", "total_oi"),
        paths=alpha_paths,
    )
    straddle = _load_atm_series(
        _UNIVERSE[0], decision_ts,
        glob_pattern="{underlying}_STRADDLE_ATM_*_5min.parquet",
        path_template="{underlying}_STRADDLE_ATM_{expiry}_5min.parquet",
        columns=("straddle", "spread_proxy"),
        paths=alpha_paths,
    )
    otm5_ce = _load_atm_series(
        _UNIVERSE[0], decision_ts,
        glob_pattern="{underlying}_OTM5_CE_*_5min_greeks.parquet",
        path_template="{underlying}_OTM5_CE_{expiry}_5min_greeks.parquet",
        columns=("iv", "delta", "vega"),
        paths=alpha_paths,
    )
    otm5_pe = _load_atm_series(
        _UNIVERSE[0], decision_ts,
        glob_pattern="{underlying}_OTM5_PE_*_5min_greeks.parquet",
        path_template="{underlying}_OTM5_PE_{expiry}_5min_greeks.parquet",
        columns=("iv", "delta", "vega"),
        paths=alpha_paths,
    )

    n = len(labels)
    out = pd.DataFrame(index=labels.index)
    for j, name in enumerate(CANDLE_FEATURE_NAMES):
        col = np.full(n, np.nan, dtype=np.float32)
        col[valid] = candle_primary[:, j]
        out[f"candle__{name}"] = col
    fut_cols = ("basis", "oi_norm", "d_oi", "volume_zscore", "calendar_spread", "dte_norm",
                "roll_flag", "d_oi_mag", "oi_available", "oi_basis_interaction",
                "long_buildup", "short_covering", "short_buildup", "long_unwinding")
    for j, name in enumerate(fut_cols):
        col = np.full(n, np.nan, dtype=np.float32)
        col[fut_valid] = fut_hist[fut_pos[fut_valid], j]
        out[f"futures__{name}"] = col
    for j, name in enumerate(REGIME_CONTEXT_FEATURES):
        out[f"regime__{name}"] = regime_rows[:, j]
    for j, name in enumerate(SURFACE_CONTEXT_FEATURES):
        col = surface_context[:, j].astype(np.float32)
        col[~surface_eligible] = np.nan
        out[f"surface__{name}"] = col
    out["_surface_eligible"] = surface_eligible

    for name in price_action_cols:
        out[f"priceaction__{name}"] = (
            tech_extra[name].reindex(decision_index).to_numpy(dtype=np.float64)
            if name in tech_extra.columns else np.full(n, np.nan)
        )
    for name in pattern_cols:
        out[f"candlepattern__{name}"] = (
            pattern_df[name].reindex(decision_index).to_numpy(dtype=np.float64)
            if name in pattern_df.columns else np.full(n, np.nan)
        )
    for name, col in atm_ce.items():
        out[f"atmce__{name}"] = col
    for name, col in atm_pe.items():
        out[f"atmpe__{name}"] = col
    for name, col in atm_pcr.items():
        out[f"atmpcr__{name}"] = col
    for name, col in straddle.items():
        out[f"straddle__{name}"] = col
    for name, col in otm5_ce.items():
        out[f"otm5ce__{name}"] = col
    for name, col in otm5_pe.items():
        out[f"otm5pe__{name}"] = col
    return out


def _spearman_ic(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 30:
        return float("nan"), n
    if np.nanstd(x[mask]) < 1e-12 or np.nanstd(y[mask]) < 1e-12:
        return 0.0, n
    rho, _ = stats.spearmanr(x[mask], y[mask])
    return float(rho), n


def _chronological_folds(n: int, n_folds: int) -> list[np.ndarray]:
    edges = np.linspace(0, n, n_folds + 1, dtype=int)
    return [np.arange(edges[k], edges[k + 1]) for k in range(n_folds)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--labels-path", default="data/processed/labels.parquet")
    parser.add_argument("--out-path", default="runs/feature_ic_report.csv")
    parser.add_argument("--max-rows", type=int, default=None, help="Debug: cap label rows.")
    parser.add_argument(
        "--horizons", type=int, nargs="+", default=None,
        help="Fixed horizons to test IC against, must match columns present in the labels "
             "parquet (horizon_return_H/horizon_vol_H/horizon_mae_H/horizon_mfe_H). Defaults "
             "to the module's full (3,6,12,48,96,192) set.",
    )
    args = parser.parse_args()
    horizons = tuple(args.horizons) if args.horizons else _HORIZONS
    summary_h = max(horizons)

    labels = pd.read_parquet(args.labels_path).sort_values("decision_ts").reset_index(drop=True)
    if args.max_rows:
        labels = labels.iloc[: args.max_rows]
    print(f"loaded {len(labels)} label rows: {labels['decision_ts'].min()} .. {labels['decision_ts'].max()}")

    feature_cache = Path(args.out_path).with_suffix(".features.parquet")
    if feature_cache.exists() and not args.max_rows:
        print(f"reusing cached feature frame at {feature_cache}")
        features = pd.read_parquet(feature_cache)
    else:
        features = _build_feature_frame(args.data_dir, labels)
        if not args.max_rows:
            features.to_parquet(feature_cache)
    feature_cols = [c for c in features.columns if c != "_surface_eligible"]

    barrier_stop = (labels["barrier"] == "stop").to_numpy(dtype=np.float32)
    barrier_target = (labels["barrier"] == "target").to_numpy(dtype=np.float32)
    # {-1, 0, +1} directional barrier outcome, at whatever management horizon H this labels
    # parquet was built with (the "barrier" column's own H, not necessarily 192).
    barrier_edge = barrier_target - barrier_stop

    # Proxy "would this row hit stop/target if the decision window were only H bars" edge, at
    # EVERY horizon: same fixed barrier geometry (stop/target return, set at decision time,
    # independent of how far forward we look), but using horizon_mae_H/horizon_mfe_H instead of
    # the management-horizon-only mae/mfe. Mirrors
    # composite_loss.py::_excursion_barrier_labels's ratio construction, kept continuous here
    # (not thresholded to a class) for a cleaner IC.
    stop_scale = labels["barrier_stop_return"].abs().to_numpy(dtype=np.float64).clip(min=1e-8)
    target_scale = labels["barrier_target_return"].abs().to_numpy(dtype=np.float64).clip(min=1e-8)
    horizon_edge = {}
    for h in horizons:
        mae_h = labels[f"horizon_mae_{h}"].to_numpy(dtype=np.float64)
        mfe_h = labels[f"horizon_mfe_{h}"].to_numpy(dtype=np.float64)
        horizon_edge[h] = mfe_h / target_scale - mae_h / stop_scale

    folds = _chronological_folds(len(labels), _N_FOLDS)

    rows = []
    for col in feature_cols:
        x_full = features[col].to_numpy(dtype=np.float64)
        record: dict[str, object] = {"feature": col}
        for h in horizons:
            y = labels[f"horizon_return_{h}"].to_numpy(dtype=np.float64)
            ic_global, n_global = _spearman_ic(x_full, y)
            fold_ics = []
            for fold_idx in folds:
                ic_f, n_f = _spearman_ic(x_full[fold_idx], y[fold_idx])
                if n_f >= 30:
                    fold_ics.append(ic_f)
            fold_ics_arr = np.array(fold_ics, dtype=float)
            record[f"ic_return_{h}"] = ic_global
            record[f"ic_return_{h}_foldmean"] = float(np.nanmean(fold_ics_arr)) if len(fold_ics_arr) else np.nan
            record[f"ic_return_{h}_foldstd"] = float(np.nanstd(fold_ics_arr)) if len(fold_ics_arr) else np.nan

            yvol = labels[f"horizon_vol_{h}"].to_numpy(dtype=np.float64)
            vb = labels["barrier_sigma"].to_numpy(dtype=np.float64)
            vol_ratio = yvol / np.clip(vb, 1e-8, None)
            ic_vol, _ = _spearman_ic(x_full, vol_ratio)
            record[f"ic_volratio_{h}"] = ic_vol

            ic_edge_h, n_edge_h = _spearman_ic(x_full, horizon_edge[h])
            fold_edge_ics = []
            for fold_idx in folds:
                ic_f, n_f = _spearman_ic(x_full[fold_idx], horizon_edge[h][fold_idx])
                if n_f >= 30:
                    fold_edge_ics.append(ic_f)
            fold_edge_arr = np.array(fold_edge_ics, dtype=float)
            record[f"ic_edge_{h}"] = ic_edge_h
            record[f"ic_edge_{h}_foldmean"] = float(np.nanmean(fold_edge_arr)) if len(fold_edge_arr) else np.nan
            record[f"ic_edge_{h}_foldstd"] = float(np.nanstd(fold_edge_arr)) if len(fold_edge_arr) else np.nan

        ic_edge, n_edge = _spearman_ic(x_full, barrier_edge)
        record["ic_barrier_edge"] = ic_edge
        record["n_barrier_edge"] = n_edge
        rows.append(record)

    report = pd.DataFrame(rows).set_index("feature")
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out_path)
    print(f"\nsaved full report to {args.out_path}")

    print(f"\n=== Top 20 |IC| vs horizon_return_{summary_h} (global, pooled) ===")
    ret_col = f"ic_return_{summary_h}"
    top = report.reindex(report[ret_col].abs().sort_values(ascending=False).index)
    print(top[[ret_col, f"{ret_col}_foldmean", f"{ret_col}_foldstd"]].head(20).to_string())

    print(f"\n=== Top 20 |IC| vs volatility-ratio (H={summary_h}) ===")
    vol_col = f"ic_volratio_{summary_h}"
    top_vol = report.reindex(report[vol_col].abs().sort_values(ascending=False).index)
    print(top_vol[[vol_col]].head(20).to_string())

    print("\n=== Top 20 |IC| vs barrier directional edge (target=+1, stop=-1) ===")
    top_edge = report.reindex(report["ic_barrier_edge"].abs().sort_values(ascending=False).index)
    print(top_edge[["ic_barrier_edge", "n_barrier_edge"]].head(20).to_string())


if __name__ == "__main__":
    main()
