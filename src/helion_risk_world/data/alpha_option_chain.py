"""Compatibility source: real option-chain snapshots from alpha_data's rolling
ATM±N-strike surface (with Black-Scholes greeks), wired into helion's own
``OptionSurfaceBuilder.align_to_atm``/``featurize_surface`` -- those two functions'
ATM-alignment and derived-feature logic (PCR, walls, max-pain, IV skew, ...) are
generic assembly code that works on whatever ``OptionContractSnapshot`` list it's
given, so this only needs to supply a REAL chain, not replace that logic.

This is a genuine CAPABILITY UPGRADE, not a parity port: ``ParquetMarketDataSource.
option_chain()`` always returns ``None`` today (Phase 2 migration audit,
docs/DATA_CATALOG.md) -- there is no historical option-chain data wired into helion
at all currently. alpha_data's own real, Upstox-sourced ATM±5-strike surface (with
Black-Scholes greeks via ``pipelines/option_greeks.py``) already exceeds what any of
the three consumer repos have wired up themselves.

Per-cycle surface files (``pipelines/option_surface.py``) are discovered by expiry
date parsed from the filename; the cycle whose expiry is the nearest one >= ``ts``'s
date is treated as the active contract at ``ts`` (matches how a trader would read
"the currently listed weekly/monthly contract" at any point in time).
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from alpha_data.io.paths import DataPaths as AlphaDataPaths
from helion_risk_world.schemas.option_chain_schema import OptionContractSnapshot, OptionType

_CYCLE_RE = re.compile(r"_SURFACE_(CE|PE)_(\d{4}-\d{2}-\d{2})_(\w+)(?:_greeks)?\.parquet$")


def _discover_cycles(underlying: str, interval: str, paths: AlphaDataPaths) -> list[date]:
    """Every expiry date with both CE and PE (+ greeks) surface files for ``underlying``."""
    expiries: set[date] = set()
    for p in paths.options.glob(f"{underlying}_SURFACE_CE_*_{interval}_greeks.parquet"):
        m = _CYCLE_RE.search(p.name)
        if m:
            expiries.add(date.fromisoformat(m.group(2)))
    return sorted(expiries)


def _active_cycle(expiries: list[date], ts: datetime) -> date | None:
    """Nearest expiry >= ts's date (the currently-listed contract at ts)."""
    d = ts.date()
    upcoming = [e for e in expiries if e >= d]
    if upcoming:
        return min(upcoming)
    return max(expiries) if expiries else None


def _load_cycle_side(
    underlying: str, expiry: date, side: str, interval: str, paths: AlphaDataPaths,
) -> pd.DataFrame | None:
    path = paths.options / f"{underlying}_SURFACE_{side}_{expiry.isoformat()}_{interval}_greeks.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    # Normalize to naive-UTC-numeric (matching ParquetMarketDataSource's own
    # convention) via tz_convert first, not a blind tz_localize(None) -- alpha_data
    # isn't internally consistent about which tz its own parquets carry (options
    # files are tz-aware UTC today, but a plain strip would silently misalign if that
    # ever changes; see alpha_futures_features.py's _to_naive_utc for the same fix
    # applied where this bit -- futures-continuous files are tz-aware Asia/Kolkata).
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df = df.sort_index()
    # Bar-to-bar OI change per strike token (the surface stacks all tokens at every
    # timestamp, so group by token before diffing).
    df["d_oi"] = df.groupby("token")["oi"].diff()
    return df


class AlphaDataOptionChainSource:
    """Supplies ``option_chain(underlying, ts)`` from alpha_data's real option
    surface data. Compose with an existing ``MarketDataSource`` (e.g.
    ``ParquetMarketDataSource``) for ``candle_window``/``spot``/etc. -- this class
    only implements the option-chain half of the protocol.
    """

    def __init__(self, *, interval: str = "5min", paths: AlphaDataPaths | None = None) -> None:
        self._interval = interval
        self._paths = paths or AlphaDataPaths()
        self._cycles_cache: dict[str, list[date]] = {}
        self._side_cache: dict[tuple[str, date, str], pd.DataFrame | None] = {}

    def _cycles(self, underlying: str) -> list[date]:
        cached = self._cycles_cache.get(underlying)
        if cached is None:
            cached = _discover_cycles(underlying, self._interval, self._paths)
            self._cycles_cache[underlying] = cached
        return cached

    def _side(self, underlying: str, expiry: date, side: str) -> pd.DataFrame | None:
        key = (underlying, expiry, side)
        if key not in self._side_cache:
            self._side_cache[key] = _load_cycle_side(underlying, expiry, side, self._interval, self._paths)
        return self._side_cache[key]

    def option_chain(self, underlying: str, ts: datetime) -> list[OptionContractSnapshot] | None:
        """Return the latest option-chain snapshot available at ``ts`` (or None)."""
        expiry = _active_cycle(self._cycles(underlying), ts)
        if expiry is None:
            return None

        contracts: list[OptionContractSnapshot] = []
        for side, opt_type in (("CE", OptionType.CALL), ("PE", OptionType.PUT)):
            df = self._side(underlying, expiry, side)
            if df is None or df.empty:
                continue
            end = df.index.searchsorted(pd.Timestamp(ts), side="right")
            if end == 0:
                continue
            snap_ts = df.index[end - 1]
            rows = df[df.index == snap_ts]
            dte = max((expiry - ts.date()).days, 0) + 1.0
            for _, row in rows.iterrows():
                contracts.append(
                    OptionContractSnapshot(
                        underlying=underlying,
                        strike=float(row["strike"]),
                        opt_type=opt_type,
                        ts=snap_ts.to_pydatetime(),
                        available_at=snap_ts.to_pydatetime(),
                        open=float(row["open"]), high=float(row["high"]),
                        low=float(row["low"]), close=float(row["close"]),
                        volume=float(row["volume"]), oi=float(row["oi"]),
                        d_oi=float(row["d_oi"]) if pd.notna(row.get("d_oi")) else None,
                        iv=float(row["iv"]) if pd.notna(row.get("iv")) else None,
                        delta=float(row["delta"]) if pd.notna(row.get("delta")) else None,
                        gamma=float(row["gamma"]) if pd.notna(row.get("gamma")) else None,
                        dte=dte,
                    )
                )
        return contracts or None


class AlphaDataSurfaceStatsLoader:
    """Per-bar option-surface derived stats from alpha_data's precomputed
    ``{underlying}_SURFACE_STATS_{expiry}_{interval}.parquet`` (generated by alpha_data's
    own ``compute_surface_stats``, ``alpha-data compute-surface-stats`` CLI -- see
    ``pipelines/option_surface.py``, whose own docstring notes it was "ported from
    helion_risk_world's ``OptionSurfaceBuilder._derived_features``").

    Feature-onboarding pass: this replaces ``OptionSurfaceBuilder``'s local recomputation
    of pcr/wall-strengths/gamma-concentration/max-pain/atm-iv/wing-iv/expiry-pressure with
    a read of alpha_data's own output for the same active expiry cycle
    (``_active_cycle`` -- the same "nearest expiry >= ts's date" rule
    ``AlphaDataOptionChainSource`` already uses, so both reads agree on which cycle is
    "current" at any ``ts``). ``iv_skew`` is NOT produced by alpha_data's
    ``compute_surface_stats`` today, so ``OptionSurfaceBuilder`` keeps computing it locally.

    Coverage may be partial (only cycles that have had ``compute-surface-stats`` run);
    ``get()`` returns ``None`` for an uncovered cycle so callers can fall back to local
    computation rather than silently zero-filling.
    """

    _COLS = (
        "pcr", "call_wall_strength", "put_wall_strength", "oi_wall_strength",
        "gamma_concentration", "max_pain", "max_pain_rel", "atm_iv", "wing_iv",
        "expiry_pressure",
    )

    def __init__(self, *, interval: str = "5min", paths: AlphaDataPaths | None = None) -> None:
        self._interval = interval
        self._paths = paths or AlphaDataPaths()
        self._cycles_cache: dict[str, list[date]] = {}
        self._stats_cache: dict[tuple[str, date], pd.DataFrame | None] = {}

    def _cycles(self, underlying: str) -> list[date]:
        cached = self._cycles_cache.get(underlying)
        if cached is None:
            cached = _discover_cycles(underlying, self._interval, self._paths)
            self._cycles_cache[underlying] = cached
        return cached

    def _stats(self, underlying: str, expiry: date) -> pd.DataFrame | None:
        key = (underlying, expiry)
        if key not in self._stats_cache:
            path = (
                self._paths.options
                / f"{underlying}_SURFACE_STATS_{expiry.isoformat()}_{self._interval}.parquet"
            )
            df = None
            if path.exists():
                df = pd.read_parquet(path)
                if df.index.tz is not None:
                    df.index = df.index.tz_convert("UTC").tz_localize(None)
                df = df.sort_index()
            self._stats_cache[key] = df
        return self._stats_cache[key]

    def get(self, underlying: str, ts: datetime) -> dict[str, float | None] | None:
        """Return the surface-stats row at or before ``ts`` for the active cycle, or
        ``None`` if that cycle has no materialized stats file (uncovered by
        ``compute-surface-stats`` yet)."""
        expiry = _active_cycle(self._cycles(underlying), ts)
        if expiry is None:
            return None
        df = self._stats(underlying, expiry)
        if df is None or df.empty:
            return None
        end = df.index.searchsorted(pd.Timestamp(ts), side="right")
        if end == 0:
            return None
        row = df.iloc[end - 1]
        return {c: (float(row[c]) if c in row.index and pd.notna(row[c]) else None) for c in self._COLS}


_ATM_GREEKS_CYCLE_RE = re.compile(r"_ATM_(CE|PE)_(\d{4}-\d{2}-\d{2})_(\w+)_greeks\.parquet$")


class AlphaDataAtmGreeksLoader:
    """Per-bar ATM call/put delta from alpha_data's precomputed dedicated rolling-ATM
    series (``{underlying}_ATM_{CE,PE}_{expiry}_{interval}_greeks.parquet``, generated
    by ``pipelines/option_greeks.py``) -- distinct from the multi-strike
    ``SURFACE_{CE,PE}`` files ``AlphaDataOptionChainSource``/``AlphaDataSurfaceStatsLoader``
    read: a separate, single-rolling-ATM-strike series re-struck to track spot, not one
    token within the ATM+-N surface grid.

    2026-07-15: added after the feature IC diagnostic found ``atm_call_delta``/
    ``atm_put_delta`` sourced from this series to be the single strongest directional
    signal in a 124-feature evaluation (rank #1/#2 by |IC| against both forward return
    and barrier-edge, fold-stable across all 5 chronological folds --
    ``runs/feature_ic_report_expanded.csv``). ``OptionSurfaceBuilder`` already computes
    a local fallback from the surface grid's own ATM-token row (``_derived_features``),
    this loader overrides it with the dedicated series when available -- same pattern
    as ``AlphaDataSurfaceStatsLoader``/``atm_iv``.

    Uses its own independent nearest-active-expiry selection (matches
    ``_active_cycle``'s "nearest expiry >= ts's date" rule) rather than sharing
    ``AlphaDataOptionChainSource``'s cache, since it reads an entirely different file
    series with its own (overlapping) per-cycle date ranges.
    """

    def __init__(self, *, interval: str = "5min", paths: AlphaDataPaths | None = None) -> None:
        self._interval = interval
        self._paths = paths or AlphaDataPaths()
        self._cycles_cache: dict[tuple[str, str], list[date]] = {}
        self._greeks_cache: dict[tuple[str, str, date], pd.DataFrame | None] = {}

    def _cycles(self, underlying: str, side: str) -> list[date]:
        key = (underlying, side)
        cached = self._cycles_cache.get(key)
        if cached is None:
            expiries: set[date] = set()
            for p in self._paths.options.glob(f"{underlying}_ATM_{side}_*_{self._interval}_greeks.parquet"):
                m = _ATM_GREEKS_CYCLE_RE.search(p.name)
                if m:
                    expiries.add(date.fromisoformat(m.group(2)))
            cached = sorted(expiries)
            self._cycles_cache[key] = cached
        return cached

    def _greeks(self, underlying: str, side: str, expiry: date) -> pd.DataFrame | None:
        key = (underlying, side, expiry)
        if key not in self._greeks_cache:
            path = (
                self._paths.options
                / f"{underlying}_ATM_{side}_{expiry.isoformat()}_{self._interval}_greeks.parquet"
            )
            df = None
            if path.exists():
                df = pd.read_parquet(path)
                if df.index.tz is not None:
                    df.index = df.index.tz_convert("UTC").tz_localize(None)
                df = df.sort_index()
            self._greeks_cache[key] = df
        return self._greeks_cache[key]

    def _delta_at(self, underlying: str, side: str, ts: datetime) -> float | None:
        expiry = _active_cycle(self._cycles(underlying, side), ts)
        if expiry is None:
            return None
        df = self._greeks(underlying, side, expiry)
        if df is None or df.empty or "delta" not in df.columns:
            return None
        end = df.index.searchsorted(pd.Timestamp(ts), side="right")
        if end == 0:
            return None
        val = df.iloc[end - 1]["delta"]
        return float(val) if pd.notna(val) else None

    def get(self, underlying: str, ts: datetime) -> dict[str, float | None]:
        """Return ``{"atm_call_delta": ..., "atm_put_delta": ...}`` for the active
        cycle at ``ts`` (``None`` for either side not covered)."""
        return {
            "atm_call_delta": self._delta_at(underlying, "CE", ts),
            "atm_put_delta": self._delta_at(underlying, "PE", ts),
        }


class CompositeMarketDataSource:
    """A ``MarketDataSource`` that delegates candle/spot access to ``base`` (e.g.
    ``ParquetMarketDataSource``) but answers ``option_chain`` from
    ``AlphaDataOptionChainSource`` -- composes the two without modifying either.
    """

    def __init__(self, base, option_source: AlphaDataOptionChainSource) -> None:
        self._base = base
        self._options = option_source

    def candle_window(self, symbol: str, end_ts: datetime, lookback: int):
        return self._base.candle_window(symbol, end_ts, lookback)

    def spot(self, symbol: str, ts: datetime) -> float:
        return self._base.spot(symbol, ts)

    def option_chain(self, underlying: str, ts: datetime) -> list[OptionContractSnapshot] | None:
        return self._options.option_chain(underlying, ts)

    def __getattr__(self, name: str):
        # Forward anything else (aligned_frames/candle_frame/timestamp_index/...)
        # straight through to the base source so build_history()'s
        # getattr(source, "aligned_frames", None) duck-typing still works.
        return getattr(self._base, name)


__all__ = ["AlphaDataOptionChainSource", "CompositeMarketDataSource"]
