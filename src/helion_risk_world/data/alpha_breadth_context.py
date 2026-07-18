"""Compatibility adapter: per-bar breadth/dispersion aligned to a candle-tensor timestamp
index, backed by alpha_data's real cross-sectional constituent breadth pipeline instead of
``FeatureBuilder`` recomputing an equivalent statistic locally from the candle tensor's own
trailing N-bar returns (feature-onboarding pass).

This is a genuine CAPABILITY UPGRADE, not a parity port: alpha_data's
``pipelines/breadth_features.py::compute_breadth`` uses the index's real constituent
universe (HDFCBANK/ICICIBANK/SBIN/AXISBANK/KOTAKBANK) with an HDFC-blackout mask already
applied, and additionally derives ``breadth_index_divergence`` (mean constituent return
minus the index's own return) -- a signal ``FeatureBuilder``'s local computation never
captured at all (that scalar is instead merged into the regime vector, see
``alpha_regime_context.py``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from quanthelion.utils.logging import get_logger

from alpha_data.io.paths import DataPaths as AlphaDataPaths

log = get_logger(__name__)

_NEUTRAL_BREADTH = 0.5
_NEUTRAL_DISPERSION = 0.0


class AlphaDataBreadthLoader:
    """Per-bar ``pct_advancing``/``return_dispersion`` from alpha_data's breadth parquet."""

    def __init__(
        self,
        index_underlying: str = "BANKNIFTY",
        *,
        interval: str = "5min",
        paths: AlphaDataPaths | None = None,
    ) -> None:
        self._paths = paths or AlphaDataPaths()
        path = self._paths.features / f"{index_underlying}_breadth_{interval}.parquet"
        self._df: pd.DataFrame | None = None
        if path.exists():
            df = pd.read_parquet(path)[["pct_advancing", "return_dispersion"]]
            idx = pd.to_datetime(df.index)
            if idx.tz is not None:
                idx = idx.tz_convert("UTC").tz_localize(None)
            df.index = idx
            self._df = df.sort_index()
        else:
            log.warning("no alpha_data breadth context found", path=str(path))

    def aligned(self, index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
        """Return (breadth, dispersion) forward-filled onto ``index`` (naive UTC/local).

        Falls back to a neutral (no cross-sectional signal) constant when no breadth
        parquet was found -- matching ``FeatureBuilder``'s previous single-asset-universe
        fallback (``breadth=0.5``, ``dispersion=0.0``).
        """
        n = len(index)
        if self._df is None:
            return (
                np.full(n, _NEUTRAL_BREADTH, dtype=np.float32),
                np.full(n, _NEUTRAL_DISPERSION, dtype=np.float32),
            )
        target = pd.DatetimeIndex(index)
        if target.tz is not None:
            target = target.tz_convert("UTC").tz_localize(None)
        aligned = self._df.reindex(target, method="ffill")
        breadth = aligned["pct_advancing"].fillna(_NEUTRAL_BREADTH).to_numpy(dtype=np.float32)
        dispersion = aligned["return_dispersion"].fillna(_NEUTRAL_DISPERSION).to_numpy(dtype=np.float32)
        return breadth, dispersion


__all__ = ["AlphaDataBreadthLoader"]
