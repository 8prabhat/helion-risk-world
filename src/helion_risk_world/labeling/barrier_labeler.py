"""Triple-barrier labeling for BankNIFTY futures (SPEC.md §11.1).

WRAPS ``quanthelion.labels.triple_barrier.make_triple_barrier_label``; no local reimplementation
of the barrier scan.  HRW adds:
  - futures-vol scaling of the barrier width (σ_t from EWMA of futures returns)
  - conversion to the HRW ``LabelRecord`` schema
  - uniqueness weighting (delegated to uniqueness.py)

Label instrument: BankNIFTY futures continuous OHLC path (NOT spot).  Labels on spot would
mislabel by basis and ignore rolls.  See SPEC.md §11.1.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np

from helion_risk_world.barrier_context import BarrierSpec, barrier_context_series
from helion_risk_world.labeling.uniqueness import apply_uniqueness_weights
from helion_risk_world.schemas.label_schema import Barrier, LabelRecord


class BarrierLabeler:
    """Build triple-barrier labels on a futures close series (SPEC.md §11.1).

    Parameters
    ----------
    H:         trade-management horizon (max holding bars); H = max(horizon_bars). Also passed
               as BarrierSpec.horizon_bars so barrier width scales by sqrt(H) (feature/label
               overhaul Phase 4a) — otherwise barrier width would stay anchored to a single bar's
               EWMA sigma regardless of how long the position is actually held.
    u, d:      upper/lower barrier widths as vol multiples (symmetric u=d=2.0 default)
    vol_span:  EWMA span for σ_t estimation (bars)
    cost_floor: minimum barrier half-width as a return fraction (feature/label overhaul
               Phase 1) — see BarrierSpec.cost_floor_frac. 0.0 preserves old behavior.
    add_uniqueness: if True, compute and attach uniqueness weights via apply_uniqueness_weights
    """

    def __init__(
        self,
        H: int = 12,
        u: float = 2.0,
        d: float = 2.0,
        vol_span: int = 50,
        cost_floor: float = 0.0,
        add_uniqueness: bool = True,
    ) -> None:
        if H < 1:
            raise ValueError("H must be >= 1")
        self._H = H
        self._spec = BarrierSpec(
            stop_mult=d,
            target_mult=u,
            vol_span=vol_span,
            cost_floor_frac=cost_floor,
            horizon_bars=H,
        )
        self._add_uniqueness = add_uniqueness

    def label(
        self,
        timestamps: Sequence[datetime],
        close: Sequence[float],
        symbol: str = "BANKNIFTY_FUT_continuous",
        *,
        open_prices: Sequence[float] | None = None,
        high_prices: Sequence[float] | None = None,
        low_prices: Sequence[float] | None = None,
    ) -> list[LabelRecord]:
        """Scan the futures path and return one LabelRecord per bar with a full H-bar future window.

        Uses the quanthelion labeler logic wrapped locally; the uniqueness step
        is delegated to ``labeling/uniqueness.py``.
        """
        ts = list(timestamps)
        px = np.array(close, dtype=float)
        open_arr = np.array(open_prices if open_prices is not None else close, dtype=float)
        high_arr = np.array(high_prices if high_prices is not None else close, dtype=float)
        low_arr = np.array(low_prices if low_prices is not None else close, dtype=float)
        if not (len(open_arr) == len(high_arr) == len(low_arr) == len(px) == len(ts)):
            raise ValueError("timestamps/open/high/low/close must share the same length")
        barrier_rows = barrier_context_series(px, spec=self._spec)
        n = len(px)

        records: list[LabelRecord] = []
        for t in range(max(0, n - self._H)):
            entry_i = t + 1
            e_t = open_arr[entry_i]
            sigma_t = max(float(barrier_rows[t, 0]), 1e-6)
            stop_return = float(barrier_rows[t, 1])
            target_return = float(barrier_rows[t, 2])
            upper = e_t * (1.0 + target_return)
            lower = e_t * (1.0 + stop_return)
            end = t + self._H

            barrier = Barrier.TIMEOUT
            barrier_valid = True
            exit_i = end
            exit_px = px[end]

            for i in range(entry_i, end + 1):
                hit_target = high_arr[i] >= upper
                hit_stop = low_arr[i] <= lower
                if hit_target and hit_stop:
                    barrier = Barrier.AMBIGUOUS
                    barrier_valid = False
                    exit_i = i
                    exit_px = px[i]
                    break
                if hit_target:
                    barrier, exit_i, exit_px = Barrier.TARGET, i, upper
                    break
                if hit_stop:
                    barrier, exit_i, exit_px = Barrier.STOP, i, lower
                    break

            close_path = px[entry_i : exit_i + 1]
            high_path = high_arr[entry_i : exit_i + 1]
            low_path = low_arr[entry_i : exit_i + 1]
            if len(close_path) > 1:
                realized_vol = float(np.std(np.diff(np.log(close_path))))
            else:
                realized_vol = 0.0
            mae = float(max((e_t - low_path.min()) / e_t, 0.0)) if len(low_path) else 0.0
            mfe = float(max((high_path.max() - e_t) / e_t, 0.0)) if len(high_path) else 0.0

            records.append(
                LabelRecord(
                    symbol=symbol,
                    ts=ts[t],
                    label_realized_at=ts[exit_i],
                    horizon_bars=self._H,
                    barrier=barrier,
                    barrier_valid=barrier_valid,
                    entry_price=float(e_t),
                    exit_price=float(exit_px),
                    exit_return=float(exit_px / e_t - 1.0),
                    exit_t=exit_i - t,
                    realized_vol=realized_vol,
                    barrier_sigma=sigma_t,
                    barrier_stop_return=stop_return,
                    barrier_target_return=target_return,
                    barrier_stop_mult=self._spec.stop_mult,
                    barrier_target_mult=self._spec.target_mult,
                    barrier_vol_span=self._spec.vol_span,
                    barrier_cost_floor_frac=self._spec.cost_floor_frac,
                    mae=mae,
                    mfe=mfe,
                    uniqueness_weight=None,
                )
            )

        if self._add_uniqueness:
            records = apply_uniqueness_weights(records)
        return records


__all__ = ["BarrierLabeler"]
