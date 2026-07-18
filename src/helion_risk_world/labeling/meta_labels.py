"""Cost-aware meta-labeling (López de Prado, AFML ch.3): a simple PRIMARY signal
proposes a trade side; the META label is the binary question that actually
determines profitability -- "would taking a trade in that direction, held via the
existing triple-barrier exit mechanics, have netted more than round-trip cost."

Why this exists (2026-07-18, see docs/investigation_log.md): the 3-class barrier
head (stop/target/timeout) requires class-weighted CE to avoid majority-class
collapse, which provably distorts predicted probabilities away from true base
rates (see prediction_calibration.py's barrier_prior_offsets). The planner then
has to reconstruct a trading decision from those distorted probabilities through
a multi-step CVaR objective. A binary, cost-aware, side-conditioned "is this trade
worth it" question is a much more direct training target for exactly the decision
the planner needs to make, and needs no class-imbalance-driven reweighting trick
of its own (its natural imbalance is far less extreme, and if it is imbalanced the
model is free to output well-calibrated small probabilities without contradicting
the direction of any prior-shift correction).

The PRIMARY here is deliberately simple and model-independent (a rolling momentum
sign) -- true to meta-labeling's design: the primary supplies a plentiful, cheap,
possibly-mediocre signal, and the ML's whole job is deciding when to trust it, not
generating the signal itself. This also makes labels reproducible without any
trained model in the loop, and the SAME causal function can be evaluated at
inference time from live candle features (see `primary_side_from_log_returns`) so
train and serve compute primary_side identically.
"""

from __future__ import annotations

import numpy as np


def primary_side_from_close(
    close: np.ndarray, idx: int, *, lookback: int = 12
) -> int:
    """Causal momentum-sign primary signal at decision bar ``idx``.

    Uses ``close[idx - lookback + 1 : idx + 1]`` (inclusive of ``idx``, the decision
    bar itself -- never a future bar). Returns +1 (long), -1 (short), or 0 (flat --
    exact-tie or insufficient history; callers should treat 0 as "no bet proposed",
    not as a class of its own).
    """
    start = max(0, idx - lookback + 1)
    window = close[start : idx + 1]
    if len(window) < 2 or window[0] <= 0.0:
        return 0
    momentum = float(window[-1] / window[0] - 1.0)
    if momentum > 0.0:
        return 1
    if momentum < 0.0:
        return -1
    return 0


def primary_side_from_log_returns(log_returns: np.ndarray) -> int:
    """Same primary signal, evaluated from a trailing window of PER-BAR LOG RETURNS
    (e.g. ``candle_features[..., 0]``, the ``log_return`` feature channel already in
    every model input tensor) instead of raw closes. ``sum(log_returns) ==
    log(close[-1] / close[start])`` to machine precision, so this is exactly
    ``primary_side_from_close`` re-derived from a feature tensor already available at
    inference time -- no extra OHLCV dependency needed to compute the SAME primary at
    serve time as was used to build training labels.
    """
    if log_returns.size < 1:
        return 0
    total = float(np.sum(log_returns))
    if total > 0.0:
        return 1
    if total < 0.0:
        return -1
    return 0


def meta_label_for_side(
    primary_side: int, exit_return: float, cost_floor_frac: float
) -> int | None:
    """Binary meta-label: did a trade in ``primary_side``'s direction, held via the
    existing (side-agnostic, long-style) triple-barrier ``exit_return``, net more
    than ``cost_floor_frac`` after accounting for direction?

    ``exit_return`` is the long-style realized return from the triple-barrier engine
    (``entry_price``/``exit_price`` from ``build_labels``, direction-agnostic by
    construction). For a short primary side the P&L is the negation of that return --
    the same "swap for short" convention already used throughout
    ``schemas/prediction_schema.py``. Returns ``None`` when ``primary_side == 0``
    (no trade proposed, so there is no profitability question to answer for this row
    -- callers should exclude these rows from meta-label training entirely rather
    than inventing a label for a bet that was never made).
    """
    if primary_side not in (-1, 1):
        return None
    pnl_in_primary_direction = float(primary_side) * float(exit_return)
    return 1 if pnl_in_primary_direction > float(cost_floor_frac) else 0


def compute_meta_labels(
    close: np.ndarray,
    exit_return: np.ndarray,
    cost_floor_frac: np.ndarray | float,
    *,
    lookback: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized-by-row wrapper: returns ``(primary_side, meta_label)`` arrays, one
    entry per row of ``exit_return`` (aligned 1:1, ``exit_return[i]`` is the label
    already computed for decision bar ``i`` in the same index space as ``close``).

    ``meta_label[i]`` is ``-1`` (sentinel, not a valid 0/1 label) wherever
    ``primary_side[i] == 0`` -- callers must filter on ``primary_side != 0`` before
    using ``meta_label`` for supervision. A sentinel (rather than NaN) keeps the
    return array integer-typed, matching ``primary_side``.
    """
    n = len(exit_return)
    if len(close) < n:
        raise ValueError("close must have at least as many rows as exit_return")
    floors = (
        np.full(n, float(cost_floor_frac))
        if np.isscalar(cost_floor_frac)
        else np.asarray(cost_floor_frac, dtype=float)
    )
    if floors.shape[0] != n:
        raise ValueError("cost_floor_frac array must align with exit_return")

    primary = np.zeros(n, dtype=np.int64)
    meta = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        side = primary_side_from_close(close, i, lookback=lookback)
        primary[i] = side
        if side == 0:
            continue
        label = meta_label_for_side(side, float(exit_return[i]), float(floors[i]))
        meta[i] = -1 if label is None else label
    return primary, meta


__all__ = [
    "compute_meta_labels",
    "meta_label_for_side",
    "primary_side_from_close",
    "primary_side_from_log_returns",
]
