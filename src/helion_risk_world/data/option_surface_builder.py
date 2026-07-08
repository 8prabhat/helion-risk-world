"""Align a raw option chain to an ATM-relative surface (SPEC.md §16, Appendix A, Day 3).

The option chain is represented as an ATM-relative set of strike tokens (ATM-N .. ATM .. ATM+N),
never a naive flat column dump. Missing/illiquid strikes are MASKED, never silently dropped, so the
encoder always sees a fixed-width surface. Derived surface features (PCR, walls, max-pain proxy, IV
skew, ...) are computed here once and reused everywhere (DRY). Market plane only.
"""

from __future__ import annotations

import statistics
from datetime import datetime

import numpy as np

from helion_risk_world.schemas.option_chain_schema import (
    OptionContractSnapshot,
    OptionSurfaceSnapshot,
    OptionType,
    StrikeRow,
)

# Canonical model-input layout for the option surface (kept in one place; DRY).
# Per-strike channels (the "set" the OptionSurfaceEncoder pools over). OI/volume are normalised to
# fractions of the surface total so values are O(1); IV/greeks are left in natural units.
SURFACE_STRIKE_CHANNELS: tuple[str, ...] = (
    "moneyness", "token_norm", "call_oi_frac", "put_oi_frac", "call_doi_frac", "put_doi_frac",
    "call_vol_frac", "put_vol_frac", "call_iv", "put_iv", "call_delta", "put_delta",
    "call_gamma", "put_gamma", "call_theta", "put_theta", "call_vega", "put_vega", "is_masked",
)
# Snapshot-level context features (fed alongside the pooled set).
SURFACE_CONTEXT_FEATURES: tuple[str, ...] = (
    "pcr", "iv_skew", "gamma_concentration", "call_wall_strength", "put_wall_strength",
    "oi_wall_strength", "max_pain_rel", "atm_iv", "wing_iv", "dte_norm",
)


def infer_strike_step(strikes: list[float]) -> float:
    """Infer the strike grid step as the median positive gap between sorted unique strikes."""
    uniq = sorted(set(strikes))
    if len(uniq) < 2:
        raise ValueError("need >= 2 distinct strikes to infer a step")
    diffs = [b - a for a, b in zip(uniq, uniq[1:], strict=False) if b > a]
    return float(statistics.median(diffs))


def _nearest(value: float, candidates: list[float]) -> float:
    return min(candidates, key=lambda k: abs(k - value))


class OptionSurfaceBuilder:
    """Align a raw option chain to an ATM-relative surface (SPEC.md §16).

    SRP: ATM alignment + surface derived-feature computation only. Missing strikes are masked,
    never silently dropped.
    """

    def __init__(self, n_strikes: int = 5) -> None:
        if n_strikes < 1:
            raise ValueError("n_strikes must be >= 1")
        self._n_strikes = n_strikes

    def align_to_atm(
        self, chain: list[OptionContractSnapshot], spot: float, ts: datetime
    ) -> OptionSurfaceSnapshot:
        """Build the 2N+1 strike-token surface centred on the ATM strike (SPEC.md Appendix A).

        ``chain`` should already be point-in-time filtered (``available_at <= ts``); a defensive
        check raises if not. Returns an ``OptionSurfaceSnapshot`` with per-token call/put rows and
        derived surface features. Tokens with no contract are masked (``is_masked=True``).
        """
        if not chain:
            raise ValueError("cannot align an empty option chain")
        for c in chain:
            if c.available_at > ts:
                raise ValueError(
                    f"point-in-time violation: contract available_at {c.available_at} > ts {ts}"
                )

        underlying = chain[0].underlying
        strikes = [c.strike for c in chain]
        atm = _nearest(spot, strikes)
        step = infer_strike_step(strikes)

        # Index contracts by (strike, type) using the nearest grid strike to tolerate float noise.
        grid_strikes = sorted(set(strikes))
        by_key: dict[tuple[float, OptionType], OptionContractSnapshot] = {}
        for c in chain:
            snapped = _nearest(c.strike, grid_strikes)
            by_key[(snapped, c.opt_type)] = c

        rows: list[StrikeRow] = []
        selected: list[OptionContractSnapshot] = []
        for i in range(-self._n_strikes, self._n_strikes + 1):
            target = atm + i * step
            grid = _nearest(target, grid_strikes)
            # Only treat as present if the grid strike is within half a step of the target token.
            present = abs(grid - target) <= step / 2.0
            call = by_key.get((grid, OptionType.CALL)) if present else None
            put = by_key.get((grid, OptionType.PUT)) if present else None
            masked = call is None and put is None
            if call is not None:
                selected.append(call)
            if put is not None:
                selected.append(put)
            rows.append(
                StrikeRow(
                    strike=target,
                    token=i,
                    is_masked=masked,
                    call_oi=getattr(call, "oi", None),
                    put_oi=getattr(put, "oi", None),
                    call_d_oi=getattr(call, "d_oi", None),
                    put_d_oi=getattr(put, "d_oi", None),
                    call_volume=getattr(call, "volume", None),
                    put_volume=getattr(put, "volume", None),
                    call_iv=getattr(call, "iv", None),
                    put_iv=getattr(put, "iv", None),
                    call_delta=getattr(call, "delta", None),
                    put_delta=getattr(put, "delta", None),
                    call_gamma=getattr(call, "gamma", None),
                    put_gamma=getattr(put, "gamma", None),
                    call_theta=getattr(call, "theta", None),
                    put_theta=getattr(put, "theta", None),
                    call_vega=getattr(call, "vega", None),
                    put_vega=getattr(put, "vega", None),
                )
            )

        if not selected:
            raise ValueError("no contracts fell within the ATM-relative window")
        available_at = max(c.available_at for c in selected)
        if available_at > ts:
            raise ValueError("point-in-time violation in selected surface contracts")
        dte = float(statistics.median([c.dte for c in selected]))

        derived = self._derived_features(rows, atm, step)
        return OptionSurfaceSnapshot(
            underlying=underlying,
            ts=ts,
            available_at=available_at,
            atm_strike=atm,
            dte=dte,
            strikes=rows,
            **derived,
        )

    def _derived_features(
        self, rows: list[StrikeRow], atm: float, step: float
    ) -> dict[str, float | None]:
        """Compute PCR, walls, max-pain proxy, IV skew, ATM/wing IV, expiry pressure."""

        def _sum(attr: str) -> float:
            return float(sum(getattr(r, attr) or 0.0 for r in rows))

        total_call_oi = _sum("call_oi")
        total_put_oi = _sum("put_oi")
        pcr = (total_put_oi / total_call_oi) if total_call_oi > 0 else None

        # Walls: concentration of OI at the single strongest strike.
        call_ois = [r.call_oi or 0.0 for r in rows]
        put_ois = [r.put_oi or 0.0 for r in rows]
        oi_totals = [c + p for c, p in zip(call_ois, put_ois, strict=False)]
        call_wall = (max(call_ois) / total_call_oi) if total_call_oi > 0 else None
        put_wall = (max(put_ois) / total_put_oi) if total_put_oi > 0 else None
        grand = sum(oi_totals)
        oi_wall = (max(oi_totals) / grand) if grand > 0 else None

        # ATM IV / wing IV / skew.
        atm_row = next((r for r in rows if r.token == 0), None)
        atm_pair = (atm_row.call_iv, atm_row.put_iv) if atm_row else ()
        atm_ivs = [v for v in atm_pair if v is not None]
        atm_iv = float(np.mean(atm_ivs)) if atm_ivs else None
        wing_ivs = [
            v
            for r in rows
            if abs(r.token) == self._n_strikes
            for v in (r.call_iv, r.put_iv)
            if v is not None
        ]
        wing_iv = float(np.mean(wing_ivs)) if wing_ivs else None
        otm_put_ivs = [r.put_iv for r in rows if r.token < 0 and r.put_iv is not None]
        otm_call_ivs = [r.call_iv for r in rows if r.token > 0 and r.call_iv is not None]
        iv_skew = (
            float(np.mean(otm_put_ivs) - np.mean(otm_call_ivs))
            if otm_put_ivs and otm_call_ivs
            else None
        )

        # Gamma concentration (OI-weighted), if greeks present.
        def _row_gamma(r: StrikeRow) -> float:
            call = (r.call_gamma or 0.0) * (r.call_oi or 0.0)
            put = (r.put_gamma or 0.0) * (r.put_oi or 0.0)
            return call + put

        gamma_mass = [_row_gamma(r) for r in rows]
        total_gamma = sum(gamma_mass)
        gamma_concentration = (max(gamma_mass) / total_gamma) if total_gamma > 0 else None

        max_pain = self._max_pain(rows)
        # Expiry pressure: normalised closeness to expiry (proxy; refined in V2). dte unknown here,
        # so express as OI mass — higher total OI implies stronger pin pressure near expiry.
        expiry_pressure = float(grand) if grand > 0 else None

        return {
            "pcr": pcr,
            "iv_skew": iv_skew,
            "gamma_concentration": gamma_concentration,
            "call_wall_strength": call_wall,
            "put_wall_strength": put_wall,
            "oi_wall_strength": oi_wall,
            "max_pain_proxy": max_pain,
            "expiry_pressure": expiry_pressure,
            "atm_iv": atm_iv,
            "wing_iv": wing_iv,
        }

    @staticmethod
    def _max_pain(rows: list[StrikeRow]) -> float | None:
        """Max-pain strike: the settlement strike minimising total option-writer payout."""
        strikes = [r.strike for r in rows]
        if not strikes:
            return None
        best_strike, best_pain = None, None
        for settle in strikes:
            pain = 0.0
            for r in rows:
                pain += (r.call_oi or 0.0) * max(settle - r.strike, 0.0)
                pain += (r.put_oi or 0.0) * max(r.strike - settle, 0.0)
            if best_pain is None or pain < best_pain:
                best_strike, best_pain = settle, pain
        return best_strike


def featurize_surface(
    snapshot: OptionSurfaceSnapshot,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn an ``OptionSurfaceSnapshot`` into model-ready arrays (torch-free; DRY).

    Returns:
        grid:    [S, C] float32, one row per strike token, channels = ``SURFACE_STRIKE_CHANNELS``.
        mask:    [S]    float32, 1.0 for present strikes, 0.0 for masked (missing/illiquid) ones.
        context: [K]    float32, snapshot-level features = ``SURFACE_CONTEXT_FEATURES``.

    OI/volume are normalised to fractions of the surface total so the encoder sees O(1) inputs;
    IV/greeks are left in natural units. None values become 0.0. The grid is permutation-stable
    (ordered by token), which the DeepSets encoder pools over.
    """
    rows = snapshot.strikes
    atm = snapshot.atm_strike or 1.0
    max_tok = max((abs(r.token) for r in rows), default=1) or 1

    def _tot(attr: str) -> float:
        return float(sum(getattr(r, attr) or 0.0 for r in rows)) or 1.0

    tot_call_oi, tot_put_oi = _tot("call_oi"), _tot("put_oi")
    tot_call_doi, tot_put_doi = _tot("call_d_oi"), _tot("put_d_oi")
    tot_call_vol, tot_put_vol = _tot("call_volume"), _tot("put_volume")

    def _g(r: StrikeRow, attr: str) -> float:
        return float(getattr(r, attr) or 0.0)

    grid = np.zeros((len(rows), len(SURFACE_STRIKE_CHANNELS)), dtype=np.float32)
    mask = np.zeros(len(rows), dtype=np.float32)
    for i, r in enumerate(rows):
        mask[i] = 0.0 if r.is_masked else 1.0
        grid[i] = [
            r.strike / atm - 1.0,                       # moneyness
            r.token / max_tok,                          # token_norm
            _g(r, "call_oi") / tot_call_oi,
            _g(r, "put_oi") / tot_put_oi,
            _g(r, "call_d_oi") / tot_call_doi,
            _g(r, "put_d_oi") / tot_put_doi,
            _g(r, "call_volume") / tot_call_vol,
            _g(r, "put_volume") / tot_put_vol,
            _g(r, "call_iv"), _g(r, "put_iv"),
            _g(r, "call_delta"), _g(r, "put_delta"),
            _g(r, "call_gamma"), _g(r, "put_gamma"),
            _g(r, "call_theta"), _g(r, "put_theta"),
            _g(r, "call_vega"), _g(r, "put_vega"),
            1.0 if r.is_masked else 0.0,
        ]

    max_pain_rel = ((snapshot.max_pain_proxy or atm) - atm) / atm
    context = np.array(
        [
            snapshot.pcr or 0.0,
            snapshot.iv_skew or 0.0,
            snapshot.gamma_concentration or 0.0,
            snapshot.call_wall_strength or 0.0,
            snapshot.put_wall_strength or 0.0,
            snapshot.oi_wall_strength or 0.0,
            max_pain_rel,
            snapshot.atm_iv or 0.0,
            snapshot.wing_iv or 0.0,
            snapshot.dte / 30.0,
        ],
        dtype=np.float32,
    )
    return grid, mask, context
