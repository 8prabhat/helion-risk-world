"""Execution instrument-spec resolution.

V1 trades only the BankNIFTY futures path, but the execution layer still needs an explicit
contract model so sizing, costs, and backtests remain executable. Resolution is config-driven and
supports simple symbol normalization for continuous/monthly futures aliases.
"""

from __future__ import annotations

from helion_risk_world.config.execution_config import CostModelConfig, InstrumentSpecConfig


def symbol_lookup_keys(symbol: str) -> tuple[str, ...]:
    """Return candidate config keys for ``symbol`` in most-specific to least-specific order."""
    sym = str(symbol or "").upper()
    if not sym:
        return ()

    keys: list[str] = [sym]
    if "_FUT_" in sym:
        underlying = sym.split("_FUT_", 1)[0]
        keys.extend((f"{underlying}_FUT", underlying))
    elif sym.endswith("FUT"):
        underlying = sym[:-3].rstrip("_")
        if underlying:
            keys.extend((f"{underlying}_FUT", underlying))
    else:
        keys.append(f"{sym}_FUT")

    seen: set[str] = set()
    ordered: list[str] = []
    for key in keys:
        if key and key not in seen:
            seen.add(key)
            ordered.append(key)
    return tuple(ordered)


def resolve_instrument_spec(
    symbol: str,
    cfg: CostModelConfig,
) -> InstrumentSpecConfig | None:
    """Resolve the configured execution spec for ``symbol`` if one exists."""
    for key in symbol_lookup_keys(symbol):
        spec = cfg.instrument_specs.get(key)
        if spec is not None:
            return spec
    return None


__all__ = ["resolve_instrument_spec", "symbol_lookup_keys"]
