"""Encoders (SRP: each only encodes its modality). Lazy import — torch loaded on first use."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

_LAZY: dict[str, str] = {
    "EncoderProtocol": "temporal_encoder",
    "TemporalEncoder": "temporal_encoder",
    "CrossAssetEncoder": "cross_asset_encoder",
    "OptionSurfaceEncoder": "option_surface_encoder",
    "SurfaceTensors": "option_surface_encoder",
    "RegimeEncoder": "regime_encoder",
    "FusionEncoder": "fusion_encoder",
}

__all__ = [
    "CrossAssetEncoder",
    "EncoderProtocol",
    "FusionEncoder",
    "OptionSurfaceEncoder",
    "SurfaceTensors",
    "RegimeEncoder",
    "TemporalEncoder",
]


def __getattr__(name: str) -> Any:  # PEP 562 lazy attribute access
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = importlib.import_module(f"{__name__}.{module}")
    return getattr(mod, name)


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # eager imports for type checkers / IDEs only
    from helion_risk_world.encoders.cross_asset_encoder import CrossAssetEncoder
    from helion_risk_world.encoders.fusion_encoder import FusionEncoder
    from helion_risk_world.encoders.option_surface_encoder import (
        OptionSurfaceEncoder,
        SurfaceTensors,
    )
    from helion_risk_world.encoders.regime_encoder import RegimeEncoder
    from helion_risk_world.encoders.temporal_encoder import EncoderProtocol, TemporalEncoder
