"""Regime head (SPEC.md §17). Emits logits over the six market regimes.

Class order matches ``REGIME_CLASSES`` (and ``schemas.market_schema.Regime``). Loss applies
cross-entropy; the bridge applies softmax to get ``regime_probs``. ISP: knows nothing of brokers.
"""

from __future__ import annotations

from torch import Tensor, nn

from helion_risk_world.schemas.market_schema import Regime

# Fixed class order for the regime head; keep in sync with the Regime enum.
REGIME_CLASSES: tuple[Regime, ...] = (
    Regime.TREND,
    Regime.RANGE,
    Regime.EVENT,
    Regime.HIGH_VOL,
    Regime.LOW_VOL,
    Regime.CHOP,
)


class RegimeHead(nn.Module):
    """Emit regime logits. Input z: [B, d]. Output: [B, 6] (REGIME_CLASSES order)."""

    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.linear = nn.Linear(latent_dim, len(REGIME_CLASSES))

    def forward(self, z: Tensor) -> Tensor:
        return self.linear(z)  # [B, 6] logits


__all__ = ["RegimeHead", "REGIME_CLASSES"]
