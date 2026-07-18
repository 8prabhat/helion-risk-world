"""Touch head: will EITHER barrier be hit within the horizon, or timeout? (2026-07-13.)

Part of the decomposed barrier architecture (see `heads/direction_head.py`'s docstring for the
full rationale): rather than one 3-way softmax entangling "will price move enough to hit a
barrier at all" with "which direction", this binary head answers only the first question --
the one the model's own diagnostics show has the strongest, most robust signal (volatility-
ratio). SRP: touch/no-touch only; direction is a separate head.
"""

from __future__ import annotations

from torch import Tensor, nn


class TouchHead(nn.Module):
    """Emit a single touch-vs-timeout logit. Input z: [B, d]. Output: [B] (raw logit).

    Stays a bare ``Linear`` (2026-07-13 4-way ablation: {Linear,MLP} x {plain BCE, asymmetric
    loss}, calibrated on real data). Diagnosis was that it over-predicts "touched" (89% vs a
    true 55.3% base rate); both the obvious fixes -- more capacity (MLP) and asymmetric loss
    (higher gamma_neg to penalize confidently-wrong negatives) -- were tried and made macro_f1
    WORSE, not better (0.317 -> 0.246 and 0.232 respectively; MLP+ASL combined -> 0.183,
    collapsing to ~constant "stop"). This modest dataset (~24k train rows) appears to reward
    the smaller-capacity, plain-loss head; keep it simple until there's a specific reason not
    to.
    """

    def __init__(self, latent_dim: int = 128, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.mlp = nn.Sequential(nn.Linear(latent_dim, 1))

    def forward(self, z: Tensor) -> Tensor:
        return self.mlp(z).squeeze(-1)  # [B]


__all__ = ["TouchHead"]
