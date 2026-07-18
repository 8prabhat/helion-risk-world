"""Meta-label head (2026-07-18): binary, cost-aware "is this trade worth taking"
probability, conditioned on the momentum-based PRIMARY side (see
``labeling/meta_labels.py`` for why this exists and how the label is built).

Unlike the other heads, this one's target genuinely depends on an input the model
doesn't otherwise see: which side (long/short) is being proposed. ``primary_side``
is concatenated into the latent before a small MLP rather than passed through the
optional ``context_gate`` pattern other heads use (e.g. ``BarrierHead``), because
it is not optional context here -- the whole prediction is meaningless without it.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MetaLabelHead(nn.Module):
    """Emit meta-label logit. Input z: [B, d], primary_side: [B] in {-1, 0, 1}.
    Output: [B] logit (apply sigmoid for P(trade worth taking | primary_side))."""

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 32) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: Tensor, primary_side: Tensor) -> Tensor:
        side = primary_side.reshape(-1, 1).to(dtype=z.dtype, device=z.device)
        combined = torch.cat([z, side], dim=-1)
        return self.net(combined).squeeze(-1)  # [B] logit


def primary_side_from_candle_features(
    features: Tensor,
    *,
    primary_asset_idx: int = 0,
    log_return_channel: int = 0,
    lookback: int = 12,
) -> Tensor:
    """Torch-native equivalent of ``labeling/meta_labels.py::primary_side_from_close``,
    evaluated directly from a ``[B, A, L, F]`` candle feature tensor already available
    wherever ``forward()`` is called -- no extra data dependency needed to compute the
    SAME primary signal at inference as was used to build training labels.

    ``lookback`` PRICE points span ``lookback - 1`` log-return steps (verified
    identical to the close-ratio computation in ``test_meta_labels.py``), so this sums
    the trailing ``lookback - 1`` values of the primary asset's ``log_return`` channel.
    Gradient-free by construction (``torch.sign``); never contributes to backprop
    through the encoder, matching how this signal is a fixed, model-independent
    primary rather than something learned.
    """
    n_steps = max(1, lookback - 1)
    window = features[:, primary_asset_idx, -n_steps:, log_return_channel]  # [B, n_steps]
    total = window.sum(dim=-1)  # [B]
    return torch.sign(total.detach())


__all__ = ["MetaLabelHead", "primary_side_from_candle_features"]
