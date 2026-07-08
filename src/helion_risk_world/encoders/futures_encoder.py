"""FuturesEncoder — encodes BankNIFTY futures microstructure signals (SPEC.md §13, §9.2).

The V1 "derivatives signal" comes entirely from FUTURES (spot volume/OI ≡ 0 on NSE; §9.2).
Features encoded here:
  - basis = F_t − S_t  (and % of spot)
  - futures OI and ΔOI
  - futures volume z-score
  - calendar spread = F_near − F_next
  - DTE (days-to-expiry) and roll_flag
  - OI-flow state: 4-state classification (long-buildup/short-covering/short-buildup/unwinding)
    + signed ΔOI magnitude
  - oi_available: 1.0 when OI is a real observed value for that bar, 0.0 when missing/NaN
  - oi_basis_interaction: d_oi * sign(basis change) — genuine accumulation vs
    short-covering-driven basis moves (feature/label overhaul Phase 2)

Input: [B, T, F] tensor of futures features (T time steps, F features per step).
Output: [B, d] encoded embedding.
SRP: encode futures microstructure only.  ISP: no broker logic, no account fields.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

OI_FLOW_CLASSES = 4  # long_buildup, short_covering, short_buildup, long_unwinding


class FuturesEncoder(nn.Module):
    """Temporal-MLP encoder for futures microstructure.  [B, T, F] → [B, out_dim].

    Uses a small 1-D temporal convolution over the lookback window followed by
    global average pooling and a projection MLP.  Compact and parameter-efficient
    for the ~28k-bar futures window.
    """

    def __init__(
        self,
        in_features: int,
        out_dim: int = 64,
        hidden_dim: int = 128,
        kernel_size: int = 3,
        layers: int = 2,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")
        self.in_features = in_features
        self.out_dim = out_dim
        self.layers = layers

        conv_layers: list[nn.Module] = []
        in_channels = in_features
        for _ in range(layers):
            conv_layers.append(
                nn.Conv1d(
                    in_channels,
                    hidden_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            conv_layers.append(nn.SiLU())
            in_channels = hidden_dim
        self.conv = nn.Sequential(*conv_layers)
        # Global average pool then project
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, T, F] → [B, out_dim]."""
        if x.ndim != 3:
            raise ValueError(f"FuturesEncoder expects [B, T, F]; got {tuple(x.shape)}")
        # Conv1d expects [B, F, T]
        h = self.conv(x.transpose(1, 2))      # [B, hidden, T]
        h = h.mean(dim=-1)                    # [B, hidden]  global average pool
        return self.norm(self.proj(h))        # [B, out_dim]


def build_futures_feature_tensor(
    basis: Tensor,
    oi: Tensor,
    d_oi: Tensor,
    volume_zscore: Tensor,
    calendar_spread: Tensor,
    dte_norm: Tensor,
    roll_flag: Tensor,
    oi_flow_onehot: Tensor,
    d_oi_mag: Tensor,
    oi_available: Tensor,
    oi_basis_interaction: Tensor,
) -> Tensor:
    """Concatenate futures features into a single [B, T, F] tensor.

    All inputs must be [B, T] except oi_flow_onehot [B, T, 4].
    This is the single definition of the futures feature vector — shared by
    training, backtest, and paper trading (DRY, SPEC.md §12).

    ``oi_available`` (review Idea #5): 1.0 where OI is a real observed value,
    0.0 where it's missing/NaN — see data/futures_window_builder.py::_oi_availability
    for the canonical (currently the only actually-used) implementation of this
    feature layout; kept in sync here since this builder is DRY's single
    definition even though no caller currently exercises it.

    ``oi_basis_interaction`` (feature/label overhaul Phase 2): d_oi * sign(basis
    change) — see data/futures_window_builder.py::_oi_basis_interaction, same
    kept-in-sync-but-unused status as oi_available above.
    """
    scalars = [basis, oi, d_oi, volume_zscore, calendar_spread, dte_norm, roll_flag, d_oi_mag,
               oi_available, oi_basis_interaction]
    scalar_stack = torch.stack(scalars, dim=-1)           # [B, T, 10]
    return torch.cat([scalar_stack, oi_flow_onehot], dim=-1)  # [B, T, 14]


FUTURES_FEATURE_DIM = 14  # matches build_futures_feature_tensor output

__all__ = ["FuturesEncoder", "build_futures_feature_tensor", "FUTURES_FEATURE_DIM", "OI_FLOW_CLASSES"]
