"""Trading-utility checkpoint-selection metric (2026-07-18).

``HRWTrainer.fit()``'s default checkpoint selection uses the composite training loss
on the validation split -- a reasonable general-purpose signal, but not the same
question as "would deploying this checkpoint's own decisions have made money." This
module scores checkpoints directly against the meta-label head's own decision rule
(trade iff predicted P(profitable) > threshold) evaluated against ``meta_label``
ground truth, which is already cost-floor-aware (see ``labeling/meta_labels.py``) --
the same signal ``PositionSizer``'s meta-label gate consumes at inference time. Pass
``trading_utility_loss`` as ``HRWTrainer(..., checkpoint_metric=trading_utility_loss)``
to select on this instead of composite loss.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from helion_risk_world.integration.quanthelion_adapter import ModelProtocol
from helion_risk_world.training.trainer import ForecastBatch, _model_forward


@torch.no_grad()
def trading_utility_score(
    model: ModelProtocol,
    batches: Sequence[ForecastBatch],
    device: torch.device,
    *,
    decision_threshold: float = 0.5,
) -> float:
    """Net edge rate of the model's own meta-label trade/no-trade decisions.

    For every row where a primary side was proposed (``primary_side != 0``) AND the
    model's own decision rule would take the trade (predicted
    ``P(profitable) > decision_threshold``), score +1 if ``meta_label == 1``
    (genuinely profitable net of cost) or -1 if ``meta_label == 0``. Average over all
    such "taken" rows across every batch.

    Returns 0.0 -- the same value ``NO_TRADE`` scores in the planner's own mean-CVaR
    objective (``planner/reward_scorer.py``) -- when the model takes zero trades on
    this split, rather than an undefined value or one that rewards total inaction
    with an artificially flat score matching a genuinely edge-free checkpoint.
    """
    was_training = bool(getattr(model, "training", False))
    model.eval()  # type: ignore[attr-defined]
    correct = 0
    incorrect = 0
    for batch in batches:
        batch = batch.to(device)
        if batch.meta_label is None:
            continue
        out = _model_forward(model, batch)
        if "meta_label_logit" not in out or "primary_side" not in out:
            continue
        meta_label = batch.meta_label.reshape(-1)
        primary_side = out["primary_side"].reshape(-1)
        prob = torch.sigmoid(out["meta_label_logit"]).reshape(-1)
        valid = torch.isfinite(meta_label) & (primary_side != 0)
        would_trade = valid & (prob > decision_threshold)
        if not bool(would_trade.any()):
            continue
        taken_labels = meta_label[would_trade]
        correct += int((taken_labels > 0.5).sum().item())
        incorrect += int((taken_labels <= 0.5).sum().item())
    if was_training:
        model.train()  # type: ignore[attr-defined]
    total = correct + incorrect
    if total == 0:
        return 0.0
    return (correct - incorrect) / total


def trading_utility_loss(
    model: ModelProtocol,
    batches: Sequence[ForecastBatch],
    device: torch.device,
    *,
    decision_threshold: float = 0.5,
) -> float:
    """Negated ``trading_utility_score`` -- a lower-is-better scalar, the contract
    ``HRWTrainer.fit()``'s ``checkpoint_metric`` requires (drop-in replacement for
    composite val_loss's selection role)."""
    return -trading_utility_score(model, batches, device, decision_threshold=decision_threshold)


__all__ = ["trading_utility_loss", "trading_utility_score"]
