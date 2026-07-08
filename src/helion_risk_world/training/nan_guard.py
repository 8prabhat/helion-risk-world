"""Guard against non-finite loss values corrupting a training run (review finding H5).

None of the training loops in this package used to check whether a computed loss was
finite before calling ``.backward()``/``optim.step()``. A single bad batch — e.g. a
division-by-zero in a rolling-vol feature, or a roll-gap bar (see review finding H4) —
can produce a NaN/Inf loss. ``.backward()`` then propagates NaN gradients,
``clip_grad_norm_`` computes a NaN norm, and ``optim.step()`` silently corrupts every
parameter for the rest of the run with no error and no signal beyond a NaN entry in the
loss history. ``skip_if_non_finite`` lets a training loop detect this before it happens
and skip the offending batch instead.
"""

from __future__ import annotations

import torch
from torch import Tensor

from helion_risk_world.integration.quanthelion_adapter import get_logger

_log = get_logger("hrw.training.nan_guard")


def skip_if_non_finite(loss: Tensor, *, context: str) -> bool:
    """Return True (caller should skip this batch) and log a warning if ``loss`` is
    not finite.

    ``context`` is a short, human-readable label (e.g. ``"HRWTrainer.fit epoch=3
    idx=17"``) included in the log line so a skipped batch is traceable to where it
    happened in the run.
    """
    if not bool(torch.isfinite(loss.detach()).all()):
        try:
            value = float(loss.detach())
        except (ValueError, RuntimeError):
            value = float("nan")
        # NOTE: get_logger() returns a plain stdlib logging.Logger here (unlike the
        # structlog-style **kwargs calls elsewhere in this package, which only avoid
        # crashing because they're at INFO level and INFO is disabled by default) —
        # warning() IS enabled by default, so use %-style args, not **kwargs, or this
        # would raise TypeError the first time it actually fires.
        _log.warning("training.non_finite_loss_skipped context=%s loss=%s", context, value)
        return True
    return False


__all__ = ["skip_if_non_finite"]
