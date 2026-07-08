from __future__ import annotations

from collections import defaultdict, deque

from torch import Tensor


class RegimeMemory:
    """Bounded per-regime memory of latent states (SPEC.md §23 innovation 6, Appendix A)."""

    def __init__(self, capacity: int = 512) -> None:
        self._capacity = capacity
        self._memory: dict[int, deque[Tensor]] = defaultdict(lambda: deque(maxlen=self._capacity))

    def update(self, regime_probs: Tensor, z_t: Tensor) -> None:
        regimes = regime_probs.detach()
        states = z_t.detach()
        if regimes.ndim == 1:
            regimes = regimes.unsqueeze(0)
        if states.ndim == 1:
            states = states.unsqueeze(0)
        if regimes.shape[0] != states.shape[0]:
            raise ValueError("regime_probs and z_t must share the batch dimension")

        for regime_id, latent in zip(regimes.argmax(dim=-1), states, strict=False):
            self._memory[int(regime_id.item())].append(latent.cpu().clone())

    def counts(self) -> dict[int, int]:
        return {regime_id: len(items) for regime_id, items in self._memory.items()}
