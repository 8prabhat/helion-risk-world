"""HRW training loop (SPEC.md §20, Day 4).

``HRWTrainer`` owns a minimal, deterministic optimisation loop over an iterable of ``ForecastBatch``
mini-batches. It depends on ``ModelProtocol`` + a loss callable (DIP). Checkpoint/early-stopping
callbacks are available via the Quanthelion ``TrainerAdapter`` and will be wired once their concrete
APIs are pinned; Day 4 keeps the loop self-contained and testable.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import inspect

import torch
from torch import Tensor

from helion_risk_world.config.training_config import TrainingConfig
from helion_risk_world.encoders.option_surface_encoder import SurfaceTensors
from helion_risk_world.integration.quanthelion_adapter import ModelProtocol, get_logger
from helion_risk_world.training.nan_guard import skip_if_non_finite

_log = get_logger("hrw.training")

# A loss callable maps (model_output_dict, target_batch) -> scalar loss tensor.
LossFn = Callable[[dict[str, Tensor], "ForecastBatch"], Tensor]


@dataclass
class ForecastBatch:
    """One training mini-batch for the forecaster (market plane + supervised targets)."""

    features: Tensor                       # [B, A, L, F]
    forward_return: Tensor                 # [B]
    direction: Tensor                      # [B] long in {0,1,2} (down/flat/up); kept for diagnostics
    regime: Tensor | None = None           # [B] long in {0..5} (regime LABEL); optional
    futures: Tensor | None = None          # [B, T, FUTURES_FEATURE_DIM] V1 microstructure; optional
    regime_context: Tensor | None = None   # [B, K] regime/event input for the regime encoder
    realized_vol: Tensor | None = None     # [B] realized volatility target; optional
    vol_baseline: Tensor | None = None     # [B] causal EWMA vol at decision time (barrier_sigma);
                                            # normalizes the volatility loss to a vol-RATIO target,
                                            # which out-of-sample diagnostics show is far more
                                            # learnable than raw vol level (regime-drift in the raw
                                            # level otherwise dominates the loss with non-stationary
                                            # scale shifts unrelated to genuine predictive skill)
    mae: Tensor | None = None              # [B] max adverse excursion target; optional
    mfe: Tensor | None = None              # [B] max favorable excursion target; optional
    return_weight: Tensor | None = None    # [B] optional mask/weight for timeout-return supervision
    barrier: Tensor | None = None          # [B] long in {0,1,2} (stop/target/neither); optional
    barrier_weight: Tensor | None = None   # [B] barrier-supervision mask/weight; optional
    sample_weight: Tensor | None = None    # [B] optional uniqueness/sample weighting
    horizon_returns: Tensor | None = None  # [B, H] multi-horizon targets for world-model training
    horizon_volatility: Tensor | None = None  # [B, H] realized vol per training horizon
    horizon_mae: Tensor | None = None      # [B, H] max adverse excursion per horizon
    horizon_mfe: Tensor | None = None      # [B, H] max favorable excursion per horizon
    barrier_context: Tensor | None = None  # [B, 3] sigma + explicit stop/target returns
    surface_grid: Tensor | None = None     # [B, S, C] option-surface strike grid; optional
    surface_mask: Tensor | None = None     # [B, S] option-surface strike mask; optional
    surface_context: Tensor | None = None  # [B, K] option-surface snapshot context; optional
    primary_side: Tensor | None = None     # [B] float in {-1,0,1}; meta-labeling primary signal
                                            # from the label file (labeling/meta_labels.py) --
                                            # passed explicitly to forward() so supervision uses
                                            # the SAME primary_side the meta_label column was
                                            # computed against, not a freshly-recomputed one.
    meta_label: Tensor | None = None       # [B] float in {0,1}, NaN where primary_side==0
    target_horizons: tuple[int, ...] = ()  # metadata only; validated by world-model losses

    def to(self, device: torch.device) -> ForecastBatch:
        def _mv(t: Tensor | None) -> Tensor | None:
            return t.to(device) if t is not None else None

        return ForecastBatch(
            features=self.features.to(device),
            forward_return=self.forward_return.to(device),
            direction=self.direction.to(device),
            regime=_mv(self.regime),
            futures=_mv(self.futures),
            regime_context=_mv(self.regime_context),
            realized_vol=_mv(self.realized_vol),
            vol_baseline=_mv(self.vol_baseline),
            mae=_mv(self.mae),
            mfe=_mv(self.mfe),
            return_weight=_mv(self.return_weight),
            barrier=_mv(self.barrier),
            barrier_weight=_mv(self.barrier_weight),
            sample_weight=_mv(self.sample_weight),
            horizon_returns=_mv(self.horizon_returns),
            horizon_volatility=_mv(self.horizon_volatility),
            horizon_mae=_mv(self.horizon_mae),
            horizon_mfe=_mv(self.horizon_mfe),
            barrier_context=_mv(self.barrier_context),
            surface_grid=_mv(self.surface_grid),
            surface_mask=_mv(self.surface_mask),
            surface_context=_mv(self.surface_context),
            primary_side=_mv(self.primary_side),
            meta_label=_mv(self.meta_label),
            target_horizons=self.target_horizons,
        )


def resolve_device(name: str) -> torch.device:
    """Resolve ``cfg.device``; 'auto' -> mps (Mac Studio) else cuda else cpu."""
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class HRWTrainer:
    """Owns the HRW optimisation loop; brokers Quanthelion utilities (DIP).

    Depends on ModelProtocol + a loss callable, never concrete classes (SPEC.md §26).
    """

    def __init__(
        self,
        model: ModelProtocol,
        loss: LossFn,
        cfg: TrainingConfig,
        *,
        checkpoint_metric: Callable[[ModelProtocol, Sequence["ForecastBatch"], torch.device], float]
        | None = None,
    ) -> None:
        """``checkpoint_metric`` (2026-07-18, optional): when given, checkpoint
        selection and early stopping use THIS value on ``val_batches`` instead of the
        composite training loss -- e.g. ``training.checkpoint_metrics.trading_utility_loss``,
        which scores "would this checkpoint's own meta-label decision rule have made
        money on held-out data" rather than a generic weighted-loss composite. Must
        return a LOWER-IS-BETTER scalar (like a loss) to be a drop-in replacement for
        the default composite val_loss selection signal. ``None`` (default) preserves
        the original behavior exactly: select on composite val_loss.
        """
        self._model = model
        self._loss = loss
        self._cfg = cfg
        self._checkpoint_metric = checkpoint_metric
        self.history: list[float] = []
        self.val_history: list[float] = []
        self.val_metric_history: list[float] = []
        self.best_epoch: int | None = None
        self.n_skipped_batches: int = 0

    def fit(
        self,
        batches: Sequence[ForecastBatch],
        *,
        epochs: int | None = None,
        val_batches: Sequence[ForecastBatch] | None = None,
    ) -> ModelProtocol:
        """Train over ``batches`` for ``epochs`` (default ``cfg.max_epochs``). Returns the model.

        Deterministic given ``cfg.seed``. Records mean per-epoch loss in ``self.history``.
        """
        torch.manual_seed(self._cfg.seed)
        device = resolve_device(self._cfg.device)
        model = self._model
        model.to(device)  # type: ignore[attr-defined]
        model.train()  # type: ignore[attr-defined]

        optim = torch.optim.Adam(
            model.parameters(), lr=self._cfg.lr, weight_decay=self._cfg.weight_decay
        )
        n_epochs = epochs if epochs is not None else self._cfg.max_epochs
        if not batches:
            raise ValueError("fit requires a non-empty sequence of batches")

        self.history = []
        self.val_history = []
        self.val_metric_history = []
        self.best_epoch = None
        self.n_skipped_batches = 0
        best_state: dict[str, Tensor] | None = None
        best_val = float("inf")
        bad_epochs = 0

        for epoch in range(n_epochs):
            running, total_mass = 0.0, 0.0
            accum_steps = self._cfg.grad_accum_steps
            optim.zero_grad()
            batch_order = _epoch_batch_indices(len(batches), seed=self._cfg.seed, epoch=epoch)
            for idx, batch_idx in enumerate(batch_order, start=1):
                batch = batches[batch_idx]
                batch = batch.to(device)
                output = _model_forward(model, batch)
                loss = self._loss(output, batch)
                if skip_if_non_finite(loss, context=f"HRWTrainer.fit epoch={epoch + 1} idx={idx}"):
                    self.n_skipped_batches += 1
                    optim.zero_grad()
                    continue
                (loss / accum_steps).backward()
                batch_mass = _batch_mass(batch)
                should_step = (idx % accum_steps == 0) or (idx == len(batch_order))
                if should_step:
                    if self._cfg.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), self._cfg.grad_clip_norm
                        )
                    optim.step()
                    optim.zero_grad()
                running += float(loss.detach()) * batch_mass
                total_mass += batch_mass
            mean_loss = running / max(total_mass, 1e-8)
            self.history.append(mean_loss)
            if val_batches:
                val_loss = self._evaluate(model, val_batches, device)
                self.val_history.append(val_loss)
                # Composite val_loss is always computed (cheap, and useful for
                # diagnostics/logging even when a custom checkpoint_metric drives
                # selection). Selection/early-stopping use the custom metric INSTEAD
                # of val_loss when one is configured (2026-07-18) -- see __init__'s
                # docstring.
                selection_value = val_loss
                if self._checkpoint_metric is not None:
                    selection_value = self._checkpoint_metric(model, val_batches, device)
                    self.val_metric_history.append(selection_value)
                if selection_value < best_val - 1e-8:
                    best_val = selection_value
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                    self.best_epoch = epoch + 1
                    bad_epochs = 0
                else:
                    bad_epochs += 1
            if epoch == 0 or (epoch + 1) % max(1, n_epochs // 5) == 0:
                payload = {"epoch": epoch + 1, "loss": round(mean_loss, 6)}
                if val_batches and self.val_history:
                    payload["val_loss"] = round(self.val_history[-1], 6)
                if self.val_metric_history:
                    payload["val_checkpoint_metric"] = round(self.val_metric_history[-1], 6)
                _log.info("hrw.train.epoch", **payload)
                print(
                    "TRAIN epoch={epoch} loss={loss}{val_loss}".format(
                        epoch=payload["epoch"],
                        loss=payload["loss"],
                        val_loss=(
                            f" val_loss={payload['val_loss']}"
                            if "val_loss" in payload
                            else ""
                        ),
                    ),
                    flush=True,
                )
            if (
                val_batches
                and self._cfg.early_stopping_patience > 0
                and bad_epochs >= self._cfg.early_stopping_patience
            ):
                _log.info(
                    "hrw.train.early_stop",
                    epoch=epoch + 1,
                    best_epoch=self.best_epoch,
                    best_val_loss=round(best_val, 6),
                )
                print(
                    "TRAIN early_stop epoch={epoch} best_epoch={best_epoch} best_val_loss={best_val_loss}".format(
                        epoch=epoch + 1,
                        best_epoch=self.best_epoch,
                        best_val_loss=round(best_val, 6),
                    ),
                    flush=True,
                )
                break
        if self.n_skipped_batches:
            _log.warning(
                "hrw.train.non_finite_batches_skipped total=%s", self.n_skipped_batches
            )
        if best_state is not None:
            model.load_state_dict(best_state)
        return model

    def _evaluate(
        self,
        model: ModelProtocol,
        batches: Sequence[ForecastBatch],
        device: torch.device,
    ) -> float:
        was_training = model.training  # type: ignore[attr-defined]
        model.eval()  # type: ignore[attr-defined]
        running, total_mass = 0.0, 0.0
        with torch.no_grad():
            for batch in batches:
                batch = batch.to(device)
                output = _model_forward(model, batch)
                loss = self._loss(output, batch)
                batch_mass = _batch_mass(batch)
                running += float(loss.detach()) * batch_mass
                total_mass += batch_mass
        if was_training:
            model.train()  # type: ignore[attr-defined]
        return running / max(total_mass, 1e-8)


def _batch_mass(batch: ForecastBatch) -> float:
    if batch.sample_weight is None:
        return float(batch.features.shape[0])
    return float(batch.sample_weight.detach().sum().clamp_min(0.0).item())


def _epoch_batch_indices(n_batches: int, *, seed: int, epoch: int) -> list[int]:
    if n_batches < 1:
        return []
    if n_batches == 1:
        return [0]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(epoch))
    return torch.randperm(n_batches, generator=generator).tolist()


def _surface_tensors(batch: ForecastBatch) -> SurfaceTensors | None:
    if batch.surface_grid is None or batch.surface_mask is None or batch.surface_context is None:
        return None
    return SurfaceTensors(grid=batch.surface_grid, mask=batch.surface_mask, context=batch.surface_context)


def _model_forward(model: ModelProtocol, batch: ForecastBatch) -> dict[str, Tensor]:
    forward = model.forward  # type: ignore[attr-defined]
    params = inspect.signature(forward).parameters
    kwargs: dict[str, object] = {}
    if "barrier_context" in params:
        kwargs["barrier_context"] = batch.barrier_context
    if "primary_side" in params and batch.primary_side is not None:
        kwargs["primary_side"] = batch.primary_side
    surface = _surface_tensors(batch)
    if surface is not None and "surface" in params:
        kwargs["surface"] = surface
    return forward(  # type: ignore[misc]
        batch.features,
        batch.futures,
        batch.regime_context,
        **kwargs,
    )


__all__ = ["HRWTrainer", "ForecastBatch", "resolve_device", "LossFn"]
