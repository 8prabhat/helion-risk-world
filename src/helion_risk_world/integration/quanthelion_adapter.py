"""Adapter layer isolating every Quanthelion-specific assumption.

HelionRiskWorld uses ``quanthelion`` as a *framework* dependency, not a folder location. This module
is the single place where Quanthelion internals are touched, so a framework version bump or a moved
symbol changes exactly one file (Dependency Inversion).

Two categories of symbol are bridged here:

1. **Real symbols** — verified to exist in the installed ``quanthelion`` package. These are imported
   directly and re-exported with stable HRW names.

2. **Assumed-but-absent symbols** — the project brief referenced an "intended style" that does not
   match the real package (e.g. ``quanthelion.training.BaseTrainer``,
   ``quanthelion.data.DatasetProtocol``, ``quanthelion.metrics.MetricRegistry``,
   ``quanthelion.experiments.ExperimentRunner``, ``quanthelion.logging.get_logger``). We do **not**
   invent fake imports. Instead we provide local adapters that reuse the *real* surface.

If ``quanthelion`` is not importable (e.g. docs build, isolated unit test), imports degrade to small
local fallbacks so HRW stays usable; ``QUANTHELION_AVAILABLE`` records which path was taken.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

# --------------------------------------------------------------------------------------
# Real Quanthelion surface (verified against the installed package). See SPEC.md §6.1.
# --------------------------------------------------------------------------------------
try:
    from quanthelion.config import ConfigLoader, ConfigResolver, MaterializedConfig
    from quanthelion.core.errors import QuanthelionError
    from quanthelion.labels.embargo import apply_embargo, make_purged_splits
    from quanthelion.options.black_scholes import (
        bs_price,
        greeks,
        implied_vol,
        time_to_expiry_years,
    )
    from quanthelion.portfolio import PortfolioStateTracker
    from quanthelion.risk import ExecutionGate, GateResult
    from quanthelion.tracking.base import ExperimentTracker
    from quanthelion.uncertainty.conformal import (
        AdaptiveConformalCalibrator,
        ConformalCalibrator,
    )
    from quanthelion.utils.logging import configure_logging as _qh_configure_logging
    from quanthelion.utils.logging import get_logger as _qh_get_logger

    QUANTHELION_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only without the framework installed
    QUANTHELION_AVAILABLE = False

    ConfigLoader = ConfigResolver = MaterializedConfig = object  # type: ignore[assignment,misc]
    ExperimentTracker = object  # type: ignore[assignment,misc]
    ExecutionGate = GateResult = object  # type: ignore[assignment,misc]
    PortfolioStateTracker = object  # type: ignore[assignment,misc]
    ConformalCalibrator = AdaptiveConformalCalibrator = object  # type: ignore[assignment,misc]
    bs_price = greeks = implied_vol = time_to_expiry_years = None  # type: ignore[assignment]
    apply_embargo = make_purged_splits = None  # type: ignore[assignment]

    class QuanthelionError(Exception):  # type: ignore[no-redef]
        """Fallback base error when the framework is unavailable."""

    def _qh_get_logger(name: str) -> Any:  # type: ignore[misc]
        import logging

        return logging.getLogger(name)

    def _qh_configure_logging(level: str = "INFO", json_output: bool = False) -> None:  # type: ignore[misc]
        import logging
        import sys

        logging.basicConfig(
            format="%(levelname)s:%(name)s:%(message)s",
            stream=sys.stderr,
            level=getattr(logging, level.upper(), logging.INFO),
        )


__all__ = [
    "QUANTHELION_AVAILABLE",
    "configure_logging",
    "get_logger",
    "load_config",
    "ConfigLoader",
    "ConfigResolver",
    "MaterializedConfig",
    "ExperimentTracker",
    "ExecutionGate",
    "GateResult",
    "PortfolioStateTracker",
    "ConformalCalibrator",
    "AdaptiveConformalCalibrator",
    "make_purged_splits",
    "apply_embargo",
    "bs_price",
    "greeks",
    "implied_vol",
    "time_to_expiry_years",
    "QuanthelionError",
    # adapters for assumed-but-absent symbols
    "DatasetProtocol",
    "ModelProtocol",
    "LossProtocol",
    "TrainerAdapter",
    "ExperimentRunnerAdapter",
    "HelionIntegrationError",
]


class HelionIntegrationError(QuanthelionError):  # type: ignore[misc]
    """Raised when the Quanthelion integration is misconfigured or a required symbol is missing."""


# --------------------------------------------------------------------------------------
# Logging — the brief assumed ``quanthelion.logging.get_logger``; the real path is
# ``quanthelion.utils.logging.get_logger``. Re-export under the name HRW code uses.
# --------------------------------------------------------------------------------------
def get_logger(name: str) -> Any:
    """Return a structured logger (delegates to ``quanthelion.utils.logging.get_logger``)."""
    return _qh_get_logger(name)


def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    """Configure the underlying logger for HRW command-line entry points."""
    _qh_configure_logging(level=level, json_output=json_output)


# --------------------------------------------------------------------------------------
# Config loading — thin wrapper so HRW callers never import the loader directly.
# --------------------------------------------------------------------------------------
def load_config(path: str) -> dict[str, Any]:
    """Load an HRW YAML config into a plain dict.

    Deliberately a plain ``yaml.safe_load``: HRW configs are *project-local* and are validated by
    HRW's own typed dataclasses in ``helion_risk_world.config`` — NOT by Quanthelion's
    ``ConfigLoader``, which is stage/environment-overlay oriented and expects a different layout
    (see SPEC.md §6.2/§6.3). Routing HRW configs through it would impose the framework's pipeline
    structure on a standalone research project, which we explicitly avoid.

    ``extends:`` is honoured one level deep so backtest/paper configs can layer over ``v1.yaml``.
    """
    import yaml

    with open(path, encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh) or {}

    parent_path = cfg.pop("extends", None)
    if parent_path:
        import os

        base_dir = os.path.dirname(os.path.abspath(path))
        resolved = parent_path if os.path.isabs(parent_path) else os.path.join(
            os.path.dirname(base_dir) if parent_path.startswith("configs/") else base_dir,
            os.path.basename(parent_path),
        )
        # Simplest robust resolution: relative to the config file's directory.
        if not os.path.exists(resolved):
            resolved = os.path.join(base_dir, os.path.basename(parent_path))
        parent = load_config(resolved)
        parent.update(cfg)  # child overrides parent (shallow merge)
        cfg = parent
    return cfg


# --------------------------------------------------------------------------------------
# Adapters for assumed-but-absent symbols (SPEC.md §6.2).
# These are *local* Protocols/classes — never faked imports from quanthelion.
# --------------------------------------------------------------------------------------
@runtime_checkable
class DatasetProtocol(Protocol):
    """Local replacement for the assumed ``quanthelion.data.DatasetProtocol``.

    HRW datasets may optionally subclass ``quanthelion.data.MultiSymbolSupervisedDataset``, but the
    contract the trainer depends on is just this Protocol (Liskov / Interface Segregation).
    """

    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> Any: ...


@runtime_checkable
class ModelProtocol(Protocol):
    """Minimal model contract the trainer depends on (Dependency Inversion)."""

    def forward(self, batch: Any) -> Any: ...

    def parameters(self) -> Any: ...


@runtime_checkable
class LossProtocol(Protocol):
    """Minimal loss contract: maps (prediction, target) -> scalar tensor."""

    def __call__(self, prediction: Any, target: Any) -> Any: ...


class TrainerAdapter:
    """Adapter between the HelionRiskWorld training loop and Quanthelion training utilities.

    The brief assumed ``quanthelion.training.BaseTrainer`` — it does not exist. The real
    ``quanthelion.training`` module exposes reusable *pieces*: ``CheckpointCallback``,
    ``EarlyStopping``, ``MetricsLogger``, ``move_to_device``, ``SAM``, ``FocalLoss`` and samplers.
    This adapter wires those pieces around an HRW-owned training step so the rest of HRW depends on
    a stable interface, not on Quanthelion's concrete classes.

    Shape contracts and the actual optimisation live in ``helion_risk_world.training.trainer``;
    this class only brokers the framework utilities.
    """

    def __init__(
        self,
        model: ModelProtocol,
        loss: LossProtocol,
        *,
        tracker: ExperimentTracker | None = None,
        checkpoint_dir: str | None = None,
        patience: int | None = None,
    ) -> None:
        self.model = model
        self.loss = loss
        self.tracker = tracker
        self._checkpoint_dir = checkpoint_dir
        self._patience = patience
        self._callbacks: list[Any] = []
        self._log = get_logger("hrw.training.adapter")
        if QUANTHELION_AVAILABLE:
            self._wire_quanthelion_callbacks()

    def _wire_quanthelion_callbacks(self) -> None:
        """Attach the real Quanthelion training callbacks when available."""
        from quanthelion.training import CheckpointCallback, EarlyStopping, MetricsLogger

        if self._checkpoint_dir is not None:
            self._callbacks.append(CheckpointCallback(self._checkpoint_dir))
        if self._patience is not None:
            self._callbacks.append(EarlyStopping(patience=self._patience))
        self._callbacks.append(MetricsLogger())

    def to_device(self, obj: Any, device: Any) -> Any:
        """Delegate to ``quanthelion.training.move_to_device`` when available."""
        if QUANTHELION_AVAILABLE:
            from quanthelion.training import move_to_device

            return move_to_device(obj, device)
        return obj

    @property
    def callbacks(self) -> list[Any]:
        return list(self._callbacks)


class ExperimentRunnerAdapter:
    """Local stand-in for the assumed ``quanthelion.experiments.ExperimentRunner``.

    There is no ``quanthelion.experiments`` module. The real framework drives runs through a
    stage-based CLI (``quanthelion run --stage ...``) plus ``quanthelion.tracking`` trackers. This
    adapter gives HRW a single ``run(stage_fn, ...)`` entry point backed by an
    ``ExperimentTracker`` (MLflow or null), so HRW scripts do not hard-code tracking internals.
    """

    def __init__(self, tracker: ExperimentTracker | None = None) -> None:
        self.tracker = tracker
        self._log = get_logger("hrw.experiments.adapter")

    def run(self, run_name: str, stage_fn: Any, params: dict[str, Any] | None = None) -> Any:
        """Execute ``stage_fn`` inside a tracked run; returns whatever ``stage_fn`` returns."""
        if self.tracker is not None:
            self.tracker.start_run(run_name)  # type: ignore[attr-defined]
            if params:
                self.tracker.log_params(params)  # type: ignore[attr-defined]
        try:
            return stage_fn()
        finally:
            if self.tracker is not None:
                self.tracker.end_run()  # type: ignore[attr-defined]
