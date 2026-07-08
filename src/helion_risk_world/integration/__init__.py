"""Quanthelion integration / adapter layer (isolates all framework assumptions)."""

from helion_risk_world.integration.quanthelion_adapter import (
    QUANTHELION_AVAILABLE,
    DatasetProtocol,
    ExperimentRunnerAdapter,
    HelionIntegrationError,
    LossProtocol,
    ModelProtocol,
    TrainerAdapter,
    configure_logging,
    get_logger,
    load_config,
)

__all__ = [
    "QUANTHELION_AVAILABLE",
    "DatasetProtocol",
    "ExperimentRunnerAdapter",
    "HelionIntegrationError",
    "LossProtocol",
    "ModelProtocol",
    "TrainerAdapter",
    "configure_logging",
    "get_logger",
    "load_config",
]
