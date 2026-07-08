from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
_SPEC = importlib.util.spec_from_file_location("train_workflow_script", _ROOT / "scripts" / "train_workflow.py")
assert _SPEC is not None and _SPEC.loader is not None
train_workflow_script = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = train_workflow_script
_SPEC.loader.exec_module(train_workflow_script)


def test_default_model_kind_prefers_world_model_for_multi_horizon_strategy() -> None:
    assert (
        train_workflow_script._default_model_kind(
            strategy_horizon=6,
            management_horizon=12,
        )
        == "world_model"
    )
    assert (
        train_workflow_script._default_model_kind(
            strategy_horizon=12,
            management_horizon=12,
        )
        == "forecaster"
    )


def test_normalize_model_artifact_path_switches_default_filename() -> None:
    paths = train_workflow_script.WorkflowPaths.resolve(
        config_path=Path("configs/v1.yaml"),
        data_dir=Path("data"),
        run_dir=Path("runs/train_workflow"),
    )

    normalized = train_workflow_script._normalize_model_artifact_path(
        paths,
        model_kind="world_model",
        explicit=False,
    )

    assert normalized.model_path == Path("runs/train_workflow/world_model.pt")
    assert normalized.labels_path == paths.labels_path
