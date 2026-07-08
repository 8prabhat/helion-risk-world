from __future__ import annotations

from datetime import datetime, timedelta
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from helion_risk_world.config.data_config import DataConfig  # noqa: E402
from helion_risk_world.config.model_config import ModelConfig  # noqa: E402
from helion_risk_world.data.market_window_builder import CANDLE_FEATURE_NAMES  # noqa: E402
from helion_risk_world.data.model_input_builder import ModelInputContract  # noqa: E402
from helion_risk_world.model import HRWWorldModel  # noqa: E402
from helion_risk_world.schemas.execution_schema import ExecutionState  # noqa: E402
from helion_risk_world.schemas.prediction_schema import (  # noqa: E402
    BarrierProbabilities,
    HorizonPrediction,
    ModelPrediction,
)
from helion_risk_world.training.artifacts import save_world_model_artifact  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
_SPEC = importlib.util.spec_from_file_location(
    "backtest_runtime_script",
    _ROOT / "scripts" / "_backtest_runtime.py",
)
assert _SPEC is not None and _SPEC.loader is not None
backtest_runtime_script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(backtest_runtime_script)


def test_load_runtime_for_horizon_accepts_world_model_supported_horizon(tmp_path) -> None:
    n_features = len(CANDLE_FEATURE_NAMES)
    path = tmp_path / "world_model.pt"
    contract = ModelInputContract.from_data_config(
        DataConfig(universe=("BANKNIFTY",), lookback_bars=12),
        feature_names=CANDLE_FEATURE_NAMES,
        barrier_horizon_bars=192,
    )
    save_world_model_artifact(
        path,
        HRWWorldModel(
            n_features=n_features,
            cfg=ModelConfig(latent_dim=16, temporal_layers=1, dropout=0.0, rollout_samples=4),
            horizons=(3, 6, 12, 192),
            n_samples=4,
        ),
        n_features=n_features,
        horizons=(3, 6, 12, 192),
        cfg=ModelConfig(latent_dim=16, temporal_layers=1, dropout=0.0, rollout_samples=4),
        input_contract=contract,
    )

    runtime = backtest_runtime_script._load_runtime_for_horizon(path, 192)
    assert runtime.available_horizons == (3, 6, 12, 192)
    assert runtime.predictor._persist_state is True

    with pytest.raises(ValueError, match="do not include requested strategy horizon 9"):
        backtest_runtime_script._load_runtime_for_horizon(path, 9)

    with pytest.raises(ValueError, match="strategy horizon must match artifact management"):
        backtest_runtime_script._load_runtime_for_horizon(path, 6)

    reset_runtime = backtest_runtime_script._load_runtime_for_horizon(path, 192, persist_state=False)
    assert reset_runtime.predictor._persist_state is False


def test_filter_steps_to_runtime_split_uses_artifact_test_boundary() -> None:
    ts0 = datetime(2026, 1, 1, 9, 15)

    def _step(i: int):
        ts = ts0 + timedelta(minutes=5 * i)
        pred = ModelPrediction(
            symbol="BANKNIFTY",
            ts=ts,
            horizon_preds=[
                HorizonPrediction(
                    horizon_bars=6,
                    return_quantiles={0.1: -0.01, 0.25: -0.005, 0.5: 0.0, 0.75: 0.005, 0.9: 0.01},
                    volatility=0.01,
                )
            ],
            barrier=BarrierProbabilities(stop=0.2, target=0.3, timeout=0.5),
            mae=0.01,
            sigma_H=0.01,
            epistemic=0.0,
            aleatoric=0.01,
            ood_score=0.0,
        )
        market = ExecutionState(
            symbol="BANKNIFTY",
            ts=ts,
            available_at=ts,
            bid=100.0,
            ask=100.1,
            spread=0.1,
        )
        return backtest_runtime_script.BacktestStep(
            prediction=pred,
            market=market,
            execution_market=market,
            realized_return=0.0,
        )

    runtime = SimpleNamespace(
        split_manifest={
            "train_end": (ts0 + timedelta(minutes=10)).isoformat(),
            "val_end": (ts0 + timedelta(minutes=20)).isoformat(),
            "total_rows": 6,
            "train_rows": 3,
            "val_rows": 1,
            "test_rows": 1,
            "train_fraction": 0.5,
            "val_fraction": 0.25,
            "val_start": (ts0 + timedelta(minutes=15)).isoformat(),
            "test_start": (ts0 + timedelta(minutes=25)).isoformat(),
            "embargo_bars": 1,
        }
    )

    kept = backtest_runtime_script._filter_steps_to_runtime_split(
        [_step(i) for i in range(6)],
        runtime,
        split="test",
    )

    assert [step.prediction.ts for step in kept] == [ts0 + timedelta(minutes=25)]


def test_build_steps_from_source_uses_next_open_execution_semantics(monkeypatch) -> None:
    timestamps = [datetime(2026, 1, 1, 9, 15) + timedelta(minutes=5 * i) for i in range(3)]

    class _FeatureBuilder:
        def __init__(self, dc, source) -> None:
            self._dc = dc

        def build_window(self, ts):
            return SimpleNamespace(
                candle_features=np.zeros((1, max(self._dc.lookback_bars, 6), len(CANDLE_FEATURE_NAMES)))
            )

    monkeypatch.setattr(backtest_runtime_script, "FeatureBuilder", _FeatureBuilder)
    monkeypatch.setattr(
        backtest_runtime_script,
        "_aligned_trade_grid",
        lambda dc, source, data_dir=None: (
            "BANKNIFTY_FUT_continuous",
            timestamps,
            np.array([100.0, 105.0, 106.0], dtype=float),
            np.array([101.0, 110.0, 111.0], dtype=float),
        ),
    )

    steps = backtest_runtime_script._build_steps_from_source(
        DataConfig(universe=("BANKNIFTY",), lookback_bars=1),
        source=object(),
        horizon=1,
    )

    assert steps[0].execution_market.bid == pytest.approx(105.0 - 0.1, abs=1e-6)
    assert steps[0].execution_market.ask == pytest.approx(105.0 + 0.1, abs=1e-6)
    assert steps[0].execution_market.spread == pytest.approx(0.2, abs=1e-6)
    assert steps[0].realized_return == pytest.approx((110.0 / 105.0) - 1.0, abs=1e-6)
    # Review finding H8: label_realized_at must be populated so LeakageReport's
    # label-future check can actually run instead of being silently skipped.
    assert steps[0].label_realized_at == timestamps[0 + 1]
    assert steps[0].label_realized_at > steps[0].prediction.ts
