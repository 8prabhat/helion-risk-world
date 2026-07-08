from __future__ import annotations

import torch

from helion_risk_world.memory import (
    CalibrationMonitor,
    DataFreshnessMonitor,
    DriftMonitor,
    RegimeMemory,
)
from helion_risk_world.schemas.prediction_schema import (
    BarrierProbabilities,
    HorizonPrediction,
    ModelPrediction,
)
from helion_risk_world.schemas.execution_schema import ExecutionState


def _prediction(ts, p50: float = 0.02) -> ModelPrediction:
    hp = HorizonPrediction(
        horizon_bars=6,
        return_quantiles={0.1: p50 - 0.02, 0.25: p50 - 0.01, 0.5: p50, 0.75: p50 + 0.01, 0.9: p50 + 0.02},
        volatility=0.01,
    )
    return ModelPrediction(
        symbol="BANKNIFTY",
        ts=ts,
        horizon_preds=[hp],
        barrier=BarrierProbabilities(stop=0.1, target=0.8, timeout=0.1),
        mae=0.01,
        sigma_H=0.01,
        epistemic=0.0,
        aleatoric=0.01,
        ood_score=0.0,
    )


def test_calibration_monitor_decays_confidence_on_bad_coverage() -> None:
    import datetime as dt

    monitor = CalibrationMonitor(coverage_threshold=0.01, decay=0.5, min_scale=0.25, min_samples=4)
    predictions = [_prediction(dt.datetime(2026, 6, 25, 10, i)) for i in range(4)]
    outcomes = [{"realized_return": -0.10} for _ in range(4)]
    report = monitor.check(predictions, outcomes)
    assert report["coverage_error"] > 0.01
    assert report["confidence_scale"] == 0.5


def test_drift_monitor_flags_large_degradation() -> None:
    monitor = DriftMonitor(alert_threshold=0.2)
    report = monitor.check(
        {"net_pnl": 100.0, "sharpe": 2.0, "max_drawdown": 0.05},
        {"net_pnl": 10.0, "sharpe": 0.5, "max_drawdown": 0.15},
    )
    assert report["drift_score"] > 0.2
    assert report["drift_alert"] == 1.0


def test_drift_monitor_flags_distribution_shift() -> None:
    monitor = DriftMonitor(alert_threshold=0.1)
    report = monitor.check_distributions(
        {
            "features": [0.0, 0.1, 0.2, 0.3, 0.4],
            "label": ["timeout", "timeout", "target"],
            "confidence": [0.2, 0.3, 0.4],
            "regime_performance": {"trend": {"mean_step_return": 0.01}},
        },
        {
            "features": [2.0, 2.1, 2.2, 2.3, 2.4],
            "label": ["stop", "stop", "stop"],
            "confidence": [0.8, 0.9, 0.95],
            "regime_performance": {"trend": {"mean_step_return": -0.01}},
        },
    )
    assert report["distribution_drift_score"] > 0.1
    assert report["distribution_drift_alert"] == 1.0


def test_regime_memory_tracks_bounded_counts() -> None:
    memory = RegimeMemory(capacity=2)
    probs = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2]])
    states = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]])
    memory.update(probs, states)
    assert memory.counts()[0] == 2
    assert memory.counts()[1] == 1


def test_data_freshness_monitor_flags_stale_and_skewed_inputs() -> None:
    import datetime as dt

    monitor = DataFreshnessMonitor(
        max_market_staleness_seconds=60.0,
        max_prediction_skew_seconds=30.0,
        require_quotes=True,
        alert_threshold=0.2,
    )
    pred = _prediction(dt.datetime(2026, 6, 25, 10, 0))
    market = ExecutionState(
        symbol="BANKNIFTY",
        ts=dt.datetime(2026, 6, 25, 10, 2),
        available_at=dt.datetime(2026, 6, 25, 10, 0),
        bid=None,
        ask=100.0,
        spread=None,
    )

    step = monitor.observe(pred, market)
    summary = monitor.snapshot()

    assert step["status"] == "alert"
    assert step["stale_market"] == 1.0
    assert step["prediction_skew_alert"] == 1.0
    assert step["quote_alert"] == 1.0
    assert summary["failure_rate"] == 1.0
    assert summary["data_alert"] == 1.0
