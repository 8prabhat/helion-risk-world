"""PositionSizer: epistemic/OOD/meta-label uncertainty gating (2026-07-18)."""

from __future__ import annotations

from datetime import datetime

import pytest

from helion_risk_world.planner.position_sizer import PositionSizer
from helion_risk_world.schemas.prediction_schema import (
    BarrierProbabilities,
    HorizonPrediction,
    ModelPrediction,
)

TS = datetime(2026, 7, 18, 10, 0)


def _pred(
    *, epistemic: float = 0.0, ood_score: float = 0.0,
    primary_side: int = 0, meta_label_prob: float | None = None,
) -> ModelPrediction:
    hp = HorizonPrediction(
        horizon_bars=12,
        return_quantiles={0.1: -0.02, 0.25: -0.01, 0.5: 0.0, 0.75: 0.01, 0.9: 0.02},
        volatility=0.01,
    )
    return ModelPrediction(
        symbol="X", ts=TS, horizon_preds=[hp],
        barrier=BarrierProbabilities(stop=0.3, target=0.3, timeout=0.4),
        mae=0.01, sigma_H=0.01,
        epistemic=epistemic, aleatoric=0.01, ood_score=ood_score,
        primary_side=primary_side, meta_label_prob=meta_label_prob,
    )


def test_no_uncertainty_no_shrinkage() -> None:
    sizer = PositionSizer()
    assert sizer.adjust(1.0, _pred()) == pytest.approx(1.0)


def test_epistemic_shrinks_size() -> None:
    sizer = PositionSizer()
    assert sizer.adjust(1.0, _pred(epistemic=0.4)) == pytest.approx(0.6)


def test_ood_shrinks_size() -> None:
    sizer = PositionSizer()
    assert sizer.adjust(1.0, _pred(ood_score=0.5)) == pytest.approx(0.5)


def test_meta_label_gate_inert_when_primary_side_flat() -> None:
    """No bet proposed -> meta_label_prob is meaningless -> gate must be neutral,
    even if meta_label_prob happens to be populated (e.g. a stale value)."""
    sizer = PositionSizer()
    assert sizer.adjust(1.0, _pred(primary_side=0, meta_label_prob=0.1)) == pytest.approx(1.0)


def test_meta_label_gate_inert_when_prob_missing() -> None:
    """Older artifact with no meta-label head -> meta_label_prob is None -> neutral gate."""
    sizer = PositionSizer()
    assert sizer.adjust(1.0, _pred(primary_side=1, meta_label_prob=None)) == pytest.approx(1.0)


def test_meta_label_gate_scales_size_by_probability() -> None:
    sizer = PositionSizer()
    assert sizer.adjust(1.0, _pred(primary_side=1, meta_label_prob=0.3)) == pytest.approx(0.3)
    assert sizer.adjust(1.0, _pred(primary_side=-1, meta_label_prob=0.9)) == pytest.approx(0.9)


def test_meta_label_scale_zero_disables_gate() -> None:
    sizer = PositionSizer(meta_label_scale=0.0)
    assert sizer.adjust(1.0, _pred(primary_side=1, meta_label_prob=0.1)) == pytest.approx(1.0)


def test_gates_compose_multiplicatively() -> None:
    sizer = PositionSizer()
    size = sizer.adjust(
        1.0, _pred(epistemic=0.5, ood_score=0.2, primary_side=1, meta_label_prob=0.5)
    )
    assert size == pytest.approx(0.5 * 0.8 * 0.5)


def test_confidence_scale_multiplies_in_too() -> None:
    sizer = PositionSizer()
    assert sizer.adjust(1.0, _pred(), confidence_scale=0.4) == pytest.approx(0.4)


def test_result_never_exceeds_base_size() -> None:
    sizer = PositionSizer()
    assert sizer.adjust(2.0, _pred()) <= 2.0 + 1e-9
