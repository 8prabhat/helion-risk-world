from __future__ import annotations

import pytest

from helion_risk_world.evaluation.ml_metrics import classification_report


def test_classification_report_includes_required_class_metrics() -> None:
    report = classification_report(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7],
        ],
        [0, 1, 2, 2],
        class_names=("stop", "target", "timeout"),
    )

    assert report["samples"] == 4
    assert report["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 1, 1]]
    assert report["per_class"]["stop"]["precision"] == pytest.approx(1.0)
    assert report["per_class"]["target"]["recall"] == pytest.approx(1.0)
    assert report["per_class"]["timeout"]["recall"] == pytest.approx(0.5)
    assert report["confidence_buckets"]
