import numpy as np
import pytest

from experiments.glunet.evaluate_weighting_ablation import (
    _event_scores,
    moving_block_ci,
    probability_diagnostics,
)


def test_event_scores_handles_perfect_and_missed_events():
    result = _event_scores(
        np.array([True, True, False, False]),
        np.array([True, False, True, False]),
    )
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["precision_pct"] == pytest.approx(50.0)
    assert result["sensitivity_pct"] == pytest.approx(50.0)


def test_probability_diagnostics_detects_perfect_classification():
    y = np.array([0, 10, 255])
    prob = np.full((3, 256), 1e-6, dtype=float)
    prob[np.arange(3), y] = 1.0
    prob /= prob.sum(axis=1, keepdims=True)
    pred = y.copy()
    result = probability_diagnostics(prob, y, pred, "test")
    assert result["top1_accuracy_pct"] == pytest.approx(100.0)
    assert result["top5_accuracy_pct"] == pytest.approx(100.0)
    assert result["predicted_unique_classes"] == 3
    assert result["endpoint_top1_accuracy_pct"] == pytest.approx(100.0)
    assert result["interior_top1_accuracy_pct"] == pytest.approx(100.0)


def test_moving_block_ci_is_deterministic_and_zero_for_identical_forecasts():
    y = np.array([100., 110., 120., 130., 140., 150.])
    pred = y + np.array([1., -1., 2., -2., 1., -1.])
    result = moving_block_ci(y, pred, pred, 2, 200, 42)
    assert len(result) == 4
    for row in result:
        assert row["median_difference_candidate_minus_base"] == pytest.approx(0.0)
        assert row["ci95_low"] == pytest.approx(0.0)
        assert row["ci95_high"] == pytest.approx(0.0)
