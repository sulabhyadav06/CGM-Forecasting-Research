import numpy as np
import pytest

from experiments.glunet.audit_weighting_failure_modes import _metrics


def test_metrics_reports_zero_error_for_exact_prediction():
    result = _metrics(np.array([80.0, 120.0]), np.array([80.0, 120.0]))
    assert result["MAE"] == pytest.approx(0.0)
    assert result["RMSE"] == pytest.approx(0.0)


def test_metrics_reports_signed_error_direction():
    result = _metrics(np.array([100.0, 100.0]), np.array([110.0, 90.0]))
    assert result["MAE"] == pytest.approx(10.0)
    assert result["mean_error_prediction_minus_target"] == pytest.approx(0.0)
