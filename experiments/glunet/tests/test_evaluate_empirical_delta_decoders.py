import numpy as np
import pytest

from experiments.glunet.evaluate_empirical_delta_decoders import metric_row


def test_metric_row_zero_error():
    result = metric_row(np.array([90.0, 120.0]), np.array([90.0, 120.0]))
    assert result["MAE_mg_dL"] == pytest.approx(0.0)
    assert result["RMSE_mg_dL"] == pytest.approx(0.0)


def test_metric_row_signed_error_direction():
    result = metric_row(np.array([100.0, 100.0]), np.array([110.0, 90.0]))
    assert result["MAE_mg_dL"] == pytest.approx(10.0)
    assert result["mean_signed_error_mg_dL"] == pytest.approx(0.0)
