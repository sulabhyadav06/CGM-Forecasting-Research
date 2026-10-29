import numpy as np
import pytest

from experiments.glunet.compare_candidate_baselines import metrics


def test_metrics_zero_error_for_exact_forecast():
    y = np.array([90.0, 120.0, 180.0])
    result = metrics(y, y)
    assert result["MAE_mg_dL"] == pytest.approx(0.0)
    assert result["RMSE_mg_dL"] == pytest.approx(0.0)
    assert result["MARD_pct"] == pytest.approx(0.0)


def test_metrics_compute_known_absolute_error():
    actual = np.array([100.0, 100.0])
    forecast = np.array([90.0, 120.0])
    result = metrics(actual, forecast)
    assert result["MAE_mg_dL"] == pytest.approx(15.0)
    assert result["RMSE_mg_dL"] == pytest.approx(np.sqrt(250.0))
    assert result["mean_signed_error_mg_dL"] == pytest.approx(5.0)
