import numpy as np
import pytest

from experiments.glunet.audit_glunet_targets import (
    class_distribution,
    glucose_metrics,
    js_divergence,
)


def test_class_distribution_reports_endpoints_and_entropy():
    result = class_distribution(np.array([0, 0, 2, 255]))
    assert result["n"] == 4
    assert result["unique_classes"] == 3
    assert result["endpoint_class_share_pct"] == pytest.approx(75.0)
    assert result["lower_endpoint_share_pct"] == pytest.approx(50.0)
    assert result["upper_endpoint_share_pct"] == pytest.approx(25.0)
    assert result["entropy_nats"] > 0


def test_js_divergence_is_zero_for_same_distribution():
    counts = np.array([1, 3, 0, 2])
    assert js_divergence(counts, counts) == pytest.approx(0.0)


def test_js_divergence_is_symmetric_and_nonnegative():
    p = np.array([10, 0, 0])
    q = np.array([0, 0, 10])
    assert js_divergence(p, q) == pytest.approx(js_divergence(q, p))
    assert js_divergence(p, q) > 0


def test_glucose_metrics_exact_forecast_has_zero_error():
    values = np.array([90.0, 120.0, 180.0])
    metrics = glucose_metrics(values, values)
    assert metrics["MAE_mg_dL"] == pytest.approx(0.0)
    assert metrics["RMSE_mg_dL"] == pytest.approx(0.0)
    assert metrics["MARD_pct"] == pytest.approx(0.0)
