from __future__ import annotations

import numpy as np
import pytest

from clinical_metrics import time_lag_minutes, mard


def make_signal(n=600, seed=42):
    """Deterministic signal with enough variation to identify a time shift."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    return (
        20 * np.sin(2 * np.pi * t / 47.0)
        + 8 * np.sin(2 * np.pi * t / 13.0)
        + 4 * np.sin(2 * np.pi * t / 7.0)
        + 0.8 * rng.normal(size=n)
    )


@pytest.mark.parametrize("k", range(7))
def test_trailing_prediction(k):
    """A prediction delayed by k samples should have positive lag."""
    truth = make_signal()
    pred = np.empty_like(truth)

    if k == 0:
        pred[:] = truth
    else:
        pred[:k] = truth[:k]
        pred[k:] = truth[:-k]

    observed = time_lag_minutes(
        truth, pred, sample_interval_min=5, max_lag_samples=12
    )
    assert observed == pytest.approx(k * 5, abs=1e-9)


@pytest.mark.parametrize("k", range(1, 7))
def test_leading_prediction_has_negative_lag(k):
    """A prediction shifted earlier should have negative lag."""
    truth = make_signal()
    pred = np.empty_like(truth)
    pred[:-k] = truth[k:]
    pred[-k:] = truth[-k:]

    observed = time_lag_minutes(
        truth, pred, sample_interval_min=5, max_lag_samples=12
    )
    assert observed == pytest.approx(-k * 5, abs=1e-9)


def test_mard_uses_absolute_reference_values():
    truth = np.array([-10.0, 0.0, 10.0])
    pred = np.array([0.0, 0.0, 0.0])
    expected = np.mean(
        np.abs(truth - pred) / np.maximum(np.abs(truth), 1e-6)
    ) * 100.0
    assert mard(truth, pred) == pytest.approx(expected)


def test_time_lag_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same shape"):
        time_lag_minutes(np.array([1., 2., 3.]), np.array([1., 2.]))


def test_time_lag_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="finite"):
        time_lag_minutes(
            np.array([1., np.nan, 3.]), np.array([1., 2., 3.])
        )


def test_time_lag_rejects_invalid_interval():
    with pytest.raises(ValueError, match="positive"):
        time_lag_minutes(
            np.array([1., 2., 3.]),
            np.array([1., 2., 3.]),
            sample_interval_min=0,
        )


def test_empty_signals_return_zero():
    assert time_lag_minutes(np.array([]), np.array([])) == 0.0
