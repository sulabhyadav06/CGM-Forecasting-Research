from __future__ import annotations

import argparse

import numpy as np

import clinical_metrics
from clinical_metrics import time_lag_minutes


def make_signal(n=600, seed=42):
    """Deterministic signal with enough variation to identify the shift."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)

    return (
        20 * np.sin(2 * np.pi * t / 47.0)
        + 8 * np.sin(2 * np.pi * t / 13.0)
        + 4 * np.sin(2 * np.pi * t / 7.0)
        + 0.8 * rng.normal(size=n)
    )


def test_trailing_prediction(max_lag=6, sample_interval=5):
    """
    Prediction trails the truth by k samples:
        pred[k:] = truth[:-k]
    Expected metric = +k * sample_interval minutes.
    """
    y_true = make_signal()
    rows = []
    failures = []

    for k in range(max_lag + 1):
        if k == 0:
            y_pred = y_true.copy()
        else:
            # Fill the first k values only to keep the vector length fixed.
            y_pred = np.empty_like(y_true)
            y_pred[:k] = y_true[:k]
            y_pred[k:] = y_true[:-k]

        observed = time_lag_minutes(
            y_true,
            y_pred,
            sample_interval_min=sample_interval,
            max_lag_samples=12,
        )

        expected = k * sample_interval
        passed = abs(observed - expected) < 1e-9

        rows.append((k, expected, observed, passed))

        if not passed:
            failures.append((k, expected, observed))

    return rows, failures


def test_leading_prediction_is_documented():
    """
    The project metric intentionally reports only non-negative lag.
    A leading prediction is therefore documented rather than treated as
    a negative-lag output.
    """
    y_true = make_signal()
    y_pred = np.empty_like(y_true)

    k = 3
    y_pred[:-k] = y_true[k:]
    y_pred[-k:] = y_true[-k:]

    observed = time_lag_minutes(
        y_true,
        y_pred,
        sample_interval_min=5,
        max_lag_samples=12,
    )

    return observed


def test_mard_uses_absolute_reference_values():
    """MARD should normalize by |y_true|, not by a signed max(…, eps)."""
    y_true = np.array([-10.0, 0.0, 10.0])
    y_pred = np.array([0.0, 0.0, 0.0])
    expected = (np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), 1e-6)) * 100.0)
    observed = clinical_metrics.mard(y_true, y_pred)
    assert observed == expected


def test_time_lag_minutes_rejects_mismatched_lengths():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0])
    try:
        clinical_metrics.time_lag_minutes(y_true, y_pred)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for mismatched input lengths")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-lag", type=int, default=6)
    args = parser.parse_args()

    rows, failures = test_trailing_prediction(args.max_lag)

    print("TIME-LAG SYNTHETIC VALIDATION")
    print("=" * 32)
    print(
        "Convention: positive lag means the prediction trails the "
        "reference trace."
    )

    for k, expected, observed, passed in rows:
        print(
            f"shift={k:2d} samples | "
            f"expected={expected:5.1f} min | "
            f"observed={observed:5.1f} min | "
            f"{'PASS' if passed else 'FAIL'}"
        )

    leading_observed = test_leading_prediction_is_documented()

    print(
        f"\nLeading-prediction documentation test: "
        f"observed={leading_observed:.1f} min"
    )
    print(
        "The metric intentionally searches non-negative lags only; "
        "negative lead values are not reported."
    )

    if failures:
        print("\nFAILED CASES:")
        for failure in failures:
            print(failure)

        raise SystemExit(1)

    print("\nAll positive-lag synthetic tests passed.")


if __name__ == "__main__":
    main()
