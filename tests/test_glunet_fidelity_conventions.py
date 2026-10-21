import numpy as np
import pandas as pd


def time_of_day_features(timestamp):
    ts = pd.to_datetime(timestamp, errors="raise")
    minutes = ts.hour * 60 + ts.minute + ts.second / 60.0
    angle = 2.0 * np.pi * minutes / 1440.0
    return np.sin(angle), np.cos(angle)


def decode_zero_based_classes(indices):
    idx = np.asarray(indices, dtype=np.int64)
    if np.any((idx < 0) | (idx > 255)):
        raise ValueError("class index must be in [0, 255]")
    return idx.astype(np.float64) / 10.0 - 12.7


def test_class_decoder_endpoints_and_equivalence():
    idx = np.array([0, 1, 127, 255])
    a = decode_zero_based_classes(idx)
    b = (idx + 1) / 10.0 - 12.8
    np.testing.assert_allclose(a, b, atol=1e-12)
    np.testing.assert_allclose(a[[0, -1]], [-12.7, 12.8])


def test_time_of_day_is_cyclic_and_finite():
    a = np.array(time_of_day_features("2026-10-20 23:55:00"))
    b = np.array(time_of_day_features("2026-10-21 00:00:00"))
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert np.linalg.norm(a - b) < 0.03


def test_timestamp_features_repeat_every_day():
    a = time_of_day_features("2026-10-20 08:30:00")
    b = time_of_day_features("2026-10-21 08:30:00")
    np.testing.assert_allclose(a, b, atol=1e-12)
