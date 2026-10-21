"""Non-destructive synthetic sanity checks for the current GluNet data helpers.

These tests check implementation invariants, not fidelity to the GluNet paper.
Run from the repository root after copying this file into tests/.
"""

import numpy as np
import pandas as pd

from experiments.glunet.glunet_data import (
    FEATURES,
    LOOKBACK,
    add_segments,
    build_sequences,
    quantize_delta,
    recover_delta,
)


def test_quantize_recover_representable_class_centres():
    centres = np.arange(256, dtype=np.float64) / 10.0 - 12.7
    indices = quantize_delta(centres)
    assert np.array_equal(indices, np.arange(256))
    recovered = recover_delta(indices)
    assert np.allclose(recovered, centres, atol=1e-6)


def test_quantizer_rejects_non_finite_values():
    try:
        quantize_delta([0.0, np.nan])
    except ValueError as exc:
        assert "finite" in str(exc).lower()
    else:
        raise AssertionError("Expected non-finite values to be rejected")


def _synthetic_frame(n=60):
    timestamps = pd.date_range("2025-01-01", periods=n, freq="5min")
    frame = pd.DataFrame({
        "Timestamp": timestamps,
        "Glucose": 110.0 + np.arange(n) * 0.25,
        "carbs_last_60min": np.zeros(n),
        "bolus_last_60min": np.zeros(n),
    })
    return frame


def test_sequences_use_exact_horizon_timestamp_and_expected_shape():
    frame = _synthetic_frame()
    X, y, meta = build_sequences(frame, horizon_minutes=30)
    assert len(X) == len(y) == len(meta) > 0
    assert X.shape[1:] == (LOOKBACK, len(FEATURES))
    assert np.all(
        (pd.to_datetime(meta["target_timestamp"]) -
         pd.to_datetime(meta["input_end"])) == pd.Timedelta(minutes=30)
    )
    assert np.all(meta["target_glucose"].to_numpy() > meta["current_glucose"].to_numpy())


def test_gap_segmentation_splits_irregular_sampling():
    frame = _synthetic_frame(8)
    frame.loc[4:, "Timestamp"] += pd.Timedelta(minutes=15)
    segmented = add_segments(frame)
    ids = segmented["segment_id"].to_numpy()
    assert np.all(ids[:4] == ids[0])
    assert np.all(ids[4:] == ids[4])
    assert ids[4] != ids[3]
