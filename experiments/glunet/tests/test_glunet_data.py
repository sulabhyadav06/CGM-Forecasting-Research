import numpy as np
import pytest

from experiments.glunet.glunet_data import (
    N_CLASSES,
    quantize_delta,
    recover_delta,
)


def test_zero_delta_maps_to_zero_class_and_recovers_zero():
    label = quantize_delta([0.0])
    assert label.tolist() == [127]
    assert recover_delta(label)[0] == pytest.approx(0.0)


def test_representable_bin_centres_round_trip():
    expected = np.array([-12.7, -5.0, 0.0, 5.0, 12.8])
    labels = quantize_delta(expected)
    recovered = recover_delta(labels)

    assert np.allclose(recovered, expected, atol=1e-6)


def test_labels_stay_within_zero_based_class_range():
    labels = quantize_delta([-100.0, -12.8, 0.0, 12.8, 100.0])

    assert labels.min() == 0
    assert labels.max() == N_CLASSES - 1


def test_clipping_at_extreme_values():
    labels = quantize_delta([-100.0, -12.8, 12.8, 100.0])

    assert labels.tolist() == [0, 0, 255, 255]


def test_half_step_rounding_is_explicit():
    # Current implementation uses half-up rounding.
    labels = quantize_delta([-12.75, -0.05, 0.05, 12.75])

    assert labels.tolist() == [0, 127, 128, 255]


def test_non_finite_delta_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        quantize_delta([np.nan])


def test_invalid_class_indices_are_rejected():
    with pytest.raises(ValueError, match="0–255"):
        recover_delta([-1])

    with pytest.raises(ValueError, match="0–255"):
        recover_delta([256])
