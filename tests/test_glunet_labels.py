import numpy as np

from experiments.glunet.glunet_data import (
    quantize_delta,
    recover_delta,
)


def test_output_class_range():
    deltas = np.array([
        -1000.0, -12.8, -12.7, 0.0,
        12.7, 12.8, 1000.0
    ])

    labels = quantize_delta(deltas)

    assert labels.min() >= 0
    assert labels.max() <= 255
    assert labels[0] == 0
    assert labels[-1] == 255


def test_known_class_representatives():
    classes = np.array([0, 1, 127, 128, 254, 255])

    recovered = recover_delta(classes)

    expected = np.array([
        -12.7, -12.6, 0.0, 0.1, 12.7, 12.8
    ])

    np.testing.assert_allclose(
        recovered, expected, atol=1e-6
    )


def test_zero_change_maps_to_zero_class_index():
    label = quantize_delta([0.0])[0]

    assert label == 127


def test_extreme_changes_are_clipped():
    labels = quantize_delta([-1000.0, 1000.0])

    np.testing.assert_array_equal(labels, [0, 255])


def test_invalid_class_indices_are_rejected():
    for invalid in [-1, 256]:
        try:
            recover_delta([invalid])
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"Expected ValueError for class {invalid}"
            )


def test_quantization_is_finite_for_valid_inputs():
    deltas = np.linspace(-12.8, 12.8, 1000)

    labels = quantize_delta(deltas)
    recovered = recover_delta(labels)

    assert np.isfinite(labels).all()
    assert np.isfinite(recovered).all()

def test_non_finite_glucose_changes_are_rejected():
    for value in [np.nan, np.inf, -np.inf]:
        try:
            quantize_delta([value])
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"Expected ValueError for glucose change {value}"
            )


def test_boundary_values_follow_documented_convention():
    deltas = np.array([
        -1000.0,
        -12.8,
        0.0,
        12.8,
        1000.0,
    ])

    labels = quantize_delta(deltas)

    np.testing.assert_array_equal(
        labels,
        np.array([0, 0, 127, 255, 255]),
    )