import numpy as np
import pytest

from experiments.glunet.train_glunet_candidate_all_patients import compute_class_weights


def test_sqrt_inverse_frequency_weights_are_training_only_and_normalized():
    labels = np.array([0] * 4 + [7] * 1 + [255] * 4)
    weights = compute_class_weights(labels, n_classes=256)
    assert weights.shape == (256,)
    assert weights[7] > weights[0]
    assert weights[7] > weights[255]
    assert weights[1] == 0.0
    counts = np.bincount(labels, minlength=256)
    assert np.sum(counts * weights) / counts.sum() == pytest.approx(1.0)


def test_class_weighting_rejects_invalid_labels():
    with pytest.raises(ValueError):
        compute_class_weights(np.array([], dtype=int))
    with pytest.raises(ValueError):
        compute_class_weights(np.array([256]))
