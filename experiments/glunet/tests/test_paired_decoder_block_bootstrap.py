import numpy as np
import pytest

from experiments.glunet.paired_decoder_block_bootstrap import (
    moving_block_indices,
    paired_block_bootstrap,
)


def test_moving_block_indices_has_requested_length_and_valid_range():
    rng = np.random.default_rng(7)
    indices = moving_block_indices(n=101, block_size=12, rng=rng)
    assert len(indices) == 101
    assert indices.min() >= 0
    assert indices.max() < 101


def test_identical_forecasts_have_zero_paired_differences():
    actual = np.array([90.0, 100.0, 110.0, 120.0] * 10)
    forecast = actual + np.array([1.0, -2.0, 3.0, -4.0] * 10)
    result = paired_block_bootstrap(
        actual, forecast, forecast, block_size=4,
        n_bootstrap=200, seed=11,
    )
    for metric in ("MAE", "RMSE", "MARD"):
        assert result[f"{metric}_difference"] == pytest.approx(0.0)
        assert result[f"{metric}_ci95_low"] == pytest.approx(0.0)
        assert result[f"{metric}_ci95_high"] == pytest.approx(0.0)


def test_candidate_improvement_has_negative_mae_difference():
    actual = np.array([100.0, 110.0, 120.0, 130.0] * 10)
    reference = actual + 10.0
    candidate = actual + 2.0
    result = paired_block_bootstrap(
        actual, candidate, reference, block_size=4,
        n_bootstrap=200, seed=19,
    )
    assert result["MAE_difference"] == pytest.approx(-8.0)
    assert result["MAE_ci95_high"] < 0.0
