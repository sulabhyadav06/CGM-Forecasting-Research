import pandas as pd
import pytest

from experiments.glunet.preprocessing_v2.gap_audit import (
    audit_cgm_gaps,
    split_at_unresolved_gaps,
)


def make_df(minutes, glucose=None):
    if glucose is None:
        glucose = [100 + i for i in range(len(minutes))]
    return pd.DataFrame({
        "Timestamp": [
            pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=m)
            for m in minutes
        ],
        "Glucose": glucose,
    })


def test_regular_five_minute_intervals():
    df = make_df([0, 5, 10, 15])
    cleaned, audit = audit_cgm_gaps(df)
    assert len(cleaned) == 4
    assert audit["gap_class"].tolist() == [
        "first_observation", "regular", "regular", "regular"
    ]


def test_short_gap_is_flagged_but_not_interpolated():
    df = make_df([0, 5, 20, 25])
    cleaned, audit = audit_cgm_gaps(df)
    assert len(cleaned) == 4
    assert audit.loc[2, "missing_samples"] == 2
    assert audit.loc[2, "gap_class"] == "short_gap_candidate"


def test_long_gap_is_flagged_and_splits_segments():
    df = make_df([0, 5, 10, 100, 105])
    cleaned, audit = audit_cgm_gaps(df)
    segments = split_at_unresolved_gaps(cleaned, audit)
    assert audit.loc[3, "gap_class"] == "long_gap"
    assert [len(segment) for segment in segments] == [3, 2]


def test_duplicate_timestamps_raise():
    df = make_df([0, 5, 5, 10])
    with pytest.raises(ValueError, match="Duplicate timestamps"):
        audit_cgm_gaps(df)


def test_irregular_interval_splits_segments():
    df = make_df([0, 5, 13, 18])
    cleaned, audit = audit_cgm_gaps(df)
    segments = split_at_unresolved_gaps(cleaned, audit)
    assert audit.loc[2, "gap_class"] == "irregular_interval"
    assert [len(segment) for segment in segments] == [2, 2]


def test_short_gap_also_splits_when_not_interpolated():
    df = make_df([0, 5, 20, 25])
    cleaned, audit = audit_cgm_gaps(df)
    segments = split_at_unresolved_gaps(cleaned, audit)
    assert [len(segment) for segment in segments] == [2, 2]
