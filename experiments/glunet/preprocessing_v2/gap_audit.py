"""Conservative CGM gap audit for GluNet reproduction.

This module deliberately does not remove outliers or interpolate data.
It identifies regular observations, short internal gaps, and long gaps
so a later pipeline can handle them without silently changing the data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def audit_cgm_gaps(
    df: pd.DataFrame,
    timestamp_col: str = "Timestamp",
    glucose_col: str = "Glucose",
    sample_minutes: int = 5,
    short_gap_max_missing_samples: int = 11,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a cleaned copy and an interval-level gap audit.

    A gap is classified by the number of missing expected samples between
    two observed timestamps. No values are inserted or removed.
    """
    required = {timestamp_col, glucose_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col], errors="coerce")
    out[glucose_col] = pd.to_numeric(out[glucose_col], errors="coerce")
    out = out.dropna(subset=[timestamp_col, glucose_col])
    out = out.sort_values(timestamp_col).reset_index(drop=True)

    if out[timestamp_col].duplicated().any():
        raise ValueError("Duplicate timestamps must be resolved explicitly.")

    delta_min = out[timestamp_col].diff().dt.total_seconds() / 60
    expected_steps = delta_min / sample_minutes
    nearest_steps = expected_steps.round()
    aligned = (
        delta_min.notna()
        & nearest_steps.ge(1)
        & ((expected_steps - nearest_steps).abs() <= 0.1)
    )
    missing_samples = (nearest_steps - 1).where(aligned)

    intervals = pd.DataFrame({
        "previous_timestamp": out[timestamp_col].shift(1),
        "timestamp": out[timestamp_col],
        "previous_glucose": out[glucose_col].shift(1),
        "glucose": out[glucose_col],
        "interval_minutes": delta_min,
        "change_mg_dL": out[glucose_col].diff(),
        "expected_steps": expected_steps,
        "missing_samples": missing_samples,
        "aligned_to_sampling_grid": aligned,
    })

    def classify(row):
        if pd.isna(row["interval_minutes"]):
            return "first_observation"
        if not row["aligned_to_sampling_grid"]:
            return "irregular_interval"
        n = int(row["missing_samples"])
        if n == 0:
            return "regular"
        if n <= short_gap_max_missing_samples:
            return "short_gap_candidate"
        return "long_gap"

    intervals["gap_class"] = intervals.apply(classify, axis=1)
    return out, intervals


def split_at_unresolved_gaps(
    df: pd.DataFrame,
    intervals: pd.DataFrame,
) -> list[pd.DataFrame]:
    """Split at every non-regular interval until gap handling is implemented."""
    if len(df) != len(intervals):
        raise ValueError("df and intervals must have matching row counts.")

    # No interpolation is performed in this version, so even short gaps
    # must break a sequence to prevent windows crossing missing observations.
    breaks = ~intervals["gap_class"].isin(
        ["first_observation", "regular"]
    ).to_numpy()
    starts = [0]
    starts.extend(np.flatnonzero(breaks[1:]) + 1)
    starts = sorted(set(starts))
    ends = starts[1:] + [len(df)]
    return [
        df.iloc[start:end].copy().reset_index(drop=True)
        for start, end in zip(starts, ends)
        if end > start
    ]
