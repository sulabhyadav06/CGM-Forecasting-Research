"""
Gap-safe sequence utilities for Phase-2 CGM forecasting.

Key behavior:
- timestamps are normalized to UTC-naive datetime64;
- sequences are created only inside continuous 5-minute segments;
- no sequence crosses a missing/irregular timestamp;
- lookback and every prediction horizon must have exact timestamps;
- no interpolation is performed;
- target glucose values are never imputed.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def prepare_timestamps(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    step_min: int = 5,
) -> pd.DataFrame:
    """
    Sort timestamps and identify continuous 5-minute segments.

    A new segment starts whenever the difference between consecutive
    timestamps is not exactly step_min minutes.

    Timestamps are converted to UTC and then made timezone-naive so that
    NumPy datetime64 operations do not produce timezone warnings.
    """
    out = df.copy()

    out[timestamp_col] = pd.to_datetime(
        out[timestamp_col],
        errors="coerce",
        utc=True,
    ).dt.tz_localize(None)

    out = (
        out.dropna(subset=[timestamp_col])
        .sort_values(timestamp_col)
        .reset_index(drop=True)
    )

    if out[timestamp_col].duplicated().any():
        duplicates = int(out[timestamp_col].duplicated().sum())
        raise ValueError(
            f"Found {duplicates} duplicate timestamps."
        )

    delta_min = (
        out[timestamp_col]
        .diff()
        .dt.total_seconds()
        .div(60.0)
    )

    out["_gap_from_previous_min"] = delta_min

    out["_continuous_from_previous"] = (
        delta_min.eq(float(step_min))
    )

    # First row starts a new segment.
    out.loc[
        out.index[0],
        "_continuous_from_previous"
    ] = False

    out["_segment_id"] = (
        ~out["_continuous_from_previous"]
    ).cumsum()

    return out


def build_sequences(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    lookback_steps: int,
    horizon_steps: Sequence[int],
    step_min: int = 5,
    timestamp_col: str = "timestamp",
):
    """
    Build gap-safe supervised sequences.

    Parameters
    ----------
    df:
        DataFrame containing timestamp, features and target.

    feature_cols:
        Input feature columns.

    target_col:
        Target glucose column.

    lookback_steps:
        Number of historical samples used as input.

    horizon_steps:
        Future offsets in samples.

        Example:
            [3, 6, 12, 18, 24]
        corresponds to:
            15, 30, 60, 90, 120 minutes
        at 5-minute sampling.

    step_min:
        Expected sampling interval in minutes.

    Returns
    -------
    X:
        Shape:
        (n_sequences, lookback_steps, n_features)

    y:
        Shape:
        (n_sequences, n_horizons)

    metadata:
        DataFrame containing the timestamp information for each sequence.
    """

    if lookback_steps <= 0:
        raise ValueError(
            "lookback_steps must be > 0"
        )

    if not horizon_steps:
        raise ValueError(
            "horizon_steps cannot be empty"
        )

    horizon_steps = [
        int(h)
        for h in horizon_steps
    ]

    if any(h <= 0 for h in horizon_steps):
        raise ValueError(
            "All horizon steps must be > 0."
        )

    required_columns = (
        [timestamp_col, target_col]
        + list(feature_cols)
    )

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns: "
            + ", ".join(missing)
        )

    data = prepare_timestamps(
        df,
        timestamp_col=timestamp_col,
        step_min=step_min,
    )

    if len(data) == 0:
        raise ValueError(
            "No valid rows remain after timestamp preparation."
        )

    # Convert features and target explicitly to numeric.
    for col in feature_cols:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce",
        )

    data[target_col] = pd.to_numeric(
        data[target_col],
        errors="coerce",
    )

    max_horizon = max(horizon_steps)

    X_sequences = []
    y_sequences = []
    metadata = []

    feature_values = data[
        list(feature_cols)
    ].to_numpy(dtype=np.float32)

    target_values = data[
        target_col
    ].to_numpy(dtype=np.float32)

    timestamps = data[
        timestamp_col
    ].to_numpy(dtype="datetime64[ns]")

    segment_ids = data[
        "_segment_id"
    ].to_numpy()

    n = len(data)

    # The latest target index required by a sequence.
    max_target_offset = max_horizon

    for end_idx in range(
        lookback_steps - 1,
        n - max_target_offset,
    ):
        start_idx = (
            end_idx
            - lookback_steps
            + 1
        )

        # All input rows must belong to the same segment.
        input_segment = segment_ids[
            start_idx:end_idx + 1
        ]

        if not np.all(
            input_segment
            == input_segment[0]
        ):
            continue

        # The target horizon must remain in the same segment.
        target_indices = [
            end_idx + h
            for h in horizon_steps
        ]

        target_segment = segment_ids[
            target_indices
        ]

        if not np.all(
            target_segment
            == input_segment[0]
        ):
            continue

        # Explicit timestamp validation.
        #
        # This protects against any irregularity even if segment IDs
        # were generated differently in a future modification.
        input_times = timestamps[
            start_idx:end_idx + 1
        ]

        expected_input_times = (
            timestamps[end_idx]
            - np.arange(
                lookback_steps - 1,
                -1,
                -1,
                dtype="timedelta64[m]",
            )
            * step_min
        )

        if not np.array_equal(
            input_times,
            expected_input_times,
        ):
            continue

        valid_targets = True
        target_times = []

        for h in horizon_steps:
            target_idx = end_idx + h

            expected_target_time = (
                timestamps[end_idx]
                + np.timedelta64(
                    h * step_min,
                    "m",
                )
            )

            actual_target_time = timestamps[
                target_idx
            ]

            if (
                actual_target_time
                != expected_target_time
            ):
                valid_targets = False
                break

            target_times.append(
                actual_target_time
            )

        if not valid_targets:
            continue

        # Never allow NaNs in model inputs or targets.
        x = feature_values[
            start_idx:end_idx + 1
        ]

        y = target_values[
            target_indices
        ]

        if not np.isfinite(x).all():
            continue

        if not np.isfinite(y).all():
            continue

        X_sequences.append(x)
        y_sequences.append(y)

        metadata.append(
            {
                "input_start": timestamps[
                    start_idx
                ],
                "input_end": timestamps[
                    end_idx
                ],
                "target_times": target_times,
                "segment_id": int(
                    segment_ids[end_idx]
                ),
            }
        )

    if not X_sequences:
        X = np.empty(
            (
                0,
                lookback_steps,
                len(feature_cols),
            ),
            dtype=np.float32,
        )

        y = np.empty(
            (
                0,
                len(horizon_steps),
            ),
            dtype=np.float32,
        )

        metadata_df = pd.DataFrame(
            columns=[
                "input_start",
                "input_end",
                "target_times",
                "segment_id",
            ]
        )

        return X, y, metadata_df

    X = np.stack(
        X_sequences
    ).astype(np.float32)

    y = np.stack(
        y_sequences
    ).astype(np.float32)

    metadata_df = pd.DataFrame(
        metadata
    )

    return X, y, metadata_df


def sequence_count_by_segment(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    lookback_steps: int,
    horizon_steps: Sequence[int],
    step_min: int = 5,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Return the number of valid sequences contributed by each continuous
    timestamp segment.

    Useful for auditing how many sequences are lost because of gaps.
    """

    _, _, metadata = build_sequences(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        lookback_steps=lookback_steps,
        horizon_steps=horizon_steps,
        step_min=step_min,
        timestamp_col=timestamp_col,
    )

    if metadata.empty:
        return pd.DataFrame(
            columns=[
                "segment_id",
                "n_sequences",
            ]
        )

    return (
        metadata
        .groupby(
            "segment_id",
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "n_sequences"
            }
        )
    )