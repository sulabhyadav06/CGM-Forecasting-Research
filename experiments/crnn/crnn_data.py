"""Gap-safe CRNN data preparation for OhioT1DM."""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path.home() / "Desktop" / "OhioT1DM_data_backup"

FEATURES = [
    "Glucose",
    "carbs_last_60min",
    "bolus_last_60min",
]

TARGET = "Glucose"
LOOKBACK = 24
STEP_MINUTES = 5


def load_split(patient: str, split: str) -> pd.DataFrame:
    """Load one predefined training or testing partition."""
    if split not in {"training", "testing"}:
        raise ValueError("split must be 'training' or 'testing'")

    path = DATA_ROOT / f"{patient}_{split}_multimodal.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = ["Timestamp", *FEATURES]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"], errors="coerce"
    )

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["Timestamp", TARGET])
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )

    if df["Timestamp"].duplicated().any():
        raise ValueError(f"Duplicate timestamps in {path}")

    return df


def add_segments(
    df: pd.DataFrame,
    tolerance_seconds: float = 30.0,
) -> pd.DataFrame:
    """
    Mark continuous segments.

    Five-minute intervals within tolerance are accepted as timestamp jitter.
    Larger deviations start a new segment. No resampling or interpolation
    is performed by this function.
    """
    out = df.copy()
    delta_seconds = out["Timestamp"].diff().dt.total_seconds()

    expected_seconds = STEP_MINUTES * 60
    continuous = (delta_seconds - expected_seconds).abs() <= tolerance_seconds
    continuous.iloc[0] = False

    out["segment_id"] = (~continuous).cumsum()
    return out


def build_sequences(
    df: pd.DataFrame,
    horizon_minutes: int = 30,
):
    """
    Build gap-safe sequences with exact timestamp-aligned targets.

    Input:
        24 samples, 3 features.

    Target:
        Glucose change at exactly horizon_minutes after input_end.

    Rows with no exact target timestamp are skipped. No interpolation
    or nearest-timestamp matching is performed.
    """
    if horizon_minutes <= 0 or horizon_minutes % STEP_MINUTES:
        raise ValueError("Horizon must be a positive multiple of 5 minutes")

    data = add_segments(df).reset_index(drop=True)

    X, delta_y, metadata = [], [], []

    values = data[FEATURES].to_numpy(dtype=np.float32)
    glucose = data[TARGET].to_numpy(dtype=np.float32)
    timestamps = pd.DatetimeIndex(data["Timestamp"])
    segments = data["segment_id"].to_numpy()

    # Map each timestamp to its row index.
    timestamp_to_index = {
        ts: i for i, ts in enumerate(timestamps)
    }

    horizon = pd.Timedelta(minutes=horizon_minutes)

    for end in range(LOOKBACK - 1, len(data)):
        start = end - LOOKBACK + 1

        # All 24 input rows must be in the same continuous segment.
        if not np.all(segments[start:end + 1] == segments[end]):
            continue

        # Require the exact target timestamp.
        target_time = timestamps[end] + horizon
        target_idx = timestamp_to_index.get(target_time)

        if target_idx is None:
            continue

        # Target must be in the same segment as the input.
        if segments[target_idx] != segments[end]:
            continue

        x = values[start:end + 1]

        if not np.isfinite(x).all():
            continue

        current_glucose = glucose[end]
        target_glucose = glucose[target_idx]

        if not np.isfinite(current_glucose) or not np.isfinite(target_glucose):
            continue

        X.append(x)
        delta_y.append(target_glucose - current_glucose)

        metadata.append({
            "input_start": timestamps[start],
            "input_end": timestamps[end],
            "target_timestamp": timestamps[target_idx],
            "current_glucose": float(current_glucose),
            "target_glucose": float(target_glucose),
            "segment_id": int(segments[end]),
        })

    if not X:
        return (
            np.empty((0, LOOKBACK, len(FEATURES)), dtype=np.float32),
            np.empty((0, 1), dtype=np.float32),
            pd.DataFrame(),
        )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(delta_y, dtype=np.float32).reshape(-1, 1),
        pd.DataFrame(metadata),
    )