"""GluNet data preparation for OhioT1DM.

This module is separate from the CRNN pipeline. It preserves the
predefined OhioT1DM training/testing partitions and creates 16-step
sequences for GluNet's 256-class glucose-change prediction task.
"""

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
LOOKBACK = 16
STEP_MINUTES = 5
N_CLASSES = 256


def load_split(patient: str, split: str) -> pd.DataFrame:
    """Load an existing predefined OhioT1DM partition."""
    if split not in {"training", "testing"}:
        raise ValueError("split must be 'training' or 'testing'")

    path = DATA_ROOT / f"{patient}_{split}_multimodal.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = ["Timestamp", *FEATURES]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

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
    """Split the timeline at intervals inconsistent with 5-minute sampling."""
    out = df.copy()
    delta = out["Timestamp"].diff().dt.total_seconds()

    continuous = (
        (delta - STEP_MINUTES * 60).abs() <= tolerance_seconds
    )
    continuous.iloc[0] = False

    out["segment_id"] = (~continuous).cumsum()
    return out

def quantize_delta(delta):
    """Map glucose changes to zero-based GluNet class indices.

    Paper labels are 1–256; this implementation returns 0–255
    for compatibility with PyTorch classification targets.

    Convention:
    - Quantization interval: 0.1 mg/dL.
    - Half-up rounding.
    - Values outside the representable range are clipped.
    - Exact boundary behavior follows this implementation's
      rounding-and-clipping convention; it is not claimed to
      resolve ambiguities in the paper's displayed equation.
    """
    values = np.asarray(delta, dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError("Glucose changes must contain only finite values.")

    scaled = 10.0 * (values + 12.8)

    # Explicit half-up rounding.
    paper_labels = np.floor(scaled + 0.5).astype(np.int64)

    # Convert to the paper's valid label range.
    paper_labels = np.clip(paper_labels, 1, N_CLASSES)

    # PyTorch targets must be zero-based: 0–255.
    return paper_labels - 1

def recover_delta(class_indices):
    indices = np.asarray(class_indices, dtype=np.int64)

    if np.any((indices < 0) | (indices >= N_CLASSES)):
        raise ValueError("Class indices must be in the range 0–255.")

    # Convert zero-based indices back to the paper's 1-based labels.
    paper_labels = indices + 1
    return paper_labels.astype(np.float32) / 10.0 - 12.8


def build_sequences(df: pd.DataFrame, horizon_minutes: int = 30):
    """Build gap-safe 16-step input windows and quantized change labels."""
    if horizon_minutes <= 0 or horizon_minutes % STEP_MINUTES:
        raise ValueError("Horizon must be a positive multiple of 5 minutes")

    data = add_segments(df).reset_index(drop=True)

    values = data[FEATURES].to_numpy(dtype=np.float32)
    glucose = data[TARGET].to_numpy(dtype=np.float32)
    timestamps = pd.DatetimeIndex(data["Timestamp"])
    segments = data["segment_id"].to_numpy()

    timestamp_to_index = {ts: i for i, ts in enumerate(timestamps)}
    horizon = pd.Timedelta(minutes=horizon_minutes)

    X, labels, metadata = [], [], []

    for end in range(LOOKBACK - 1, len(data)):
        start = end - LOOKBACK + 1

        if not np.all(segments[start:end + 1] == segments[end]):
            continue

        target_idx = timestamp_to_index.get(timestamps[end] + horizon)
        if target_idx is None or segments[target_idx] != segments[end]:
            continue

        x = values[start:end + 1]
        if not np.isfinite(x).all():
            continue

        current_glucose = glucose[end]
        target_glucose = glucose[target_idx]
        delta = target_glucose - current_glucose

        if not np.isfinite(delta):
            continue

        X.append(x)
        labels.append(quantize_delta([delta])[0])

        metadata.append({
            "input_start": timestamps[start],
            "input_end": timestamps[end],
            "target_timestamp": timestamps[target_idx],
            "current_glucose": float(current_glucose),
            "target_glucose": float(target_glucose),
            "delta_glucose": float(delta),
            "segment_id": int(segments[end]),
        })

    if not X:
        return (
            np.empty((0, LOOKBACK, len(FEATURES)), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            pd.DataFrame(),
        )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        pd.DataFrame(metadata),
    )
