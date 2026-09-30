"""Leakage-safe sequence dataset for Phase-2 multimodal CGM forecasting.

Key design choices:
- Missing physiological measurements are imputed BEFORE sequence creation.
- Missingness indicators remain explicit model features.
- A sequence is created only when the 5-min timeline is continuous.
- Targets are taken at a future horizon from the final input timestamp.
- No random train/validation split is performed here; the training script uses
  a chronological split with a purge gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SequenceConfig:
    lookback_minutes: int = 60
    horizon_minutes: int = 30
    sampling_minutes: int = 5

    @property
    def lookback_steps(self) -> int:
        return self.lookback_minutes // self.sampling_minutes

    @property
    def horizon_steps(self) -> int:
        return self.horizon_minutes // self.sampling_minutes


class CGMSequenceDataset(Dataset):
    """Build continuous multimodal sequences from one patient's dataframe.

    The dataframe must already have missing values imputed/scaled as desired.
    ``timestamp`` must be a datetime column and ``target_column`` is the future
    glucose target after the caller has applied the same target transformation
    used for training.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: Sequence[str],
        config: SequenceConfig,
        target_column: str = "glucose",
    ) -> None:
        self.config = config
        self.feature_columns = list(feature_columns)
        self.target_column = target_column

        required = {"timestamp", target_column, *self.feature_columns}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Missing dataframe columns: {sorted(missing)}")

        work = df.copy()
        work["timestamp"] = pd.to_datetime(work["timestamp"])
        work = work.sort_values("timestamp").reset_index(drop=True)

        X = work[self.feature_columns].to_numpy(dtype=np.float32)
        y = work[target_column].to_numpy(dtype=np.float32)
        ts = work["timestamp"].to_numpy(dtype="datetime64[ns]")

        L = config.lookback_steps
        H = config.horizon_steps
        expected = np.timedelta64(config.sampling_minutes, "m")

        sequences: list[np.ndarray] = []
        targets: list[float] = []
        end_times: list[np.datetime64] = []
        target_times: list[np.datetime64] = []

        for end_idx in range(L - 1, len(work) - H):
            start_idx = end_idx - L + 1
            target_idx = end_idx + H

            # Require the complete input window and target horizon to lie on the
            # expected 5-min grid. This prevents sequences from crossing gaps.
            input_ts = ts[start_idx : end_idx + 1]
            future_ts = ts[end_idx : target_idx + 1]
            if len(input_ts) != L or len(future_ts) != H + 1:
                continue
            if not np.all(np.diff(input_ts) == expected):
                continue
            if not np.all(np.diff(future_ts) == expected):
                continue

            window = X[start_idx : end_idx + 1]
            target = y[target_idx]
            if not np.isfinite(window).all() or not np.isfinite(target):
                # This should normally be zero after train-only imputation.
                continue

            sequences.append(window)
            targets.append(float(target))
            end_times.append(ts[end_idx])
            target_times.append(ts[target_idx])

        if sequences:
            self.X = np.stack(sequences).astype(np.float32)
            self.y = np.asarray(targets, dtype=np.float32)
            self.end_times = np.asarray(end_times, dtype="datetime64[ns]")
            self.target_times = np.asarray(target_times, dtype="datetime64[ns]")
        else:
            self.X = np.empty((0, L, len(self.feature_columns)), dtype=np.float32)
            self.y = np.empty((0,), dtype=np.float32)
            self.end_times = np.empty((0,), dtype="datetime64[ns]")
            self.target_times = np.empty((0,), dtype="datetime64[ns]")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.float32)

    def subset(self, indices: Sequence[int]) -> "CGMSequenceSubset":
        return CGMSequenceSubset(self, np.asarray(indices, dtype=np.int64))


class CGMSequenceSubset(Dataset):
    def __init__(self, parent: CGMSequenceDataset, indices: np.ndarray):
        self.parent = parent
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.parent[int(self.indices[idx])]
