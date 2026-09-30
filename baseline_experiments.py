"""Matched Phase-2 baseline comparison for Ohio 2018/2020 CGM forecasting.

Runs GRU, LSTM and TCN with the SAME:
- Phase-2 multimodal features
- train-only median imputation and standardization
- train-only target standardization
- continuous 5-minute sequence construction
- chronological validation split
- conservative purge gap
- train/test patient split
- optimizer, epochs, batch size and early stopping

Example:
python baseline_experiments.py \
  --data-root data/phase2/ohio2018 \
  --patient 559 \
  --lookbacks 60 \
  --horizons 15 30 60 90 120 \
  --epochs 15

For a controlled comparison with the Transformer pilot, defaults are:
hidden=64, layers=2, dropout=0.10, lr=1e-3, weight_decay=1e-4.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


COHORT_FEATURES = {
    "ohio2018": [
        "glucose",
        "carbs_last_60min",
        "bolus_last_60min",
        "basal_rate",
        "heart_rate_mean_5m",
        "heart_rate_std_5m",
        "heart_rate_min_5m",
        "heart_rate_max_5m",
        "heart_rate_count_5m",
        "heart_rate_coverage_5m",
        "heart_rate_missing_5m",
        "steps_mean_5m",
        "steps_std_5m",
        "steps_min_5m",
        "steps_max_5m",
        "steps_count_5m",
        "steps_coverage_5m",
        "steps_missing_5m",
    ],
    "ohio2020": [
        "glucose",
        "carbs_last_60min",
        "bolus_last_60min",
        "basal_rate",
        "accel_mean_5m",
        "accel_std_5m",
        "accel_min_5m",
        "accel_max_5m",
        "accel_count_5m",
        "accel_coverage_5m",
        "accel_missing_5m",
    ],
}


class Standardizer:
    def __init__(self):
        self.median_: dict[str, float] = {}
        self.mean_: dict[str, float] = {}
        self.std_: dict[str, float] = {}

    def fit(self, frames: Iterable[pd.DataFrame], columns: list[str]):
        joined = pd.concat([f[columns] for f in frames], ignore_index=True)
        for col in columns:
            s = pd.to_numeric(joined[col], errors="coerce")
            med = float(s.median()) if s.notna().any() else 0.0
            filled = s.fillna(med)
            mean = float(filled.mean())
            std = float(filled.std(ddof=0))
            if not np.isfinite(std) or std < 1e-8:
                std = 1.0
            self.median_[col] = med
            self.mean_[col] = mean
            self.std_[col] = std
        return self

    def transform(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        out = df.copy()
        for col in columns:
            s = pd.to_numeric(out[col], errors="coerce").fillna(self.median_[col])
            out[col] = ((s - self.mean_[col]) / self.std_[col]).astype(np.float32)
        return out


class SequenceDataset(Dataset):
    """Continuous sequence dataset. Data must already be imputed/scaled."""

    def __init__(self, df, feature_columns, lookback_minutes, horizon_minutes, sampling_minutes=5):
        self.feature_columns = list(feature_columns)
        self.lookback_steps = lookback_minutes // sampling_minutes
        self.horizon_steps = horizon_minutes // sampling_minutes

        work = df.copy()
        work["timestamp"] = pd.to_datetime(work["timestamp"])
        work = work.sort_values("timestamp").reset_index(drop=True)

        X = work[self.feature_columns].to_numpy(dtype=np.float32)
        y = work["glucose"].to_numpy(dtype=np.float32)
        ts = work["timestamp"].to_numpy(dtype="datetime64[ns]")

        L = self.lookback_steps
        H = self.horizon_steps
        expected = np.timedelta64(sampling_minutes, "m")

        xs, ys = [], []
        for end_idx in range(L - 1, len(work) - H):
            start = end_idx - L + 1
            target_idx = end_idx + H

            input_ts = ts[start:end_idx + 1]
            future_ts = ts[end_idx:target_idx + 1]

            if len(input_ts) != L or len(future_ts) != H + 1:
                continue
            if not np.all(np.diff(input_ts) == expected):
                continue
            if not np.all(np.diff(future_ts) == expected):
                continue

            window = X[start:end_idx + 1]
            target = y[target_idx]

            if not np.isfinite(window).all() or not np.isfinite(target):
                continue

            xs.append(window)
            ys.append(target)

        self.X = np.stack(xs).astype(np.float32) if xs else np.empty((0, L, len(self.feature_columns)), np.float32)
        self.y = np.asarray(ys, dtype=np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.float32)

    def subset(self, indices):
        return SubsetDataset(self, np.asarray(indices, dtype=np.int64))


class SubsetDataset(Dataset):
    def __init__(self, parent, indices):
        self.parent = parent
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.parent[int(self.indices[idx])]


class ConcatDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.offsets = []
        total = 0
        for ds in datasets:
            self.offsets.append(total)
            total += len(ds)
        self.total = total

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        for start, ds in reversed(list(zip(self.offsets, self.datasets))):
            if idx >= start:
                return ds[idx - start]
        raise IndexError(idx)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_cohort(path: Path):
    name = path.name.lower()
    if "2018" in name:
        return "ohio2018"
    if "2020" in name:
        return "ohio2020"
    raise ValueError("data-root must contain ohio2018 or ohio2020")


def load_frames(data_root, patient):
    def load(folder):
        files = sorted(folder.glob("*.csv"))
        if patient:
            files = [p for p in files if p.stem == str(patient)]
        if not files:
            raise FileNotFoundError(f"No CSV files in {folder} for patient={patient}")
        out = []
        for p in files:
            out.append((p.stem, pd.read_csv(p, parse_dates=["timestamp"])))
        return out

    return load(data_root / "train"), load(data_root / "test")


def build_sequences(frames, feature_columns, lookback, horizon, sampling, scaler, target_mean, target_std):
    result = []
    for patient, df in frames:
        work = scaler.transform(df, feature_columns)
        raw_glucose = pd.to_numeric(df["glucose"], errors="coerce")
        work["glucose"] = ((raw_glucose - target_mean) / target_std).astype(np.float32)
        ds = SequenceDataset(work, feature_columns, lookback, horizon, sampling)
        result.append((patient, ds))
    return result


def temporal_split(ds, val_fraction, purge):
    n = len(ds)
    if n < 20:
        raise ValueError(f"Only {n} sequences available.")
    cut = int(n * (1 - val_fraction))
    train_end = max(1, cut - purge)
    val_start = min(n - 1, cut)
    return ds.subset(np.arange(train_end)), ds.subset(np.arange(val_start, n))


def split_multi(patient_datasets, val_fraction, purge):
    tr_parts, va_parts = [], []
    for patient, ds in patient_datasets:
        tr, va = temporal_split(ds, val_fraction, purge)
        print(f"  {patient}: total={len(ds):,}, train={len(tr):,}, val={len(va):,}, purge={purge}")
        tr_parts.append(tr)
        va_parts.append(va)
    return ConcatDataset(tr_parts), ConcatDataset(va_parts)


# --------------------------- Models ---------------------------

class MLPHead(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden=64, layers=2, dropout=0.1):
        super().__init__()
        self.rnn = nn.GRU(
            input_dim, hidden, num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0
        )
        self.norm = nn.LayerNorm(hidden)
        self.head = MLPHead(hidden, dropout)

    def forward(self, x):
        z, _ = self.rnn(x)
        return self.head(self.norm(z[:, -1]))


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden=64, layers=2, dropout=0.1):
        super().__init__()
        self.rnn = nn.LSTM(
            input_dim, hidden, num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0
        )
        self.norm = nn.LayerNorm(hidden)
        self.head = MLPHead(hidden, dropout)

    def forward(self, x):
        z, _ = self.rnn(x)
        return self.head(self.norm(z[:, -1]))


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.norm2 = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    @staticmethod
    def chomp(x, trim):
        return x[:, :, :-trim] if trim > 0 else x

    def forward(self, x):
        y = self.conv1(x)
        trim = self.conv1.padding[0]
        y = self.chomp(y, trim)
        y = self.drop(self.act(self.norm1(y)))

        y = self.conv2(y)
        trim = self.conv2.padding[0]
        y = self.chomp(y, trim)
        y = self.drop(self.act(self.norm2(y)))

        return self.act(y + self.down(x))


class TCNModel(nn.Module):
    def __init__(self, input_dim, hidden=64, levels=2, kernel_size=3, dropout=0.1):
        super().__init__()
        blocks = []
        channels = [input_dim] + [hidden] * levels
        for i in range(levels):
            blocks.append(
                TemporalBlock(
                    channels[i], channels[i + 1],
                    kernel_size=kernel_size,
                    dilation=2 ** i,
                    dropout=dropout,
                )
            )
        self.tcn = nn.Sequential(*blocks)
        self.norm = nn.LayerNorm(hidden)
        self.head = MLPHead(hidden, dropout)

    def forward(self, x):
        z = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        return self.head(self.norm(z[:, -1]))


def build_model(model_name, input_dim, hidden, layers, dropout):
    """Build all baselines through one common interface."""
    if model_name == "GRU":
        return GRUModel(
            input_dim=input_dim,
            hidden=hidden,
            layers=layers,
            dropout=dropout,
        )
    if model_name == "LSTM":
        return LSTMModel(
            input_dim=input_dim,
            hidden=hidden,
            layers=layers,
            dropout=dropout,
        )
    if model_name == "TCN":
        # TCN uses `levels` rather than recurrent `layers`.
        return TCNModel(
            input_dim=input_dim,
            hidden=hidden,
            levels=layers,
            dropout=dropout,
        )
    raise ValueError(f"Unknown model: {model_name}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_epoch(model, loader, optimizer, device, train, clip):
    model.train(train)
    loss_fn = nn.MSELoss()
    total, n = 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        pred = model(x).squeeze(-1)
        loss = loss_fn(pred, y)

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        total += float(loss.item()) * len(y)
        n += len(y)

    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, target_mean, target_std):
    model.eval()
    preds, ys = [], []

    for x, y in loader:
        pred = model(x.to(device)).squeeze(-1).cpu().numpy()
        preds.append(pred)
        ys.append(y.numpy())

    p = np.concatenate(preds) * target_std + target_mean
    y = np.concatenate(ys) * target_std + target_mean
    err = p - y

    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
    }


def train_one(model, train_loader, val_loader, test_loader, args, device, target_mean, target_std):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val = float("inf")
    best_state = None
    bad = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss = run_epoch(model, train_loader, optimizer, device, True, args.grad_clip)
        va_loss = run_epoch(model, val_loader, optimizer, device, False, args.grad_clip)
        va_metrics = evaluate(model, val_loader, device, target_mean, target_std)

        print(
            f"    Epoch {epoch:02d} | train_mse={tr_loss:.5f} | "
            f"val_mse={va_loss:.5f} | val_MAE={va_metrics['mae']:.3f} | "
            f"val_RMSE={va_metrics['rmse']:.3f}"
        )

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print("    Early stopping.")
                break

    model.load_state_dict(best_state)
    return evaluate(model, test_loader, device, target_mean, target_std)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--patient", type=str, default=None,
                    help="One patient. Omit to run every patient in train/*.csv.")
    ap.add_argument("--lookbacks", type=int, nargs="+", default=[60])
    ap.add_argument("--horizons", type=int, nargs="+", default=[15, 30, 60, 90, 120])
    ap.add_argument("--sampling", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--models", nargs="+", choices=["GRU", "LSTM", "TCN"],
                    default=["GRU", "LSTM", "TCN"])
    ap.add_argument("--output-dir", type=Path, default=Path("output/phase2_baselines"))
    ap.add_argument("--force", action="store_true",
                    help="Rerun combinations already present in the result CSV.")
    args = ap.parse_args()

    set_seed(args.seed)

    data_root = args.data_root.resolve()
    cohort = infer_cohort(data_root)
    features = COHORT_FEATURES[cohort]

    # Discover patients from train CSVs when --patient is omitted.
    train_dir = data_root / "train"
    test_dir = data_root / "test"
    all_train_files = sorted(train_dir.glob("*.csv"))
    all_test_files = sorted(test_dir.glob("*.csv"))

    if args.patient:
        patients = [str(args.patient)]
    else:
        train_patients = {p.stem for p in all_train_files}
        test_patients = {p.stem for p in all_test_files}
        patients = sorted(train_patients & test_patients)

    if not patients:
        raise FileNotFoundError(f"No matched train/test patients under {data_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / f"{cohort}_all_patients_baseline_results.csv"

    if result_path.exists() and not args.force:
        existing = pd.read_csv(result_path)
    else:
        existing = pd.DataFrame()

    print(f"Cohort: {cohort}")
    print(f"Patients: {patients}")
    print(f"Input features ({len(features)}): {features}")
    print(f"Resumable result file: {result_path}")

    for patient in patients:
        print("\n" + "#" * 78)
        print(f"PATIENT {patient}")
        print("#" * 78)

        train_frames, test_frames = load_frames(data_root, patient)

        missing = [c for c in features if c not in train_frames[0][1].columns]
        if missing:
            raise ValueError(f"Missing required features for patient {patient}: {missing}")

        scaler = Standardizer().fit([df for _, df in train_frames], features)
        glucose = pd.concat(
            [df["glucose"] for _, df in train_frames], ignore_index=True
        ).astype(float)
        target_mean = float(glucose.mean())
        target_std = float(glucose.std(ddof=0))
        if not np.isfinite(target_std) or target_std < 1e-8:
            target_std = 1.0

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        for lookback in args.lookbacks:
            for horizon in args.horizons:
                for model_name in args.models:
                    # Resume logic: skip an exact completed combination.
                    if not args.force and not existing.empty:
                        mask = (
                            existing["cohort"].astype(str).eq(cohort)
                            & existing["patient"].astype(str).eq(str(patient))
                            & existing["model"].astype(str).eq(model_name)
                            & existing["lookback_min"].eq(lookback)
                            & existing["horizon_min"].eq(horizon)
                        )
                        if mask.any():
                            print(
                                f"\nSKIP existing: patient={patient}, "
                                f"model={model_name}, lookback={lookback}, horizon={horizon}"
                            )
                            continue

                    print("\n" + "=" * 78)
                    print(
                        f"PATIENT={patient} | LOOKBACK={lookback} min | "
                        f"HORIZON={horizon} min | MODEL={model_name}"
                    )
                    print("=" * 78)

                    train_seq = build_sequences(
                        train_frames, features, lookback, horizon, args.sampling,
                        scaler, target_mean, target_std
                    )
                    test_seq = build_sequences(
                        test_frames, features, lookback, horizon, args.sampling,
                        scaler, target_mean, target_std
                    )

                    purge = lookback // args.sampling + horizon // args.sampling
                    train_ds, val_ds = split_multi(
                        train_seq, args.val_fraction, purge
                    )
                    test_ds = ConcatDataset([ds for _, ds in test_seq])

                    print(
                        f"Train/val/test sequences: "
                        f"{len(train_ds):,} / {len(val_ds):,} / {len(test_ds):,}"
                    )

                    train_loader = DataLoader(
                        train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers
                    )
                    val_loader = DataLoader(
                        val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers
                    )
                    test_loader = DataLoader(
                        test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers
                    )

                    set_seed(args.seed)
                    model = build_model(
                        model_name=model_name,
                        input_dim=len(features),
                        hidden=args.hidden,
                        layers=args.layers,
                        dropout=args.dropout,
                    ).to(device)

                    params = count_parameters(model)
                    print(f"Trainable parameters: {params:,}")

                    metrics = train_one(
                        model, train_loader, val_loader, test_loader,
                        args, device, target_mean, target_std
                    )

                    print(
                        f"TEST | MAE={metrics['mae']:.3f} mg/dL | "
                        f"RMSE={metrics['rmse']:.3f} mg/dL"
                    )

                    row = {
                        "cohort": cohort,
                        "patient": str(patient),
                        "model": model_name,
                        "lookback_min": lookback,
                        "horizon_min": horizon,
                        "parameters": params,
                        "train_sequences": len(train_ds),
                        "val_sequences": len(val_ds),
                        "test_sequences": len(test_ds),
                        "test_mae_mgdl": metrics["mae"],
                        "test_rmse_mgdl": metrics["rmse"],
                        "hidden": args.hidden,
                        "layers": args.layers,
                        "dropout": args.dropout,
                        "batch_size": args.batch_size,
                        "epochs_max": args.epochs,
                        "patience": args.patience,
                        "seed": args.seed,
                    }

                    # Append immediately so an interruption loses at most one run.
                    existing = pd.concat(
                        [existing, pd.DataFrame([row])], ignore_index=True
                    )
                    existing.to_csv(result_path, index=False)

    # Patient-level aggregate: mean ± SD across patients.
    if not existing.empty:
        numeric_cols = ["test_mae_mgdl", "test_rmse_mgdl"]
        group_cols = ["cohort", "model", "lookback_min", "horizon_min"]

        grouped = existing.groupby(group_cols, dropna=False)[numeric_cols].agg(
            ["mean", "std", "count"]
        ).reset_index()

        grouped.columns = [
            "_".join([str(x) for x in col if str(x) != ""]).rstrip("_")
            if isinstance(col, tuple) else str(col)
            for col in grouped.columns
        ]

        summary_path = args.output_dir / f"{cohort}_patient_mean_sd.csv"
        grouped.to_csv(summary_path, index=False)

        mae_table = existing.pivot_table(
            index=["patient", "model"],
            columns="horizon_min",
            values="test_mae_mgdl",
            aggfunc="first",
        )
        mae_table.to_csv(args.output_dir / f"{cohort}_patient_mae_table.csv")

        rmse_table = existing.pivot_table(
            index=["patient", "model"],
            columns="horizon_min",
            values="test_rmse_mgdl",
            aggfunc="first",
        )
        rmse_table.to_csv(args.output_dir / f"{cohort}_patient_rmse_table.csv")

        print("\n" + "=" * 78)
        print("FINAL PATIENT-LEVEL MEAN ± SD")
        print("=" * 78)
        print(grouped.to_string(index=False))
        print(f"\nSaved raw results: {result_path}")
        print(f"Saved mean/SD: {summary_path}")


if __name__ == "__main__":
    main()
