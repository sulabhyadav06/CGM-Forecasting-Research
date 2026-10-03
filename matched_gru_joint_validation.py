"""
Gap-safe, capacity-matched GRU joint validation.

This preserves the original matched-GRU architecture/configuration:
- 6 Ohio 2018 patients: 559, 563, 570, 575, 588, 591
- 120-min lookback
- 15/30/60/90/120-min horizons
- 85/15 chronological split
- patient-specific train-only feature scaling
- 2-layer GRU, hidden=152
- patient embedding=16
- horizon embedding=32
- fusion 128 -> 64
- five horizon-specific heads
- AdamW, lr=3e-4, weight_decay=1e-4
- batch=128
- SmoothL1 loss
- gradient clipping=1.0
- early stopping patience=6
- seed=42

The only methodological changes relative to the old runner are:
1. gap-safe sequence construction using phase2_sequence_utils;
2. train-only median imputation for missing input features before scaling;
3. explicit float assignment to avoid pandas dtype warnings.

Glucose targets are never imputed.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from phase2_sequence_utils import build_sequences


FEATURES = [
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
]

PATIENTS = [559, 563, 570, 575, 588, 591]
HORIZONS = [15, 30, 60, 90, 120]
STEP = 5


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
    )

    df = (
        df.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    for col in FEATURES:
        if col not in df.columns:
            raise ValueError(f"{path}: missing feature {col!r}")

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    # Do NOT drop feature-NaN rows here.
    # We retain timestamps so the sequence builder can preserve gaps.
    if df["glucose"].isna().any():
        raise ValueError(
            f"{path}: glucose contains missing values; "
            "target values must never be imputed."
        )

    return df


def chronological_split(
    df: pd.DataFrame,
    fraction: float = 0.85,
):
    split = int(len(df) * fraction)

    if split <= 0 or split >= len(df):
        raise ValueError("Invalid chronological split.")

    return (
        df.iloc[:split].copy(),
        df.iloc[split:].copy(),
    )


def train_only_scale(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
):
    """
    Median-impute input features using training rows only, then standardize
    using training mean/std only.

    This preserves the original runner's mean/std scaling protocol while
    making missing wearable values explicit and leakage-safe.
    """
    train = train_df.copy()
    val = val_df.copy()

    train_values = train[FEATURES].to_numpy(dtype=np.float64)
    val_values = val[FEATURES].to_numpy(dtype=np.float64)

    # Medians are computed only from the chronological training partition.
    medians = np.nanmedian(train_values, axis=0)

    if not np.isfinite(medians).all():
        bad = [
            FEATURES[i]
            for i, x in enumerate(medians)
            if not np.isfinite(x)
        ]
        raise ValueError(
            "Training feature has no finite median: "
            + ", ".join(bad)
        )

    train_missing = int(
        (~np.isfinite(train_values)).any(axis=1).sum()
    )
    val_missing = int(
        (~np.isfinite(val_values)).any(axis=1).sum()
    )

    train_values = np.where(
        np.isfinite(train_values),
        train_values,
        medians,
    )

    val_values = np.where(
        np.isfinite(val_values),
        val_values,
        medians,
    )

    # Preserve original mean/std scaling convention.
    means = train_values.mean(axis=0)
    stds = train_values.std(axis=0, ddof=1)

    stds = np.where(
        np.isfinite(stds) & (stds > 0),
        stds,
        1.0,
    )

    train_scaled = (train_values - means) / stds
    val_scaled = (val_values - means) / stds

    train = train.copy()
    val = val.copy()

    train[FEATURES] = pd.DataFrame(
    train_scaled,
    columns=FEATURES,
    index=train.index,
).astype(np.float64)

    val[FEATURES] = pd.DataFrame(
    val_scaled,
    columns=FEATURES,
    index=val.index,
).astype(np.float64)

    scaler = {
        "mean": dict(zip(FEATURES, means.tolist())),
        "std": dict(zip(FEATURES, stds.tolist())),
        "median": dict(zip(FEATURES, medians.tolist())),
    }

    return (
        train,
        val,
        scaler,
        train_missing,
        val_missing,
    )


class DS(Dataset):
    def __init__(self, parts):
        if not parts:
            raise ValueError("No sequence parts supplied.")

        self.x = torch.from_numpy(
            np.concatenate([p[0] for p in parts], axis=0)
        )
        self.y = torch.from_numpy(
            np.concatenate([p[1] for p in parts], axis=0)
        )
        self.p = torch.from_numpy(
            np.concatenate(
                [
                    np.full(
                        len(p[0]),
                        p[2],
                        dtype=np.int64,
                    )
                    for p in parts
                ],
                axis=0,
            )
        )

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return (
            self.x[index],
            self.p[index],
            self.y[index],
        )


class MatchedGRU(nn.Module):
    """
    Exact capacity-controlled architecture from the previous matched-GRU
    experiment.
    """

    def __init__(
        self,
        input_dim: int,
        hidden: int = 152,
        layers: int = 2,
        n_h: int = 5,
        drop: float = 0.10,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_dim,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=drop if layers > 1 else 0.0,
        )

        self.pemb = nn.Embedding(6, 16)
        self.hemb = nn.Embedding(n_h, 32)

        self.fuse = nn.Sequential(
            nn.Linear(hidden + 16, 128),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(128, 64),
            nn.GELU(),
        )

        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(96, 32),
                    nn.GELU(),
                    nn.Linear(32, 1),
                )
                for _ in range(n_h)
            ]
        )

    def forward(self, x, patient_index):
        z, _ = self.gru(x)
        z = z[:, -1]

        z = self.fuse(
            torch.cat(
                [
                    z,
                    self.pemb(patient_index),
                ],
                dim=-1,
            )
        )

        return torch.stack(
            [
                head(
                    torch.cat(
                        [
                            z,
                            self.hemb.weight[i].expand(
                                len(x),
                                -1,
                            ),
                        ],
                        dim=-1,
                    )
                ).squeeze(-1)
                for i, head in enumerate(self.heads)
            ],
            dim=1,
        )


def nparams(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def inverse_glucose(values, scaler):
    mu = scaler["mean"]["glucose"]
    sd = scaler["std"]["glucose"]
    return values * sd + mu


def evaluate_patient(
    model,
    X,
    y,
    patient_index,
    scaler,
    batch_size,
    device,
):
    loader = DataLoader(
        DS([(X, y, patient_index)]),
        batch_size=batch_size,
        shuffle=False,
    )

    preds = []
    trues = []

    model.eval()

    with torch.no_grad():
        for x, p, target in loader:
            pred = model(
                x.to(device),
                p.to(device),
            ).cpu().numpy()

            preds.append(pred)
            trues.append(target.numpy())

    pred = np.concatenate(preds, axis=0)
    true = np.concatenate(trues, axis=0)

    pred = inverse_glucose(pred, scaler)
    true = inverse_glucose(true, scaler)

    error = pred - true

    mae = np.mean(np.abs(error), axis=0)
    rmse = np.sqrt(np.mean(error ** 2, axis=0))

    return mae, rmse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        default="data/phase2/ohio2018",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    args = parser.parse_args()

    if args.lookback % STEP != 0:
        raise ValueError(
            "--lookback must be divisible by 5."
        )

    seed_all(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    root = Path(args.data_root)

    train_parts = []
    val_parts = []
    scalers = {}
    counts = {}
    imputation = {}

    for patient_index, patient in enumerate(PATIENTS):
        df = read_csv(
            root / "train" / f"{patient}.csv"
        )

        train_df, val_df = chronological_split(df)

        (
            train_scaled,
            val_scaled,
            scaler,
            train_missing,
            val_missing,
        ) = train_only_scale(
            train_df,
            val_df,
        )

        X_train, y_train, _ = build_sequences(
            train_scaled,
            FEATURES,
            "glucose",
            args.lookback // STEP,
            [h // STEP for h in HORIZONS],
            step_min=STEP,
        )

        X_val, y_val, _ = build_sequences(
            val_scaled,
            FEATURES,
            "glucose",
            args.lookback // STEP,
            [h // STEP for h in HORIZONS],
            step_min=STEP,
        )

        if len(X_train) == 0 or len(X_val) == 0:
            raise RuntimeError(
                f"Patient {patient}: no valid sequences."
            )

        train_parts.append(
            (X_train, y_train, patient_index)
        )

        val_parts.append(
            (X_val, y_val, patient_index)
        )

        scalers[patient] = scaler

        counts[patient] = {
            "train": len(X_train),
            "val": len(X_val),
        }

        imputation[patient] = {
            "train_rows_with_feature_imputation": train_missing,
            "val_rows_with_feature_imputation": val_missing,
        }

    loader = DataLoader(
        DS(train_parts),
        batch_size=args.batch_size,
        shuffle=True,
    )

    model = MatchedGRU(
        len(FEATURES),
        hidden=152,
        layers=2,
        n_h=5,
        drop=0.10,
    ).to(device)

    parameters = nparams(model)

    print(f"Device: {device}")
    print(f"Lookback: {args.lookback} min")
    print(f"Horizons: {HORIZONS}")
    print(f"Trainable parameters: {parameters}")
    print(f"Patient sequence counts: {counts}")
    print(f"Feature-imputation rows: {imputation}")

    # Sanity check: this should reproduce the established ~264K budget.
    if not (250_000 <= parameters <= 275_000):
        raise RuntimeError(
            f"Unexpected parameter count: {parameters}. "
            "Expected approximately 264K."
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_score = float("inf")
    best_state = None
    best_epoch = 0
    bad = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        n = 0

        for x, p, y in loader:
            x = x.to(device)
            p = p.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            prediction = model(x, p)

            loss = F.smooth_l1_loss(
                prediction,
                y,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            total_loss += loss.item() * len(x)
            n += len(x)

        patient_mae = []
        patient_rmse = []

        for patient_index, patient in enumerate(PATIENTS):
            X_val, y_val, _ = val_parts[patient_index]

            mae, rmse = evaluate_patient(
                model,
                X_val,
                y_val,
                patient_index,
                scalers[patient],
                args.batch_size,
                device,
            )

            patient_mae.append(mae)
            patient_rmse.append(rmse)

        patient_mae = np.stack(patient_mae)
        patient_rmse = np.stack(patient_rmse)

        mean_mae = patient_mae.mean(axis=0)
        mean_rmse = patient_rmse.mean(axis=0)

        score = float(mean_mae.mean())

        print(
            f"epoch={epoch:02d} "
            f"train={total_loss / n:.5f} "
            f"val_mean_mae={score:.4f} "
            f"MAE={np.round(mean_mae, 3)} "
            f"RMSE={np.round(mean_rmse, 3)}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / n,
                "val_mean_mae": score,
                **{
                    f"mae_{h}": float(v)
                    for h, v in zip(HORIZONS, mean_mae)
                },
                **{
                    f"rmse_{h}": float(v)
                    for h, v in zip(HORIZONS, mean_rmse)
                },
            }
        )

        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            bad = 0
        else:
            bad += 1

            if bad >= args.patience:
                print(
                    f"Early stopping at epoch {epoch}; "
                    f"best epoch={best_epoch}"
                )
                break

    if best_state is None:
        raise RuntimeError(
            "No best state was saved."
        )

    model.load_state_dict(best_state)

    output = Path(
        "output/matched_gru_gap_safe_exact"
    )
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    patient_rows = []

    for patient_index, patient in enumerate(PATIENTS):
        X_val, y_val, _ = val_parts[patient_index]

        mae, rmse = evaluate_patient(
            model,
            X_val,
            y_val,
            patient_index,
            scalers[patient],
            args.batch_size,
            device,
        )

        for horizon, m, r in zip(
            HORIZONS,
            mae,
            rmse,
        ):
            patient_rows.append(
                {
                    "patient": patient,
                    "model": "matched_gru_gap_safe_exact",
                    "lookback_min": args.lookback,
                    "horizon_min": horizon,
                    "mae_mgdl": float(m),
                    "rmse_mgdl": float(r),
                    "parameters": parameters,
                    "best_epoch": best_epoch,
                    "n_val_sequences": counts[patient]["val"],
                }
            )

    patient_results = pd.DataFrame(
        patient_rows
    )

    patient_results.to_csv(
        output / "patient_results.csv",
        index=False,
    )

    summary = (
        patient_results
        .groupby("horizon_min")
        .agg(
            mae_mean=("mae_mgdl", "mean"),
            mae_sd=("mae_mgdl", "std"),
            rmse_mean=("rmse_mgdl", "mean"),
            rmse_sd=("rmse_mgdl", "std"),
            n_patients=("patient", "nunique"),
        )
        .reset_index()
    )

    summary.to_csv(
        output / "summary.csv",
        index=False,
    )

    pd.DataFrame(history).to_csv(
        output / "training_history.csv",
        index=False,
    )

    checkpoint = (
        output / "matched_gru_gap_safe_exact_best.pt"
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "patients": PATIENTS,
            "lookback": args.lookback,
            "horizons": HORIZONS,
            "features": FEATURES,
            "parameters": parameters,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "best_validation_mean_mae": best_score,
            "scalers": scalers,
            "sequence_counts": counts,
            "imputation": imputation,
        },
        checkpoint,
    )

    metadata = {
        "parameters": parameters,
        "best_epoch": best_epoch,
        "best_validation_mean_mae": best_score,
        "patients": PATIENTS,
        "horizons": HORIZONS,
        "lookback_min": args.lookback,
        "sequence_counts": counts,
        "imputation": imputation,
        "checkpoint": str(checkpoint),
    }

    (output / "summary.json").write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("FINAL GAP-SAFE CAPACITY-MATCHED GRU RESULTS")
    print(summary.to_string(index=False))
    print()
    print(f"Best epoch: {best_epoch}")
    print(
        f"Best validation mean MAE: "
        f"{best_score:.4f}"
    )
    print(
        f"Trainable parameters: {parameters}"
    )
    print(f"Saved: {checkpoint}")


if __name__ == "__main__":
    main()
