from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from hybrid_models import HybridTCNGRUTransformer, count_parameters
from phase2_sequence_utils import build_sequences


PATIENTS_2018 = [559, 563, 570, 575, 588, 591]
PATIENTS_2020 = [540, 544, 552, 567, 584, 596]
HORIZONS = [15, 30, 60, 90, 120]
STEP_MIN = 5


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SequenceDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/phase2")
    p.add_argument("--lookback", type=int, default=120)
    p.add_argument("--d-model", type=int, default=56)
    p.add_argument("--gru-hidden", type=int, default=56)
    p.add_argument("--gru-layers", type=int, default=2)
    p.add_argument("--tcn-levels", type=int, default=3)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--transformer-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="output/phase2_hybrid_all12")
    return p.parse_args()


def normalize_time(df):
    df = df.copy()
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        .dt.tz_localize(None)
    )
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def load_patient_data(root, cohort, patient):
    train = pd.read_csv(
        Path(root) / cohort / "train" / f"{patient}.csv"
    )
    test = pd.read_csv(
        Path(root) / cohort / "test" / f"{patient}.csv"
    )
    return normalize_time(train), normalize_time(test)


def get_feature_columns(df):
    excluded = {
        "timestamp",
        "gap_from_previous_min",
        "valid_5min_interval",
        "carbs_observed_60min",
        "bolus_observed_60min",
    }
    cols = [c for c in df.columns if c not in excluded]
    cols.remove("glucose")
    return ["glucose"] + cols


def fit_imputer(df, columns):
    medians = {}
    for c in columns:
        values = pd.to_numeric(df[c], errors="coerce")
        median = values.median()
        if not np.isfinite(median):
            raise ValueError(f"Invalid training median for {c}")
        medians[c] = float(median)
    return medians


def apply_imputer(df, medians):
    out = df.copy()
    for c, median in medians.items():
        out[c] = (
            pd.to_numeric(out[c], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(median)
            .astype(np.float64)
        )
    return out


def fit_scaler(df, columns):
    scaler = {}
    for c in columns:
        values = df[c].to_numpy(dtype=np.float64)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        if not np.isfinite(mean):
            raise ValueError(f"Invalid training mean for {c}")
        if not np.isfinite(std) or std < 1e-8:
            std = 1.0
        scaler[c] = {"mean": mean, "std": std}
    return scaler


def apply_scaler(df, scaler):
    out = df.copy()
    for c, s in scaler.items():
        out[c] = (
            (out[c].to_numpy(dtype=np.float64) - s["mean"])
            / s["std"]
        )
    return out


def split_train_validation(df, fraction):
    n_val = max(int(len(df) * fraction), 1)
    if n_val >= len(df):
        raise ValueError("Validation split leaves no training rows.")
    return df.iloc[:-n_val].copy(), df.iloc[-n_val:].copy()


def make_sequences(df, features, lookback_min):
    X, y, metadata = build_sequences(
        df=df,
        feature_cols=features,
        target_col="glucose",
        lookback_steps=lookback_min // STEP_MIN,
        horizon_steps=[h // STEP_MIN for h in HORIZONS],
        step_min=STEP_MIN,
        timestamp_col="timestamp",
    )
    return X, y, metadata


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    lr,
    weight_decay,
    patience,
    grad_clip,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.SmoothL1Loss()

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            pred = model(x)

            if not torch.isfinite(pred).all():
                raise ValueError("Model produced NaN/Inf predictions.")

            loss = criterion(pred, y)

            if not torch.isfinite(loss):
                raise ValueError("Training loss became NaN/Inf.")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip
            )
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                if not torch.isfinite(loss):
                    raise ValueError("Validation loss became NaN/Inf.")
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        print(
            f"    Epoch {epoch:02d} | "
            f"Train {train_loss:.5f} | Val {val_loss:.5f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is None:
        raise RuntimeError("No best model state was saved.")

    model.load_state_dict(best_state)
    return model, history, best_epoch, best_val


def evaluate(model, loader, device, scaler):
    model.eval()
    preds = []
    targets = []
    gates = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred, gate = model(
                x,
                return_gates=True,
            )
            preds.append(pred.cpu().numpy())
            targets.append(y.numpy())
            gates.append(gate.cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    gates = np.concatenate(gates)

    glucose_mean = scaler["glucose"]["mean"]
    glucose_std = scaler["glucose"]["std"]

    preds = preds * glucose_std + glucose_mean
    targets = targets * glucose_std + glucose_mean

    rows = []
    for i, horizon in enumerate(HORIZONS):
        error = preds[:, i] - targets[:, i]
        rows.append({
            "horizon_min": horizon,
            "mae_mgdl": float(np.mean(np.abs(error))),
            "rmse_mgdl": float(np.sqrt(np.mean(error ** 2))),
            "n_samples": int(len(error)),
        })

    gate_summary = {
        "tcn_mean": float(gates[:, 0].mean()),
        "gru_mean": float(gates[:, 1].mean()),
        "transformer_mean": float(gates[:, 2].mean()),
        "tcn_sd": float(gates[:, 0].std()),
        "gru_sd": float(gates[:, 1].std()),
        "transformer_sd": float(gates[:, 2].std()),
    }

    return pd.DataFrame(rows), gate_summary


def run_patient(
    args,
    cohort,
    patient,
    device,
    output_dir,
):
    print("\n" + "=" * 70)
    print(f"COHORT={cohort} | PATIENT={patient}")
    print("=" * 70)

    train_df, test_df = load_patient_data(
        args.data_root, cohort, patient
    )

    features = get_feature_columns(train_df)

    print("Features:")
    print(", ".join(features))

    train_part, val_part = split_train_validation(
        train_df,
        args.val_fraction,
    )

    # Train-only imputation.
    imputer = fit_imputer(train_part, features)

    train_part = apply_imputer(train_part, imputer)
    val_part = apply_imputer(val_part, imputer)
    test_part = apply_imputer(test_df, imputer)

    # Train-only scaling.
    scaler = fit_scaler(train_part, features)

    train_scaled = apply_scaler(train_part, scaler)
    val_scaled = apply_scaler(val_part, scaler)
    test_scaled = apply_scaler(test_part, scaler)

    # Gap-safe sequences.
    X_train, y_train, train_meta = make_sequences(
        train_scaled, features, args.lookback
    )
    X_val, y_val, val_meta = make_sequences(
        val_scaled, features, args.lookback
    )
    X_test, y_test, test_meta = make_sequences(
        test_scaled, features, args.lookback
    )

    if min(len(X_train), len(X_val), len(X_test)) == 0:
        raise RuntimeError(
            f"{cohort}/{patient}: one split has zero valid sequences."
        )

    print(
        f"Sequences: train={len(X_train)}, "
        f"val={len(X_val)}, test={len(X_test)}"
    )

    train_loader = DataLoader(
        SequenceDataset(X_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        SequenceDataset(X_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        SequenceDataset(X_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = HybridTCNGRUTransformer(
        input_dim=len(features),
        d_model=args.d_model,
        gru_hidden=args.gru_hidden,
        gru_layers=args.gru_layers,
        tcn_levels=args.tcn_levels,
        transformer_heads=args.heads,
        transformer_layers=args.transformer_layers,
        horizons=HORIZONS,
        dropout=args.dropout,
        max_len=args.lookback // STEP_MIN,
    ).to(device)

    params = count_parameters(model)
    print(f"Trainable parameters: {params:,}")

    model, history, best_epoch, best_val = train_model(
        model,
        train_loader,
        val_loader,
        device,
        args.epochs,
        args.lr,
        args.weight_decay,
        args.patience,
        args.grad_clip,
    )

    results, gates = evaluate(
        model,
        test_loader,
        device,
        scaler,
    )

    results.insert(0, "patient", patient)
    results.insert(0, "cohort", cohort)
    results["lookback_min"] = args.lookback
    results["parameters"] = params
    results["best_epoch"] = best_epoch
    results["best_val_loss"] = best_val

    patient_dir = output_dir / cohort / str(patient)
    patient_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        patient_dir / "test_results.csv",
        index=False,
    )
    pd.DataFrame(history).to_csv(
        patient_dir / "training_history.csv",
        index=False,
    )

    with open(patient_dir / "config.json", "w") as f:
        json.dump(
            {
                "cohort": cohort,
                "patient": patient,
                "features": features,
                "lookback_min": args.lookback,
                "horizons": HORIZONS,
                "parameters": params,
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
                "gate_summary": gates,
                "imputer_medians": imputer,
                "scaler": scaler,
                "n_sequences": {
                    "train": len(X_train),
                    "validation": len(X_val),
                    "test": len(X_test),
                },
            },
            f,
            indent=2,
        )

    torch.save(
        model.state_dict(),
        patient_dir / "best_model.pt",
    )

    return results, gates


def main():
    args = parse_args()

    if args.lookback % STEP_MIN:
        raise ValueError("Lookback must be divisible by 5 minutes.")

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Lookback: {args.lookback} min")
    print(f"Horizons: {HORIZONS}")
    print("Protocol: gap-safe + train-only imputation/scaling")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    all_gates = []

    for cohort, patients in [
        ("ohio2018", PATIENTS_2018),
        ("ohio2020", PATIENTS_2020),
    ]:
        for patient in patients:
            results, gates = run_patient(
                args,
                cohort,
                patient,
                device,
                output_dir,
            )
            all_results.append(results)

            all_gates.append({
                "cohort": cohort,
                "patient": patient,
                **gates,
            })

    results = pd.concat(
        all_results,
        ignore_index=True,
    )

    results.to_csv(
        output_dir / "all_12_test_results.csv",
        index=False,
    )

    summary = (
        results
        .groupby(
            ["cohort", "horizon_min"],
            as_index=False,
        )
        .agg(
            mae_mean=("mae_mgdl", "mean"),
            mae_sd=("mae_mgdl", "std"),
            rmse_mean=("rmse_mgdl", "mean"),
            rmse_sd=("rmse_mgdl", "std"),
            n_patients=("patient", "nunique"),
        )
    )

    overall = (
        results
        .groupby("horizon_min", as_index=False)
        .agg(
            mae_mean=("mae_mgdl", "mean"),
            mae_sd=("mae_mgdl", "std"),
            rmse_mean=("rmse_mgdl", "mean"),
            rmse_sd=("rmse_mgdl", "std"),
            n_patients=("patient", "nunique"),
        )
    )

    summary.to_csv(
        output_dir / "cohort_summary.csv",
        index=False,
    )
    overall.to_csv(
        output_dir / "all12_summary.csv",
        index=False,
    )
    pd.DataFrame(all_gates).to_csv(
        output_dir / "gate_summary.csv",
        index=False,
    )

    print("\n" + "=" * 70)
    print("FINAL ALL-12 HYBRID RESULTS")
    print("=" * 70)
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
