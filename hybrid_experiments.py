from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from hybrid_models import HybridTCNGRUTransformer, count_parameters


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CGMSequenceDataset(Dataset):
    def __init__(self, data, feature_columns, target_columns, lookback_steps):
        values = data[feature_columns + target_columns].values.astype(np.float32)
        n_features = len(feature_columns)
        self.X = np.asarray(
            [values[i-lookback_steps:i, :n_features]
             for i in range(lookback_steps, len(data))],
            dtype=np.float32,
        )
        self.y = np.asarray(
            [values[i, n_features:]
             for i in range(lookback_steps, len(data))],
            dtype=np.float32,
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--patient", required=True)
    p.add_argument("--lookback", type=int, default=120)
    p.add_argument("--horizons", type=int, nargs="+",
                   default=[15, 30, 60, 90, 120])
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
    p.add_argument("--output-dir", default="output/phase2_hybrid")
    return p.parse_args()


def load_patient_data(root, patient):
    train = pd.read_csv(
        os.path.join(root, "train", f"{patient}.csv"),
        parse_dates=["timestamp"],
    )
    test = pd.read_csv(
        os.path.join(root, "test", f"{patient}.csv"),
        parse_dates=["timestamp"],
    )
    return (
        train.sort_values("timestamp").reset_index(drop=True),
        test.sort_values("timestamp").reset_index(drop=True),
    )


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


def add_targets(train, test, horizons):
    train, test = train.copy(), test.copy()
    for h in horizons:
        if h % 5:
            raise ValueError("Horizons must be multiples of 5 minutes.")
        steps = h // 5
        train[f"target_{h}"] = train["glucose"].shift(-steps)
        test[f"target_{h}"] = test["glucose"].shift(-steps)
    return train, test


def fit_imputer(df, columns):
    out = {}
    for c in columns:
        median = df[c].median()
        if not np.isfinite(median):
            raise ValueError(f"Invalid median for {c}")
        out[c] = float(median)
    return out


def apply_imputer(df, medians):
    df = df.copy()
    for c, median in medians.items():
        df[c] = (
            df[c]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(median)
        )
    return df


def fit_scaler(df, columns):
    out = {}
    for c in columns:
        mean = df[c].mean()
        std = df[c].std()
        if not np.isfinite(mean):
            raise ValueError(f"Invalid mean for {c}")
        if not np.isfinite(std) or std < 1e-8:
            std = 1.0
        out[c] = {"mean": float(mean), "std": float(std)}
    return out


def apply_scaler(df, scaler):
    df = df.copy()
    for c, s in scaler.items():
        df[c] = (df[c] - s["mean"]) / s["std"]
    return df


def sanity_check(df, columns, name):
    a = df[columns].to_numpy(dtype=np.float32)
    if not np.isfinite(a).all():
        bad = [c for c in columns
               if not np.isfinite(df[c].to_numpy(dtype=np.float32)).all()]
        raise ValueError(f"{name} contains NaN/Inf: {bad}")
    print(f"{name}: all features/targets are finite")


def train_model(model, train_loader, val_loader, device,
                epochs, lr, weight_decay, patience, grad_clip):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    wait = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            if not torch.isfinite(x).all():
                raise ValueError("Non-finite input batch.")
            if not torch.isfinite(y).all():
                raise ValueError("Non-finite target batch.")

            optimizer.zero_grad()
            pred = model(x)

            if not torch.isfinite(pred).all():
                raise ValueError("Model produced NaN/Inf predictions.")

            loss = criterion(pred, y)
            if not torch.isfinite(loss):
                raise ValueError("Training loss became NaN/Inf.")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
            f"Epoch {epoch:02d} | "
            f"Train {train_loss:.5f} | Val {val_loss:.5f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def evaluate_model(model, loader, device, horizons, scaler):
    model.eval()
    preds, targets, gates = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred, gate = model(x, return_gates=True)
            preds.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
            gates.append(gate.cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    gates = np.concatenate(gates)

    rows = []
    for i, h in enumerate(horizons):
        s = scaler[f"target_{h}"]
        pred = preds[:, i] * s["std"] + s["mean"]
        true = targets[:, i] * s["std"] + s["mean"]
        err = pred - true
        rows.append({
            "horizon": h,
            "mae_mgdl": float(np.mean(np.abs(err))),
            "rmse_mgdl": float(np.sqrt(np.mean(err ** 2))),
            "n_samples": len(true),
        })

    gate_summary = {
        "tcn_mean": float(gates[:, 0].mean()),
        "gru_mean": float(gates[:, 1].mean()),
        "transformer_mean": float(gates[:, 2].mean()),
        "tcn_std": float(gates[:, 0].std()),
        "gru_std": float(gates[:, 1].std()),
        "transformer_std": float(gates[:, 2].std()),
    }

    return pd.DataFrame(rows), gate_summary, gates


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Patient: {args.patient}")
    print(f"Lookback: {args.lookback} min")
    print(f"Horizons: {args.horizons}")

    train_df, test_df = load_patient_data(
        args.data_root, args.patient
    )

    features = get_feature_columns(train_df)

    print("\nFeatures:")
    for c in features:
        print(f"  - {c}")

    train_df, test_df = add_targets(
        train_df, test_df, args.horizons
    )

    targets = [f"target_{h}" for h in args.horizons]

    train_df = train_df.dropna(
        subset=targets
    ).reset_index(drop=True)
    test_df = test_df.dropna(
        subset=targets
    ).reset_index(drop=True)

    val_size = max(
        int(len(train_df) * args.val_fraction), 1
    )

    train_part = train_df.iloc[:-val_size].copy()
    val_part = train_df.iloc[-val_size:].copy()

    # Train-only imputation.
    imputer = fit_imputer(train_part, features)
    train_part = apply_imputer(train_part, imputer)
    val_part = apply_imputer(val_part, imputer)
    test_df = apply_imputer(test_df, imputer)

    # Train-only scaling.
    scaler = fit_scaler(
        train_part,
        features + targets,
    )
    train_part = apply_scaler(train_part, scaler)
    val_part = apply_scaler(val_part, scaler)
    test_scaled = apply_scaler(test_df, scaler)

    sanity_check(
        train_part, features + targets, "train"
    )
    sanity_check(
        val_part, features + targets, "validation"
    )
    sanity_check(
        test_scaled, features + targets, "test"
    )

    if args.lookback % 5:
        raise ValueError("Lookback must be a multiple of 5 minutes.")

    lookback_steps = args.lookback // 5

    train_dataset = CGMSequenceDataset(
        train_part, features, targets, lookback_steps
    )
    val_dataset = CGMSequenceDataset(
        val_part, features, targets, lookback_steps
    )
    test_dataset = CGMSequenceDataset(
        test_scaled, features, targets, lookback_steps
    )

    print(f"\nTrain sequences: {len(train_dataset)}")
    print(f"Validation sequences: {len(val_dataset)}")
    print(f"Test sequences: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    model = HybridTCNGRUTransformer(
        input_dim=len(features),
        d_model=args.d_model,
        gru_hidden=args.gru_hidden,
        gru_layers=args.gru_layers,
        tcn_levels=args.tcn_levels,
        transformer_heads=args.heads,
        transformer_layers=args.transformer_layers,
        horizons=args.horizons,
        dropout=args.dropout,
        max_len=lookback_steps,
    ).to(device)

    params = count_parameters(model)
    print(f"\nTrainable parameters: {params:,}")

    model, history = train_model(
        model, train_loader, val_loader, device,
        args.epochs, args.lr, args.weight_decay,
        args.patience, args.grad_clip
    )

    results, gate_summary, gate_values = evaluate_model(
        model, test_loader, device, args.horizons, scaler
    )

    results.insert(0, "patient", args.patient)
    results.insert(1, "lookback_min", args.lookback)

    print("\nTest results:")
    print(results.to_string(index=False))

    print("\nAverage adaptive fusion weights:")
    print(f"TCN:         {gate_summary['tcn_mean']:.4f}")
    print(f"GRU:         {gate_summary['gru_mean']:.4f}")
    print(f"Transformer: {gate_summary['transformer_mean']:.4f}")

    prefix = os.path.join(
        args.output_dir,
        f"patient_{args.patient}"
    )

    results.to_csv(f"{prefix}_results.csv", index=False)

    pd.DataFrame(
        gate_values,
        columns=[
            "tcn_gate",
            "gru_gate",
            "transformer_gate",
        ],
    ).to_csv(f"{prefix}_gates.csv", index=False)

    pd.DataFrame(history).to_csv(
        f"{prefix}_training.csv", index=False
    )

    torch.save(
        model.state_dict(),
        f"{prefix}_model.pt",
    )

    config = {
        "patient": args.patient,
        "lookback_min": args.lookback,
        "lookback_steps": lookback_steps,
        "horizons": args.horizons,
        "features": features,
        "parameters": params,
        "d_model": args.d_model,
        "gru_hidden": args.gru_hidden,
        "gru_layers": args.gru_layers,
        "tcn_levels": args.tcn_levels,
        "transformer_heads": args.heads,
        "transformer_layers": args.transformer_layers,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "imputer_medians": imputer,
        "scaler": scaler,
        "gate_summary": gate_summary,
    }

    with open(f"{prefix}_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\nSaved:")
    print(f"{prefix}_results.csv")
    print(f"{prefix}_gates.csv")
    print(f"{prefix}_training.csv")
    print(f"{prefix}_model.pt")
    print(f"{prefix}_config.json")


if __name__ == "__main__":
    main()
