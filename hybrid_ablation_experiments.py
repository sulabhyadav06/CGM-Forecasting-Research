from __future__ import annotations

import argparse
import json
import os
import random
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from hybrid_models import (
    TCNBranch,
    GRUBranch,
    TransformerBranch,
    AdaptiveFusion,
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset
# ============================================================

class CGMSequenceDataset(Dataset):
    """Windowed sequences with explicit anchor bounds for chronological splits."""

    def __init__(self, data, features, targets, lookback_steps, start_anchor=None, end_anchor=None):
        self.values = data[features + targets].values.astype(np.float32)
        self.n_features = len(features)
        self.lookback_steps = lookback_steps
        self.start_anchor = max(lookback_steps, start_anchor if start_anchor is not None else lookback_steps)
        self.end_anchor = len(data) if end_anchor is None else min(end_anchor, len(data))
        self.end_anchor = max(self.start_anchor, self.end_anchor)
        self.anchors = range(self.start_anchor, self.end_anchor)

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        i = self.anchors[idx]
        x = self.values[i-self.lookback_steps:i, :self.n_features]
        y = self.values[i, self.n_features:]
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"Non-finite sequence or target at anchor row {i}.")
        return torch.tensor(x), torch.tensor(y)

# ============================================================
# Generic ablation model
# ============================================================

class AblationModel(nn.Module):
    """
    Shared embedding followed by selectable branches.

    Architectures:
      gru
      tcn
      transformer
      tcn_gru
      gru_transformer
      tcn_transformer
      tcn_gru_transformer
      adaptive_tcn_gru_transformer

    For non-adaptive variants, branch representations are
    concatenated and projected back to d_model.

    The adaptive variant uses the AdaptiveFusion module.
    """

    def __init__(
        self,
        input_dim,
        architecture,
        d_model=56,
        gru_hidden=56,
        gru_layers=2,
        tcn_levels=3,
        transformer_heads=2,
        transformer_layers=1,
        horizons=(15, 30, 60, 90, 120),
        dropout=0.1,
        max_len=24,
    ):
        super().__init__()

        self.architecture = architecture
        self.horizons = tuple(horizons)
        self.d_model = d_model

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.use_tcn = architecture in {
            "tcn",
            "tcn_gru",
            "tcn_transformer",
            "tcn_gru_transformer",
            "adaptive_tcn_gru_transformer",
        }

        self.use_gru = architecture in {
            "gru",
            "tcn_gru",
            "gru_transformer",
            "tcn_gru_transformer",
            "adaptive_tcn_gru_transformer",
        }

        self.use_transformer = architecture in {
            "transformer",
            "gru_transformer",
            "tcn_transformer",
            "tcn_gru_transformer",
            "adaptive_tcn_gru_transformer",
        }

        if self.use_tcn:
            self.tcn = TCNBranch(
                d_model=d_model,
                levels=tcn_levels,
                dropout=dropout,
            )

        if self.use_gru:
            self.gru = GRUBranch(
                d_model=d_model,
                hidden_size=gru_hidden,
                layers=gru_layers,
                dropout=dropout,
            )

        if self.use_transformer:
            self.transformer = TransformerBranch(
                d_model=d_model,
                heads=transformer_heads,
                layers=transformer_layers,
                dropout=dropout,
                max_len=max_len,
            )

        n_branches = sum([
            self.use_tcn,
            self.use_gru,
            self.use_transformer,
        ])

        self.n_branches = n_branches

        if architecture == "adaptive_tcn_gru_transformer":
            self.fusion = AdaptiveFusion(
                d_model=d_model,
                dropout=dropout,
            )
            self.fusion_projection = nn.Identity()
        else:
            self.fusion = None
            self.fusion_projection = nn.Sequential(
                nn.Linear(
                    n_branches * d_model,
                    d_model,
                ),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.shared = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.heads = nn.ModuleDict({
            str(h): nn.Linear(d_model, 1)
            for h in self.horizons
        })

    def forward(self, x, return_gates=False):
        x = self.embedding(x)

        reps = []
        tcn_repr = None
        gru_repr = None
        transformer_repr = None

        if self.use_tcn:
            tcn_repr = self.tcn(x)
            reps.append(tcn_repr)

        if self.use_gru:
            gru_repr = self.gru(x)
            reps.append(gru_repr)

        if self.use_transformer:
            transformer_repr = self.transformer(x)
            reps.append(transformer_repr)

        gates = None

        if self.architecture == "adaptive_tcn_gru_transformer":
            fused, gates = self.fusion(
                tcn_repr,
                gru_repr,
                transformer_repr,
            )
        else:
            fused = self.fusion_projection(
                torch.cat(reps, dim=-1)
            )

        fused = self.shared(fused)

        prediction = torch.cat(
            [
                self.heads[str(h)](fused)
                for h in self.horizons
            ],
            dim=-1,
        )

        if return_gates:
            return prediction, gates

        return prediction


# ============================================================
# Arguments
# ============================================================

ARCHITECTURES = [
    "gru",
    "tcn",
    "transformer",
    "tcn_gru",
    "gru_transformer",
    "tcn_transformer",
    "tcn_gru_transformer",
    "adaptive_tcn_gru_transformer",
]


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data-root", required=True)
    p.add_argument("--patient", required=True)

    p.add_argument("--lookback", type=int, default=120)
    p.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[15, 30, 60, 90, 120],
    )

    p.add_argument(
        "--architectures",
        nargs="+",
        default=["gru", "tcn", "transformer",
                 "tcn_gru", "gru_transformer",
                 "tcn_transformer",
                 "tcn_gru_transformer",
                 "adaptive_tcn_gru_transformer"],
        choices=ARCHITECTURES,
    )

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

    p.add_argument(
        "--output-dir",
        default="output/phase2_hybrid_ablations",
    )

    return p.parse_args()


# ============================================================
# Data preparation
# ============================================================

def load_data(root, patient):
    train = pd.read_csv(
        os.path.join(root, "train", f"{patient}.csv"),
        parse_dates=["timestamp"],
    )
    test = pd.read_csv(
        os.path.join(root, "test", f"{patient}.csv"),
        parse_dates=["timestamp"],
    )

    train = train.sort_values("timestamp").reset_index(drop=True)
    test = test.sort_values("timestamp").reset_index(drop=True)

    return train, test


def get_features(df):
    excluded = {
        "timestamp",
        "gap_from_previous_min",
        "valid_5min_interval",
        "carbs_observed_60min",
        "bolus_observed_60min",
    }

    features = [
        c for c in df.columns
        if c not in excluded
    ]

    features.remove("glucose")

    return ["glucose"] + features


def add_targets(train, test, horizons):
    train = train.copy()
    test = test.copy()

    for h in horizons:
        if h % 5:
            raise ValueError(
                "All horizons must be multiples of 5."
            )

        steps = h // 5

        train[f"target_{h}"] = (
            train["glucose"].shift(-(steps - 1))
        )

        test[f"target_{h}"] = (
            test["glucose"].shift(-(steps - 1))
        )

    return train, test


def fit_imputer(df, columns):
    medians = {}

    for c in columns:
        m = df[c].median()

        if not np.isfinite(m):
            raise ValueError(
                f"Invalid training median: {c}"
            )

        medians[c] = float(m)

    return medians


def apply_imputer(df, medians):
    df = df.copy()

    for c, m in medians.items():
        df[c] = (
            df[c]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(m)
        )

    return df


def fit_scaler(df, columns):
    scaler = {}

    for c in columns:
        mean = df[c].mean()
        std = df[c].std()

        if not np.isfinite(mean):
            raise ValueError(
                f"Invalid mean: {c}"
            )

        if not np.isfinite(std) or std < 1e-8:
            std = 1.0

        scaler[c] = {
            "mean": float(mean),
            "std": float(std),
        }

    return scaler


def apply_scaler(df, scaler):
    df = df.copy()

    for c, s in scaler.items():
        df[c] = (
            df[c] - s["mean"]
        ) / s["std"]

    return df


# ============================================================
# Training / evaluation
# ============================================================

def train_one(
    model,
    train_loader,
    val_loader,
    device,
    args,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    wait = 0
    history = []

    for epoch in range(1, args.epochs + 1):

        model.train()
        train_losses = []

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            pred = model(x)
            loss = criterion(pred, y)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Training loss became NaN/Inf."
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip,
            )

            optimizer.step()

            train_losses.append(
                loss.item()
            )

        model.eval()
        val_losses = []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                pred = model(x)
                loss = criterion(pred, y)

                val_losses.append(
                    loss.item()
                )

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        print(
            f"    Epoch {epoch:02d} | "
            f"Train {train_loss:.5f} | "
            f"Val {val_loss:.5f}"
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

            if wait >= args.patience:
                print("    Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def evaluate(
    model,
    loader,
    device,
    horizons,
    scaler,
):
    model.eval()

    all_preds = []
    all_targets = []
    all_gates = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            pred, gates = model(
                x,
                return_gates=True,
            )

            all_preds.append(
                pred.cpu().numpy()
            )

            all_targets.append(
                y.cpu().numpy()
            )

            if gates is not None:
                all_gates.append(
                    gates.cpu().numpy()
                )

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    rows = []

    for i, h in enumerate(horizons):

        s = scaler[f"target_{h}"]

        pred = (
            preds[:, i] * s["std"]
            + s["mean"]
        )

        true = (
            targets[:, i] * s["std"]
            + s["mean"]
        )

        error = pred - true

        rows.append({
            "horizon": h,
            "mae_mgdl": float(
                np.mean(np.abs(error))
            ),
            "rmse_mgdl": float(
                np.sqrt(np.mean(error ** 2))
            ),
            "n_samples": len(true),
        })

    gate_summary = None

    if all_gates:
        gates = np.concatenate(
            all_gates
        )

        gate_summary = {
            "tcn_mean": float(gates[:, 0].mean()),
            "gru_mean": float(gates[:, 1].mean()),
            "transformer_mean": float(gates[:, 2].mean()),
            "tcn_std": float(gates[:, 0].std()),
            "gru_std": float(gates[:, 1].std()),
            "transformer_std": float(gates[:, 2].std()),
        }

    return (
        pd.DataFrame(rows),
        gate_summary,
        all_gates,
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    print(f"Device: {device}")
    print(f"Patient: {args.patient}")
    print(f"Lookback: {args.lookback} min")
    print(f"Horizons: {args.horizons}")

    train_df, test_df = load_data(
        args.data_root,
        args.patient,
    )

    features = get_features(train_df)

    print("\nFeatures:")
    for f in features:
        print(f"  - {f}")

    train_df, test_df = add_targets(
        train_df,
        test_df,
        args.horizons,
    )

    targets = [
        f"target_{h}"
        for h in args.horizons
    ]

    if args.lookback % 5:
        raise ValueError("Lookback must be a multiple of 5 minutes.")
    lookback_steps = args.lookback // 5
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1.")

    # Chronological split is defined before building windows. With inputs
    # ending at i-1, horizon h targets row i + h/5 - 1.
    split_idx = int(len(train_df) * (1.0 - args.val_fraction))
    max_offset = max(args.horizons) // 5 - 1
    train_end = split_idx - max_offset
    val_end = len(train_df) - max_offset
    if train_end <= lookback_steps or val_end <= split_idx:
        raise ValueError(
            "Not enough rows for the requested lookback, horizons, and validation fraction."
        )

    # Keep pre-split history available to validation windows, while excluding
    # every training anchor whose longest-horizon label crosses the boundary.
    test_df = test_df.dropna(subset=targets).reset_index(drop=True)
    train_fit_rows = train_df.iloc[:split_idx].copy()
    train_anchor_rows = train_df.iloc[lookback_steps:train_end].copy()
    imputer = fit_imputer(train_fit_rows, features)
    train_scaled = apply_imputer(train_df, imputer)
    test_df = apply_imputer(test_df, imputer)
    scaler = fit_scaler(train_anchor_rows, features + targets)
    train_scaled = apply_scaler(train_scaled, scaler)
    test_scaled = apply_scaler(test_df, scaler)

    sanity_check(train_scaled.iloc[lookback_steps:train_end], features + targets, "train")
    sanity_check(train_scaled.iloc[split_idx:val_end], features + targets, "validation")
    sanity_check(test_scaled, features + targets, "test")

    train_dataset = CGMSequenceDataset(
        train_scaled, features, targets, lookback_steps,
        start_anchor=lookback_steps, end_anchor=train_end,
    )

    val_dataset = CGMSequenceDataset(
        train_scaled, features, targets, lookback_steps,
        start_anchor=split_idx, end_anchor=val_end,
    )

    test_dataset = CGMSequenceDataset(
        test_scaled,
        features,
        targets,
        lookback_steps,
    )

    print(
        f"\nTrain sequences: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation sequences: "
        f"{len(val_dataset)}"
    )

    print(
        f"Test sequences: "
        f"{len(test_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    summary_rows = []

    for architecture in args.architectures:

        print("\n" + "=" * 70)
        print(
            f"ARCHITECTURE: {architecture}"
        )
        print("=" * 70)

        # Re-seed so each architecture starts
        # from a deterministic initialization.
        set_seed(args.seed)

        model = AblationModel(
            input_dim=len(features),
            architecture=architecture,
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

        parameters = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        print(
            f"Parameters: {parameters:,}"
        )

        model, history = train_one(
            model,
            train_loader,
            val_loader,
            device,
            args,
        )

        results, gate_summary, gate_values = evaluate(
            model,
            test_loader,
            device,
            args.horizons,
            scaler,
        )

        results.insert(
            0,
            "architecture",
            architecture,
        )

        results.insert(
            1,
            "patient",
            args.patient,
        )

        results.insert(
            2,
            "parameters",
            parameters,
        )

        summary_rows.append(
            results
        )

        prefix = os.path.join(
            args.output_dir,
            f"{args.patient}_{architecture}",
        )

        results.to_csv(
            f"{prefix}_results.csv",
            index=False,
        )

        pd.DataFrame(history).to_csv(
            f"{prefix}_training.csv",
            index=False,
        )

        torch.save(
            model.state_dict(),
            f"{prefix}_model.pt",
        )

        if gate_values:

            gates = np.concatenate(
                gate_values
            )

            pd.DataFrame(
                gates,
                columns=[
                    "tcn_gate",
                    "gru_gate",
                    "transformer_gate",
                ],
            ).to_csv(
                f"{prefix}_gates.csv",
                index=False,
            )

        config = {
            "architecture": architecture,
            "patient": args.patient,
            "lookback_min": args.lookback,
            "horizons": args.horizons,
            "features": features,
            "parameters": parameters,
            "gate_summary": gate_summary,
            "imputer_medians": imputer,
            "scaler": scaler,
        }

        with open(
            f"{prefix}_config.json",
            "w",
        ) as f:
            json.dump(
                config,
                f,
                indent=2,
            )

        print("\nTest results:")
        print(
            results.to_string(
                index=False
            )
        )

        if gate_summary:
            print("\nFusion weights:")
            print(
                f"  TCN: "
                f"{gate_summary['tcn_mean']:.4f}"
            )
            print(
                f"  GRU: "
                f"{gate_summary['gru_mean']:.4f}"
            )
            print(
                f"  Transformer: "
                f"{gate_summary['transformer_mean']:.4f}"
            )

    combined = pd.concat(
        summary_rows,
        ignore_index=True,
    )

    combined_path = os.path.join(
        args.output_dir,
        f"patient_{args.patient}_all_ablations.csv",
    )

    combined.to_csv(
        combined_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("ALL ABLATIONS")
    print("=" * 70)

    print(
        combined.to_string(
            index=False
        )
    )

    print(
        f"\nSaved combined results: "
        f"{combined_path}"
    )


if __name__ == "__main__":
    main()
