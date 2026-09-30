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

from hybrid_models_direct_residual import (
    TCNBranch,
    GRUBranch,
    TransformerBranch,
    count_parameters,
)


VARIANTS = ["GRU", "GRU-TCN", "GRU-Transformer", "Full"]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CGMSequenceDataset(Dataset):
    def __init__(self, data, features, targets, lookback_steps):
        values = data[features + targets].values.astype(np.float32)
        nf = len(features)
        self.X = np.asarray([
            values[i - lookback_steps:i, :nf]
            for i in range(lookback_steps, len(data))
        ], dtype=np.float32)
        self.y = np.asarray([
            values[i, nf:]
            for i in range(lookback_steps, len(data))
        ], dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--patient", required=True)
    p.add_argument("--lookback", type=int, default=120)
    p.add_argument("--horizons", type=int, nargs="+", default=[15, 30, 60, 90, 120])
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
    p.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    p.add_argument("--output-dir", default="output/phase2_residual_ablation")
    return p.parse_args()


def load_data(root, patient):
    train = pd.read_csv(os.path.join(root, "train", f"{patient}.csv"), parse_dates=["timestamp"])
    test = pd.read_csv(os.path.join(root, "test", f"{patient}.csv"), parse_dates=["timestamp"])
    return (
        train.sort_values("timestamp").reset_index(drop=True),
        test.sort_values("timestamp").reset_index(drop=True),
    )


def get_features(df):
    excluded = {"timestamp", "gap_from_previous_min", "valid_5min_interval",
                "carbs_observed_60min", "bolus_observed_60min"}
    cols = [c for c in df.columns if c not in excluded]
    cols.remove("glucose")
    return ["glucose"] + cols


def add_targets(df, horizons):
    df = df.copy()
    for h in horizons:
        if h % 5:
            raise ValueError("Horizons must be multiples of 5 minutes.")
        df[f"target_{h}"] = df["glucose"].shift(-(h // 5))
    return df


def fit_imputer(df, columns):
    med = {}
    for c in columns:
        x = df[c].replace([np.inf, -np.inf], np.nan)
        m = x.median()
        if not np.isfinite(m):
            raise ValueError(f"Invalid median for {c}")
        med[c] = float(m)
    return med


def apply_imputer(df, medians):
    df = df.copy()
    for c, m in medians.items():
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(m)
    return df


def fit_scaler(df, columns):
    out = {}
    for c in columns:
        mean = float(df[c].mean())
        std = float(df[c].std())
        if not np.isfinite(mean):
            raise ValueError(f"Invalid mean for {c}")
        if not np.isfinite(std) or std < 1e-8:
            std = 1.0
        out[c] = {"mean": mean, "std": std}
    return out


def apply_scaler(df, scaler):
    df = df.copy()
    for c, s in scaler.items():
        df[c] = (df[c] - s["mean"]) / s["std"]
    return df


def make_split(train_df, lookback_steps, max_horizon_steps, val_fraction):
    n = len(train_df)
    split = int(n * (1.0 - val_fraction))
    purge = lookback_steps + max_horizon_steps
    train_end = split - purge
    if train_end <= lookback_steps:
        raise ValueError("Not enough training data after purge.")
    train_part = train_df.iloc[:train_end].copy()
    val_part = train_df.iloc[split:].copy()
    return train_part, val_part, purge


class ResidualAblationModel(nn.Module):
    """Four controlled variants sharing the same feature embedding and branch widths.

    GRU: base GRU forecast only.
    GRU-TCN: GRU base + gated TCN residual.
    GRU-Transformer: GRU base + gated Transformer residual.
    Full: GRU base + gated TCN + gated Transformer residuals.
    """

    def __init__(self, input_dim, horizons, variant, d_model=56, gru_hidden=56,
                 gru_layers=2, tcn_levels=3, heads=2, transformer_layers=1,
                 dropout=0.1, max_len=512):
        super().__init__()
        self.variant = variant
        self.horizons = tuple(horizons)
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.gru = GRUBranch(d_model, gru_hidden, gru_layers, dropout)
        self.tcn = TCNBranch(d_model, tcn_levels, dropout) if variant in {"GRU-TCN", "Full"} else None
        self.transformer = TransformerBranch(
            d_model, heads, transformer_layers, dropout, max_len
        ) if variant in {"GRU-Transformer", "Full"} else None

        self.heads = nn.ModuleDict()
        for h in self.horizons:
            modules = {
                "base": nn.Sequential(
                    nn.Linear(d_model, d_model), nn.GELU(),
                    nn.Dropout(dropout), nn.Linear(d_model, 1)
                )
            }
            if variant in {"GRU-TCN", "Full"}:
                modules["tcn_delta"] = nn.Sequential(
                    nn.Linear(d_model, d_model), nn.GELU(),
                    nn.Dropout(dropout), nn.Linear(d_model, 1)
                )
                modules["tcn_gate"] = nn.Sequential(
                    nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(d_model, 1)
                )
                modules["tcn_scale"] = nn.Parameter(torch.tensor(0.10))
            if variant in {"GRU-Transformer", "Full"}:
                modules["tr_delta"] = nn.Sequential(
                    nn.Linear(d_model, d_model), nn.GELU(),
                    nn.Dropout(dropout), nn.Linear(d_model, 1)
                )
                modules["tr_gate"] = nn.Sequential(
                    nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(d_model, 1)
                )
                modules["tr_scale"] = nn.Parameter(torch.tensor(0.10))
            # ModuleDict cannot store Parameters, so register them separately below.
            self.heads[str(h)] = nn.ModuleDict({k: v for k, v in modules.items() if isinstance(v, nn.Module)})
            if "tcn_scale" in modules:
                self.register_parameter(f"tcn_scale_{h}", modules["tcn_scale"])
            if "tr_scale" in modules:
                self.register_parameter(f"tr_scale_{h}", modules["tr_scale"])

    def forward(self, x, return_details=False):
        x = self.embedding(x)
        gru = self.gru(x)
        tcn = self.tcn(x) if self.tcn is not None else None
        tr = self.transformer(x) if self.transformer is not None else None

        predictions, bases = [], []
        tcn_contribs, tr_contribs, tcn_gates, tr_gates = [], [], [], []
        for h in self.horizons:
            head = self.heads[str(h)]
            base = head["base"](gru)
            pred = base
            gate_input = torch.cat([tcn if tcn is not None else gru,
                                    gru,
                                    tr if tr is not None else gru], dim=-1)
            tc = torch.zeros_like(base)
            trc = torch.zeros_like(base)
            tg = torch.zeros_like(base)
            rg = torch.zeros_like(base)
            if self.variant in {"GRU-TCN", "Full"}:
                tg = torch.sigmoid(head["tcn_gate"](gate_input))
                tc = getattr(self, f"tcn_scale_{h}") * tg * head["tcn_delta"](tcn)
                pred = pred + tc
            if self.variant in {"GRU-Transformer", "Full"}:
                rg = torch.sigmoid(head["tr_gate"](gate_input))
                trc = getattr(self, f"tr_scale_{h}") * rg * head["tr_delta"](tr)
                pred = pred + trc
            predictions.append(pred)
            bases.append(base)
            tcn_contribs.append(tc)
            tr_contribs.append(trc)
            tcn_gates.append(tg)
            tr_gates.append(rg)

        out = torch.cat(predictions, dim=-1)
        if not return_details:
            return out
        return {
            "prediction": out,
            "base": torch.cat(bases, dim=-1),
            "tcn_contribution": torch.cat(tcn_contribs, dim=-1),
            "transformer_contribution": torch.cat(tr_contribs, dim=-1),
            "tcn_gate": torch.cat(tcn_gates, dim=-1),
            "transformer_gate": torch.cat(tr_gates, dim=-1),
        }


def train_model(model, train_loader, val_loader, device, args):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    wait = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step(); losses.append(loss.item())
        model.eval(); vloss = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                vloss.append(criterion(model(x), y).item())
        tr = float(np.mean(losses)); va = float(np.mean(vloss))
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va})
        print(f"    Epoch {epoch:02d} | Train {tr:.5f} | Val {va:.5f}")
        if va < best_val:
            best_val = va; wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def evaluate(model, loader, device, horizons, scaler):
    model.eval(); store = {k: [] for k in ["prediction", "target", "base", "tcn_contribution", "transformer_contribution", "tcn_gate", "transformer_gate"]}
    with torch.no_grad():
        for x, y in loader:
            d = model(x.to(device), return_details=True)
            store["prediction"].append(d["prediction"].cpu().numpy())
            store["target"].append(y.numpy())
            for k in store:
                if k not in {"prediction", "target"}:
                    store[k].append(d[k].cpu().numpy())
    for k in store:
        store[k] = np.concatenate(store[k])

    rows = []; diagnostics = []
    for i, h in enumerate(horizons):
        s = scaler[f"target_{h}"]
        pred = store["prediction"][:, i] * s["std"] + s["mean"]
        true = store["target"][:, i] * s["std"] + s["mean"]
        base = store["base"][:, i] * s["std"] + s["mean"]
        err = pred - true; berr = base - true
        rows.append({
            "horizon": h,
            "mae_mgdl": float(np.mean(np.abs(err))),
            "rmse_mgdl": float(np.sqrt(np.mean(err ** 2))),
            "base_gru_mae_mgdl": float(np.mean(np.abs(berr))),
            "base_gru_rmse_mgdl": float(np.sqrt(np.mean(berr ** 2))),
            "n_samples": len(true),
        })
        tc = store["tcn_contribution"][:, i] * s["std"]
        rc = store["transformer_contribution"][:, i] * s["std"]
        diagnostics.append({
            "horizon": h,
            "mean_abs_tcn_contribution_mgdl": float(np.mean(np.abs(tc))),
            "mean_signed_tcn_contribution_mgdl": float(np.mean(tc)),
            "mean_abs_transformer_contribution_mgdl": float(np.mean(np.abs(rc))),
            "mean_signed_transformer_contribution_mgdl": float(np.mean(rc)),
            "mean_abs_total_correction_mgdl": float(np.mean(np.abs(pred - base))),
            "mean_abs_tcn_gate": float(np.mean(store["tcn_gate"][:, i])),
            "mean_abs_transformer_gate": float(np.mean(store["transformer_gate"][:, i])),
        })
    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def run_variant(variant, train_df, val_df, test_df, features, targets, args, scaler, device):
    lookback_steps = args.lookback // 5
    train_ds = CGMSequenceDataset(train_df, features, targets, lookback_steps)
    val_ds = CGMSequenceDataset(val_df, features, targets, lookback_steps)
    test_ds = CGMSequenceDataset(test_df, features, targets, lookback_steps)
    loaders = [
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=args.batch_size, shuffle=False),
    ]
    model = ResidualAblationModel(
        input_dim=len(features), horizons=args.horizons, variant=variant,
        d_model=args.d_model, gru_hidden=args.gru_hidden, gru_layers=args.gru_layers,
        tcn_levels=args.tcn_levels, heads=args.heads,
        transformer_layers=args.transformer_layers, dropout=args.dropout,
        max_len=lookback_steps,
    ).to(device)
    params = count_parameters(model)
    print(f"\n=== {variant} | parameters={params:,} ===")
    model, history = train_model(model, loaders[0], loaders[1], device, args)
    results, diagnostics = evaluate(model, loaders[2], device, args.horizons, scaler)
    results.insert(0, "variant", variant); results.insert(1, "patient", args.patient)
    results.insert(2, "lookback_min", args.lookback); results.insert(3, "parameters", params)
    diagnostics.insert(0, "variant", variant); diagnostics.insert(1, "patient", args.patient)
    return results, diagnostics, pd.DataFrame(history), model


def main():
    args = parse_args(); set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_raw, test_raw = load_data(args.data_root, args.patient)
    features = get_features(train_raw)
    train_raw = add_targets(train_raw, args.horizons)
    test_raw = add_targets(test_raw, args.horizons)
    targets = [f"target_{h}" for h in args.horizons]
    train_raw = train_raw.dropna(subset=targets).reset_index(drop=True)
    test_raw = test_raw.dropna(subset=targets).reset_index(drop=True)

    lb = args.lookback // 5; mh = max(args.horizons) // 5
    train_part, val_part, purge = make_split(train_raw, lb, mh, args.val_fraction)
    imputer = fit_imputer(train_part, features)
    train_part = apply_imputer(train_part, imputer)
    val_part = apply_imputer(val_part, imputer)
    test_part = apply_imputer(test_raw, imputer)
    scaler = fit_scaler(train_part, features + targets)
    train_part = apply_scaler(train_part, scaler)
    val_part = apply_scaler(val_part, scaler)
    test_part = apply_scaler(test_part, scaler)

    print(f"Patient {args.patient} | lookback={args.lookback} | purge={purge} rows")
    print(f"Features ({len(features)}): {features}")
    all_results=[]; all_diag=[]
    for j, variant in enumerate(args.variants):
        set_seed(args.seed + j)
        r, d, history, model = run_variant(variant, train_part, val_part, test_part, features, targets, args, scaler, device)
        print(r.to_string(index=False))
        print(d.to_string(index=False))
        all_results.append(r); all_diag.append(d)
        prefix=os.path.join(args.output_dir, f"patient_{args.patient}_{variant.lower().replace('-', '_')}")
        r.to_csv(prefix + "_results.csv", index=False)
        d.to_csv(prefix + "_diagnostics.csv", index=False)
        history.to_csv(prefix + "_training.csv", index=False)
        torch.save(model.state_dict(), prefix + "_model.pt")

    results=pd.concat(all_results, ignore_index=True)
    diagnostics=pd.concat(all_diag, ignore_index=True)
    results.to_csv(os.path.join(args.output_dir, f"patient_{args.patient}_all_results.csv"), index=False)
    diagnostics.to_csv(os.path.join(args.output_dir, f"patient_{args.patient}_all_diagnostics.csv"), index=False)
    with open(os.path.join(args.output_dir, f"patient_{args.patient}_config.json"), "w") as f:
        json.dump({"args": vars(args), "features": features, "purge_rows": purge, "imputer": imputer, "scaler": scaler}, f, indent=2)
    print("\n=== Combined results ===")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
