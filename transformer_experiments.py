"""
Phase-2 Improved Transformer pilot/experiment runner.

Start with a small pilot before launching the full 12-patient experiment.

Example:
    python transformer_experiments.py \
        --data-root /path/to/ohio2018_multimodal/features \
        --split train \
        --epochs 10

The script is intentionally conservative:
- deterministic seed
- train-only standardization
- patient-wise sequences
- no sequence across timestamp gaps
- validation is a temporal tail of the TRAIN split
- test remains untouched until final evaluation
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
from torch import nn
from torch.utils.data import DataLoader, Subset

from improved_transformer import ImprovedTransformer
from multimodal_dataset import CGMSequenceDataset, SequenceConfig


SEED = 42


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinism is useful for research comparisons.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_transformer_parameters(
    input_dim: int,
    d_model: int,
    heads: int,
    layers: int,
    pooling: str,
):
    """Return the exact parameter count for the existing Transformer class."""
    model = ImprovedTransformer(
        input_dim=input_dim,
        d_model=d_model,
        nhead=heads,
        num_layers=layers,
        pooling=pooling,
        num_outputs=1,
    )
    return sum(p.numel() for p in model.parameters())


def print_parameter_budget(input_dim: int, pooling: str = "attention"):
    """
    Print candidate Transformer configurations near the recurrent baseline
    parameter budget (~45k–60k) without training anything.
    """
    target_low = 40_000
    target_high = 65_000
    candidates = []

    for d_model in [48, 56, 64, 72, 80, 96]:
        for heads in [2, 4, 6, 8]:
            if d_model % heads != 0:
                continue
            for layers in [1, 2, 3]:
                params = count_transformer_parameters(
                    input_dim, d_model, heads, layers, pooling
                )
                if target_low <= params <= target_high:
                    candidates.append(
                        {
                            "d_model": d_model,
                            "heads": heads,
                            "layers": layers,
                            "parameters": params,
                            "distance_to_52k": abs(params - 52_000),
                        }
                    )

    candidates.sort(key=lambda x: (x["distance_to_52k"], x["parameters"]))

    print("\nParameter-matched Transformer candidates")
    print("Target comparison range: 40k–65k parameters")
    print("Reference baselines: GRU ~45.4k, TCN ~46.7k, LSTM ~59.1k")
    print()
    print(f"{'d_model':>8} {'heads':>6} {'layers':>7} {'parameters':>12}")
    print("-" * 40)
    for c in candidates:
        print(
            f"{c['d_model']:>8} {c['heads']:>6} {c['layers']:>7} "
            f"{c['parameters']:>12,}"
        )

    return candidates


def fit_standardizer(df: pd.DataFrame, columns: list[str]):
    """Fit numeric imputation/scaling statistics on training rows only."""
    numeric = df[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mean = numeric.mean().fillna(0.0)
    std = numeric.std(ddof=0).replace(0, 1.0).fillna(1.0)
    std = std.where(std.abs() > 1e-8, 1.0)
    return mean, std


def apply_standardizer(
    df: pd.DataFrame,
    columns: list[str],
    mean: pd.Series,
    std: pd.Series,
):
    out = df.copy()
    numeric = out[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(mean[columns])
    out[columns] = (numeric - mean[columns]) / std[columns]
    return out


def train_one(
    model,
    train_loader,
    val_loader,
    device,
    epochs=20,
    lr=1e-3,
    weight_decay=1e-4,
    patience=5,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(-1)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device).unsqueeze(-1)
                pred = model(x)
                val_losses.append(criterion(pred, y).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))

        history.append(
            {
                "epoch": epoch,
                "train_mse": train_loss,
                "val_mse": val_loss,
            }
        )

        print(
            f"Epoch {epoch:02d} | train MSE={train_loss:.6f} "
            f"| val MSE={val_loss:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print("Early stopping.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


@torch.no_grad()
def evaluate(model, loader, device, glucose_mean, glucose_std):
    """
    Evaluate on the original glucose scale (mg/dL).

    The model is trained on standardized glucose targets, so predictions
    and targets are inverse-standardized before calculating MAE/RMSE.
    """
    model.eval()
    ys = []
    preds = []

    for x, y in loader:
        pred = model(x.to(device)).squeeze(-1).cpu().numpy()
        ys.append(y.numpy())
        preds.append(pred)

    y_std = np.concatenate(ys)
    p_std = np.concatenate(preds)

    # Return to the original glucose scale before reporting clinical metrics.
    y = y_std * float(glucose_std) + float(glucose_mean)
    p = p_std * float(glucose_std) + float(glucose_mean)

    mae = float(np.mean(np.abs(y - p)))
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "n": len(y),
        "metric_unit": "mg/dL",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--patient", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lookback", type=int, default=120)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument(
        "--horizons", type=int, nargs="+", default=None,
        help="One or more forecast horizons in minutes."
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--ff-dim", type=int, default=None)
    parser.add_argument(
        "--pooling", choices=["last", "mean", "attention"], default="attention"
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--parameter-budget-check",
        action="store_true",
        help="Print exact parameter-matched Transformer candidates and exit.",
    )
    parser.add_argument(
        "--output-dir", "--output", dest="output_dir",
        default="output/phase2_transformer_mgdl"
    )
    args = parser.parse_args()

    if args.horizons is not None:
        horizons = list(dict.fromkeys(args.horizons))
    elif args.horizon is not None:
        horizons = [args.horizon]
    else:
        horizons = [30]

    if any(h <= 0 or h % 5 != 0 for h in horizons):
        raise ValueError("All horizons must be positive multiples of 5 minutes.")

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    root = Path(args.data_root)
    train_dir = root / "train"
    test_dir = root / "test"

    train_files = sorted(train_dir.glob("*.csv"))
    if args.patient:
        train_files = [p for p in train_files if args.patient in p.stem]

    if not train_files:
        raise FileNotFoundError(f"No training CSVs found in {train_dir}")

    excluded = {
        "timestamp", "glucose", "gap_from_previous_min", "valid_5min_interval"
    }

    first = pd.read_csv(train_files[0])
    feature_columns = [c for c in first.columns if c not in excluded]
    feature_columns = ["glucose"] + feature_columns

    print("Features:", feature_columns)
    print("Patients:", [p.stem for p in train_files])
    print("Horizons:", horizons)

    if args.parameter_budget_check:
        print_parameter_budget(
            input_dim=len(feature_columns),
            pooling=args.pooling,
        )
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "transformer_results.csv"
    summary_path = out_dir / "transformer_patient_mean_sd.csv"
    mae_table_path = out_dir / "transformer_patient_mae_table.csv"
    rmse_table_path = out_dir / "transformer_patient_rmse_table.csv"
    config_path = out_dir / "transformer_run_config.json"

    if results_path.exists():
        result_df = pd.read_csv(results_path)
        print(f"Resuming from {results_path} ({len(result_df)} completed runs).")
    else:
        result_df = pd.DataFrame()

    run_config = vars(args).copy()
    run_config["horizons"] = horizons
    run_config["feature_columns"] = feature_columns
    run_config["device"] = str(device)
    run_config["num_patients"] = len(train_files)
    run_config["metric_scale"] = "original glucose scale (mg/dL)"
    run_config["scaler_fit_protocol"] = "per-patient; chronological pre-validation training rows only"
    config_path.write_text(json.dumps(run_config, indent=2))

    def is_completed(patient_id, horizon):
        if result_df.empty:
            return False
        mask = (
            result_df["patient"].astype(str).eq(str(patient_id))
            & result_df["horizon_min"].eq(horizon)
            & result_df["lookback_min"].eq(args.lookback)
            & result_df["d_model"].eq(args.d_model)
            & result_df["heads"].eq(args.heads)
            & result_df["layers"].eq(args.layers)
            & result_df["pooling"].eq(args.pooling)
        )
        return bool(mask.any())

    def save_outputs(df):
        if df.empty:
            return
        df.to_csv(results_path, index=False)

        summary = (
            df.groupby(["lookback_min", "horizon_min"])[["MAE", "RMSE"]]
            .agg(["mean", "std"]).reset_index()
        )
        summary.to_csv(summary_path, index=False)

        df.pivot_table(
            index="patient", columns="horizon_min", values="MAE", aggfunc="first"
        ).to_csv(mae_table_path)

        df.pivot_table(
            index="patient", columns="horizon_min", values="RMSE", aggfunc="first"
        ).to_csv(rmse_table_path)

    patient_data = {}
    for train_path in train_files:
        patient_id = train_path.stem
        test_path = test_dir / train_path.name

        if not test_path.exists():
            print(f"Skipping {patient_id}: matching test file not found.")
            continue

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
        test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])

        # Fit per-patient preprocessing only on the chronological pre-validation rows.
        fit_end = max(1, int(len(train_df) * (1.0 - args.val_fraction)))
        mean, std = fit_standardizer(train_df.iloc[:fit_end], feature_columns)
        train_df = apply_standardizer(train_df, feature_columns, mean, std)
        test_df = apply_standardizer(test_df, feature_columns, mean, std)
        patient_data[patient_id] = (train_df, test_df, mean.copy(), std.copy())

    for patient_index, train_path in enumerate(train_files, start=1):
        patient_id = train_path.stem
        if patient_id not in patient_data:
            continue

        train_df, test_df, patient_mean, patient_std = patient_data[patient_id]

        for horizon in horizons:
            if is_completed(patient_id, horizon):
                print(
                    f"[{patient_index}/{len(train_files)}] "
                    f"Patient {patient_id} | horizon {horizon} min | "
                    "already completed — skipping."
                )
                continue

            print(
                f"\n{'=' * 72}\n"
                f"[{patient_index}/{len(train_files)}] Patient {patient_id} | "
                f"horizon={horizon} min | lookback={args.lookback} min\n"
                f"{'=' * 72}"
            )

            config = SequenceConfig(
                lookback_minutes=args.lookback,
                horizon_minutes=horizon,
            )

            full_dataset = CGMSequenceDataset(
                train_df, feature_columns, config
            )

            if len(full_dataset) < 2:
                print("Too few training sequences; skipping.")
                continue

            n_val = max(1, int(args.val_fraction * len(full_dataset)))
            purge_steps = max(1, config.horizon_steps)
            val_start = len(full_dataset) - n_val
            train_end = max(1, val_start - purge_steps)

            if train_end < 1 or val_start >= len(full_dataset):
                print("Insufficient sequences for temporal validation; skipping.")
                continue

            train_ds = Subset(full_dataset, range(0, train_end))
            val_ds = Subset(full_dataset, range(val_start, len(full_dataset)))

            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers
            )
            val_loader = DataLoader(
                val_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers
            )

            # Keep the constructor aligned with the existing
            # ImprovedTransformer implementation used by the pilot.
            # Do not pass optional constructor arguments that this
            # implementation does not expose.
            model = ImprovedTransformer(
                input_dim=len(feature_columns),
                d_model=args.d_model,
                nhead=args.heads,
                num_layers=args.layers,
                pooling=args.pooling,
                num_outputs=1,
            ).to(device)

            ff_dim = args.ff_dim if args.ff_dim is not None else None

            parameter_count = sum(p.numel() for p in model.parameters())
            print(
                f"Sequences={len(full_dataset)} | train={len(train_ds)} | "
                f"val={len(val_ds)} | parameters={parameter_count:,}"
            )

            train_one(
                model, train_loader, val_loader, device,
                epochs=args.epochs, lr=args.lr,
                weight_decay=args.weight_decay, patience=args.patience
            )

            test_dataset = CGMSequenceDataset(
                test_df, feature_columns, config
            )
            if len(test_dataset) == 0:
                print("No test sequences; skipping.")
                continue

            test_loader = DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers
            )

            metrics = evaluate(
                model,
                test_loader,
                device,
                glucose_mean=patient_mean["glucose"],
                glucose_std=patient_std["glucose"],
            )
            metrics.update({
                "patient": patient_id,
                "lookback_min": args.lookback,
                "horizon_min": horizon,
                "d_model": args.d_model,
                "heads": args.heads,
                "layers": args.layers,
                "pooling": args.pooling,
                "ff_dim": ff_dim,
                "parameters": parameter_count,
                "train_sequences": len(train_ds),
                "val_sequences": len(val_ds),
                "test_sequences": len(test_dataset),
            })

            print("Test:", metrics)

            result_df = pd.concat(
                [result_df, pd.DataFrame([metrics])], ignore_index=True
            )

            # Critical for a long 30-run experiment: save immediately.
            save_outputs(result_df)
            print(f"Saved: {results_path}")

    if not result_df.empty:
        save_outputs(result_df)
        print("\nFinal saved outputs:")
        print(results_path)
        print(summary_path)
        print(mae_table_path)
        print(rmse_table_path)
        print(config_path)

        summary_display = (
            result_df.groupby(["lookback_min", "horizon_min"])[["MAE", "RMSE"]]
            .agg(["mean", "std"])
        )
        print("\nMean ± SD:")
        print(summary_display.to_string())
    else:
        print("No completed test results.")


if __name__ == "__main__":
    main()
