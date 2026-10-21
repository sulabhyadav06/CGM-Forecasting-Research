"""Single-patient continuous-regression diagnostic on candidate-classifier windows.

Experimental diagnostic only; not an exact reproduction of published GluNet.
Uses the candidate classifier's partitioning, five-feature preprocessing, and
window construction so its test timestamps can be compared with the classifier.
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
from torch.utils.data import DataLoader, TensorDataset

from experiments.glunet.glunet_regression_model import GluNetRegressor
from experiments.glunet.train_glunet_candidate_all_patients import (
    apply_preprocessor,
    fit_preprocessor,
    load_partition,
    make_sequences,
)
from clinical_metrics import full_metrics_row


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(X, y=None, batch_size=128, shuffle=False, device=None):
    tensors = [torch.from_numpy(np.asarray(X, dtype=np.float32))]
    if y is not None:
        tensors.append(torch.from_numpy(np.asarray(y, dtype=np.float32)))
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device is not None and device.type == "cuda"),
    )


def predict_delta(model, X, batch_size, device):
    model.eval()
    parts = []
    with torch.no_grad():
        for (xb,) in make_loader(
            X, batch_size=batch_size, device=device
        ):
            parts.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="~/Desktop/OhioT1DM_data_backup")
    p.add_argument(
        "--output-dir",
        default="experiments/glunet/runs/regression_diagnostic_540_h30",
    )
    p.add_argument("--patient", default="540")
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--validation-fraction", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    if args.horizon <= 0 or args.horizon % 5:
        p.error("--horizon must be a positive multiple of 5")
    if not 0.05 <= args.validation_fraction <= 0.4:
        p.error("--validation-fraction must be between 0.05 and 0.4")

    set_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / pd.Timestamp.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=False, exist_ok=False)

    train_all = load_partition(root, args.patient, "training")
    test_df = load_partition(root, args.patient, "testing")
    cut = int(len(train_all) * (1.0 - args.validation_fraction))
    train_df = train_all.iloc[:cut].copy()
    val_df = train_all.iloc[cut:].copy()

    prep = fit_preprocessor(train_df)
    train_df, val_df, test_df = [
        apply_preprocessor(frame, prep)
        for frame in (train_df, val_df, test_df)
    ]
    (run_dir / "preprocessing.json").write_text(
        json.dumps(prep, indent=2)
    )

    Xtr, _, mtr = make_sequences(train_df, args.horizon)
    Xv, _, mv = make_sequences(val_df, args.horizon)
    Xte, _, mte = make_sequences(test_df, args.horizon)

    if min(len(Xtr), len(Xv), len(Xte)) == 0:
        raise RuntimeError(
            f"Empty windows: train={len(Xtr)}, "
            f"val={len(Xv)}, test={len(Xte)}"
        )

    # Use unclipped continuous glucose changes, not the 256 classes.
    ytr = mtr["delta_glucose"].to_numpy(dtype=np.float32)
    yv = mv["delta_glucose"].to_numpy(dtype=np.float32)
    yte = mte["delta_glucose"].to_numpy(dtype=np.float32)

    if not all(
        np.isfinite(x).all()
        for x in (Xtr, Xv, Xte, ytr, yv, yte)
    ):
        raise RuntimeError("Non-finite feature or target values found")

    model = GluNetRegressor(
        input_features=Xtr.shape[-1], channels=32
    ).to(device)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = make_loader(
        Xtr, ytr, batch_size=args.batch_size,
        shuffle=True, device=device
    )

    best_val_mae = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        weighted_loss = 0.0
        n_seen = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            weighted_loss += float(loss.item()) * len(yb)
            n_seen += len(yb)

        val_pred = predict_delta(model, Xv, args.batch_size, device)
        val_err = val_pred - yv
        val_mae = float(np.mean(np.abs(val_err)))
        val_rmse = float(np.sqrt(np.mean(val_err ** 2)))

        row = {
            "epoch": epoch,
            "train_huber_loss": weighted_loss / max(n_seen, 1),
            "validation_delta_mae_mg_dL": val_mae,
            "validation_delta_rmse_mg_dL": val_rmse,
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train Huber={row['train_huber_loss']:.4f} | "
            f"val delta MAE={val_mae:.3f} | "
            f"val delta RMSE={val_rmse:.3f}",
            flush=True,
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1

        if stale >= args.patience:
            print(f"Early stopping at epoch {epoch}.", flush=True)
            break

    if best_state is None:
        raise RuntimeError("No best model checkpoint was produced")

    model.load_state_dict(best_state)
    torch.save(
        {
            "state_dict": best_state,
            "patient": args.patient,
            "horizon_minutes": args.horizon,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "best_validation_delta_mae_mg_dL": best_val_mae,
            "features": list(prep["medians"].keys()),
            "status": (
                "experimental regression diagnostic; "
                "not exact GluNet reproduction"
            ),
        },
        run_dir / "best.pt",
    )
    pd.DataFrame(history).to_csv(
        run_dir / "training_history.csv", index=False
    )

    predicted_delta = predict_delta(
        model, Xte, args.batch_size, device
    )
    current = mte["current_glucose"].to_numpy(dtype=float)
    actual = mte["target_glucose"].to_numpy(dtype=float)
    forecast = current + predicted_delta
    persistence = current

    metric_rows = []
    for name, pred_glucose in (
        ("huber_regression", forecast),
        ("persistence", persistence),
    ):
        row = {
            "patient": args.patient,
            "horizon_minutes": args.horizon,
            "model": name,
            "n_test_windows": len(actual),
            "best_epoch": best_epoch if name == "huber_regression" else None,
            "best_validation_delta_mae_mg_dL": (
                best_val_mae if name == "huber_regression" else None
            ),
        }
        row.update(
            full_metrics_row(
                actual, pred_glucose, sample_interval_min=5
            )
        )
        metric_rows.append(row)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(run_dir / "test_metrics.csv", index=False)

    pred_df = mte.copy()
    pred_df["actual_delta_mg_dL"] = yte
    pred_df["predicted_delta_mg_dL"] = predicted_delta
    pred_df["regression_forecast_glucose_mg_dL"] = forecast
    pred_df["persistence_forecast_glucose_mg_dL"] = persistence
    pred_df["regression_error_mg_dL"] = forecast - actual
    pred_df["persistence_error_mg_dL"] = persistence - actual
    pred_df.to_csv(run_dir / "test_predictions.csv", index=False)

    config = vars(args).copy()
    config.update(
        {
            "data_root": str(root),
            "run_dir": str(run_dir),
            "device": str(device),
            "features": list(prep["medians"].keys()),
            "lookback": int(Xtr.shape[1]),
            "n_train_windows": int(len(Xtr)),
            "n_validation_windows": int(len(Xv)),
            "n_test_windows": int(len(Xte)),
            "best_epoch": int(best_epoch),
            "best_validation_delta_mae_mg_dL": float(best_val_mae),
            "target": (
                "unclipped future glucose minus current glucose, mg/dL"
            ),
            "loss": "HuberLoss(delta=1.0)",
            "status": (
                "experimental diagnostic; "
                "not exact reproduction of published GluNet"
            ),
        }
    )
    (run_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2)
    )

    print("\n=== TEST METRICS ===")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved run to: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
