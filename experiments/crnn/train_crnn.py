"""Train the CRNN reproduction on OhioT1DM.

The predefined testing partition is held out until final evaluation.
Validation uses the latest portion of the training partition.
"""

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

from experiments.crnn.crnn import CRNN
from experiments.crnn.crnn_data import load_split, build_sequences

PATIENTS = [
    "540", "544", "552", "559", "563", "567",
    "570", "575", "584", "588", "591", "596",
]

OUTPUT_ROOT = Path("output/crnn_initial")
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prepare_sequences(patient, split, horizon):
    df = load_split(patient, split)
    X, delta_y, metadata = build_sequences(df, horizon)

    if len(X) == 0:
        raise ValueError(
            f"No usable sequences for patient {patient}, "
            f"split={split}, horizon={horizon}"
        )

    # Absolute target glucose is retained for evaluation.
    y_absolute = metadata["target_glucose"].to_numpy(dtype=np.float32)
    current_glucose = metadata["current_glucose"].to_numpy(dtype=np.float32)

    return X, delta_y, y_absolute, current_glucose, metadata



def prepare_chronological_train_validation(patient, horizon, fraction=0.15):
    """Split the original training timeline before building sequences."""
    from experiments.crnn.crnn_data import add_segments

    df = load_split(patient, "training").reset_index(drop=True)

    if len(df) < 100:
        raise ValueError(f"Too few rows for patient {patient}")

    # Split at a row boundary in chronological order.
    split_at = int(len(df) * (1 - fraction))
    train_df = df.iloc[:split_at].copy()
    val_df = df.iloc[split_at:].copy()

    # Build each side independently so no sequence crosses the boundary.
    X_train, dy_train, train_meta = build_sequences(train_df, horizon)
    X_val, dy_val, val_meta = build_sequences(val_df, horizon)

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError(
            f"Empty train/validation sequences for patient {patient}: "
            f"train={len(X_train)}, val={len(X_val)}"
        )

    # Additional safeguard: no training target may reach validation time.
    validation_start = val_df["Timestamp"].min()
    if (pd.to_datetime(train_meta["target_timestamp"]) >= validation_start).any():
        raise RuntimeError("Training targets overlap the validation period")

    return X_train, dy_train, X_val, dy_val

def chronological_validation(X, delta_y, fraction=0.15):
    """Split already-chronological sequences without shuffling."""
    split_at = int(len(X) * (1 - fraction))

    if split_at < 1 or split_at >= len(X):
        raise ValueError("Not enough sequences for chronological validation")

    return (
        X[:split_at],
        delta_y[:split_at],
        X[split_at:],
        delta_y[split_at:],
    )


def make_loader(X, y, batch_size, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(X, dtype=np.float32)),
        torch.from_numpy(np.asarray(y, dtype=np.float32)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_count = 0

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            if training:
                optimizer.zero_grad()

            predictions = model(X)
            loss = loss_fn(predictions, y)

            if training:
                loss.backward()
                optimizer.step()

            count = len(X)
            total_loss += loss.item() * count
            total_count += count

    return total_loss / max(total_count, 1)


def evaluate_absolute_glucose(model, X, current_glucose, y_absolute, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for start in range(0, len(X), 512):
            batch = torch.from_numpy(
                np.asarray(X[start:start + 512], dtype=np.float32)
            ).to(device)
            predicted_delta = model(batch).cpu().numpy().reshape(-1)
            predictions.extend(predicted_delta.tolist())

    predicted_glucose = current_glucose + np.asarray(predictions)
    errors = predicted_glucose - y_absolute

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # MARD is calculated only where reference glucose is positive.
    valid = y_absolute > 0
    mard = float(
        np.mean(np.abs(errors[valid] / y_absolute[valid])) * 100
    ) if valid.any() else float("nan")

    return {
        "n": int(len(y_absolute)),
        "MAE_mg_dL": mae,
        "RMSE_mg_dL": rmse,
        "MARD_percent": mard,
    }


def train_patient(patient, horizon, args, device):
    # Reset randomness for reproducible, patient-specific training runs.
    set_seed(SEED)

    print(f"\n{'=' * 64}")
    print(f"Patient {patient} | Horizon {horizon} minutes")
    print(f"{'=' * 64}")

    X_train, dy_train, X_val, dy_val = (
        prepare_chronological_train_validation(
            patient,
            horizon,
            fraction=args.validation_fraction,
        )
    )

    print(
        f"Train sequences: {len(X_train):,} | "
        f"Validation sequences: {len(X_val):,}"
    )

    train_loader = make_loader(
        X_train, dy_train, args.batch_size, shuffle=True
    )
    val_loader = make_loader(
        X_val, dy_val, args.batch_size, shuffle=False
    )

    model = CRNN(n_features=X_train.shape[-1], dropout=args.dropout).to(device)
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate)

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    patience_left = args.patience
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model, train_loader, loss_fn, device, optimizer=optimizer
        )
        val_loss = run_epoch(
            model, val_loader, loss_fn, device, optimizer=None
        )

        history.append({
            "patient": patient,
            "horizon_minutes": horizon,
            "epoch": epoch,
            "train_MAE_delta": train_loss,
            "validation_MAE_delta": val_loss,
        })

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train MAE(delta)={train_loss:.4f} | "
            f"val MAE(delta)={val_loss:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_left = args.patience
        else:
            patience_left -= 1

        if patience_left <= 0:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")

    model.load_state_dict(best_state)

    out_dir = OUTPUT_ROOT / f"patient_{patient}" / f"horizon_{horizon}min"
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = out_dir / "best_model.pt"
    torch.save({
        "patient": patient,
        "horizon_minutes": horizon,
        "model_state_dict": model.state_dict(),
        "n_features": int(X_train.shape[-1]),
        "dropout": args.dropout,
        "best_epoch": best_epoch,
        "best_validation_MAE_delta": best_val,
        "seed": SEED,
    }, checkpoint_path)

    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    # Test partition is accessed only after model selection is complete.
    X_test, _, y_test, current_test, test_meta = prepare_sequences(
        patient, "testing", horizon
    )

    metrics = evaluate_absolute_glucose(
        model, X_test, current_test, y_test, device
    )
    metrics.update({
        "patient": patient,
        "horizon_minutes": horizon,
        "best_epoch": best_epoch,
        "best_validation_MAE_delta": best_val,
        "checkpoint": str(checkpoint_path),
        "n_train": int(len(X_train)),
        "n_validation": int(len(X_val)),
    })

    # Save row-level predictions for later clinical analyses.
    model.eval()
    predicted_deltas = []

    with torch.no_grad():
        for start in range(0, len(X_test), 512):
            batch = torch.from_numpy(
                X_test[start:start + 512].astype(np.float32)
            ).to(device)
            predicted_deltas.extend(
                model(batch).cpu().numpy().reshape(-1).tolist()
            )

    predictions = test_meta.copy()
    predictions["predicted_delta"] = predicted_deltas
    predictions["predicted_glucose"] = (
        predictions["current_glucose"] + predictions["predicted_delta"]
    )
    predictions["error_mg_dL"] = (
        predictions["predicted_glucose"] - predictions["target_glucose"]
    )
    predictions.to_csv(out_dir / "test_predictions.csv", index=False)

    print(
        f"TEST | n={metrics['n']:,} | "
        f"MAE={metrics['MAE_mg_dL']:.3f} mg/dL | "
        f"RMSE={metrics['RMSE_mg_dL']:.3f} mg/dL | "
        f"MARD={metrics['MARD_percent']:.3f}%"
    )

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=30, choices=[30, 60])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument(
        "--patients", nargs="*", default=PATIENTS,
        help="Optional patient IDs for a small smoke run, e.g. --patients 540",
    )
    args = parser.parse_args()

    invalid = sorted(set(args.patients) - set(PATIENTS))
    if invalid:
        raise ValueError(f"Unknown patient IDs: {invalid}")

    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be between 0 and 0.5")

    set_seed(SEED)
    device = choose_device()

    print("CRNN OhioT1DM experiment")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")
    print(f"Horizon: {args.horizon} minutes")
    print(f"Patients: {', '.join(args.patients)}")
    print(f"Epochs: {args.epochs}; patience: {args.patience}")
    print(f"Validation fraction: {args.validation_fraction:.0%}")
    print(f"Output root: {OUTPUT_ROOT.resolve()}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    results = []
    for patient in args.patients:
        results.append(train_patient(patient, args.horizon, args, device))

    results_df = pd.DataFrame(results)
    summary_path = OUTPUT_ROOT / f"crnn_{args.horizon}min_test_metrics.csv"
    results_df.to_csv(summary_path, index=False)

    summary = {
        "horizon_minutes": args.horizon,
        "patients_completed": len(results),
        "mean_patient_MAE_mg_dL": float(results_df["MAE_mg_dL"].mean()),
        "sd_patient_MAE_mg_dL": float(results_df["MAE_mg_dL"].std(ddof=1)),
        "mean_patient_RMSE_mg_dL": float(results_df["RMSE_mg_dL"].mean()),
        "sd_patient_RMSE_mg_dL": float(results_df["RMSE_mg_dL"].std(ddof=1)),
        "mean_patient_MARD_percent": float(results_df["MARD_percent"].mean()),
        "sd_patient_MARD_percent": float(results_df["MARD_percent"].std(ddof=1)),
    }

    with open(OUTPUT_ROOT / f"crnn_{args.horizon}min_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Patient-level summary ===")
    print(results_df.to_string(index=False))
    print("\n=== Mean ± SD across patients ===")
    for metric in [
        "MAE_mg_dL", "RMSE_mg_dL", "MARD_percent"
    ]:
        print(
            f"{metric}: {results_df[metric].mean():.3f} ± "
            f"{results_df[metric].std(ddof=1):.3f}"
        )
    print(f"\nSaved metrics: {summary_path}")


if __name__ == "__main__":
    main()
