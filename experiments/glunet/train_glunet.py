"""Patient-level GluNet pilot training and evaluation.

Uses existing OhioT1DM training/testing partitions.
Validation is carved chronologically from the training partition.
The testing partition is held out until final evaluation.

This is a GluNet-inspired experiment, not a claim of exact paper reproduction.
"""

import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.glunet.glunet_data import (
    FEATURES,
    LOOKBACK,
    N_CLASSES,
    TARGET,
    build_sequences,
    load_split,
)
from experiments.glunet.glunet_model import GluNet


# ------------------------------------------------------------
# Experiment configuration
# ------------------------------------------------------------

PATIENT = "540"
HORIZONS = [30, 60]

SEED = 42
VALIDATION_FRACTION = 0.20

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
MAX_EPOCHS = 30
PATIENCE = 5

OUTPUT_DIR = Path("experiments/glunet/runs/pilot_patient540")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_sequences(df, horizon):
    X, y, metadata = build_sequences(
        df,
        horizon_minutes=horizon,
    )

    if len(X) == 0:
        raise RuntimeError(
            f"No sequences generated for horizon={horizon}."
        )

    if not (len(X) == len(y) == len(metadata)):
        raise RuntimeError("Sequence and metadata lengths differ.")

    if not np.isfinite(X).all():
        raise RuntimeError("Input sequences contain non-finite values.")

    if not np.isfinite(metadata["target_glucose"].to_numpy()).all():
        raise RuntimeError("Target glucose contains non-finite values.")

    return X, y, metadata


def make_loader(X, y, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(X, dtype=np.float32)),
        torch.from_numpy(np.asarray(y, dtype=np.int64)),
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )


def evaluate_loss(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            count = len(y_batch)
            total_loss += loss.item() * count
            total_count += count

    return total_loss / total_count


def train_model(X_train, y_train, X_val, y_val, horizon):
    model = GluNet(
        input_features=len(FEATURES),
        n_classes=N_CLASSES,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader = make_loader(X_val, y_val, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        total_train_loss = 0.0
        total_train_count = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()

            count = len(y_batch)
            total_train_loss += loss.item() * count
            total_train_count += count

        train_loss = total_train_loss / total_train_count
        val_loss = evaluate_loss(model, val_loader, criterion)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
        })

        print(
            f"  Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"train CE={train_loss:.5f} | "
            f"validation CE={val_loss:.5f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)

    history_path = OUTPUT_DIR / f"patient{PATIENT}_{horizon}min_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    checkpoint_path = (
        OUTPUT_DIR / f"patient{PATIENT}_{horizon}min_best.pt"
    )

    torch.save({
        "patient": PATIENT,
        "horizon_minutes": horizon,
        "features": FEATURES,
        "lookback": LOOKBACK,
        "n_classes": N_CLASSES,
        "seed": SEED,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "model_state_dict": best_state,
        "model_config": {
            "input_features": len(FEATURES),
            "channels": 32,
            "n_classes": N_CLASSES,
        },
    }, checkpoint_path)

    print(f"  Best epoch: {best_epoch}")
    print(f"  Best validation CE: {best_val_loss:.5f}")
    print(f"  Checkpoint: {checkpoint_path}")

    return model, best_epoch, best_val_loss


def calculate_metrics(model, X, y, metadata, horizon):
    model.eval()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X)),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    predicted_classes = []
    predicted_deltas = []

    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch = X_batch.to(DEVICE)

            logits = model(X_batch)
            class_predictions = logits.argmax(dim=1)
            delta_predictions = model.predict_delta(X_batch)

            predicted_classes.extend(
                class_predictions.cpu().numpy().tolist()
            )
            predicted_deltas.extend(
                delta_predictions.cpu().numpy().tolist()
            )

    predicted_classes = np.asarray(predicted_classes, dtype=np.int64)
    predicted_deltas = np.asarray(predicted_deltas, dtype=np.float64)

    true_classes = np.asarray(y, dtype=np.int64)

    current_glucose = metadata["current_glucose"].to_numpy(
        dtype=np.float64
    )
    actual_glucose = metadata["target_glucose"].to_numpy(
        dtype=np.float64
    )
    true_deltas = metadata["delta_glucose"].to_numpy(
        dtype=np.float64
    )

    predicted_glucose = current_glucose + predicted_deltas

    errors = predicted_glucose - actual_glucose
    absolute_errors = np.abs(errors)

    persistence_predictions = current_glucose
    persistence_errors = persistence_predictions - actual_glucose
    persistence_absolute_errors = np.abs(persistence_errors)

    # These thresholds describe the unsaturated delta interval.
    # The strict inequalities match the documented audit convention.
    outside_nominal_range = (
        (true_deltas < -12.75) | (true_deltas > 12.85)
    )

    valid_mard = actual_glucose > 0

    metrics = {
        "patient": PATIENT,
        "split": "testing",
        "horizon_minutes": horizon,
        "n_sequences": int(len(y)),
        "cross_entropy": float(
            nn.functional.cross_entropy(
                torch.from_numpy(
                    np.asarray(
                        _get_logits_for_metrics(model, X),
                        dtype=np.float32,
                    )
                ),
                torch.from_numpy(true_classes),
            ).item()
        ),
        "class_accuracy": float(
            np.mean(predicted_classes == true_classes)
        ),
        "glucose_MAE_mg_dL": float(np.mean(absolute_errors)),
        "glucose_RMSE_mg_dL": float(
            np.sqrt(np.mean(errors ** 2))
        ),
        "glucose_MARD_percent": (
            float(
                100.0 * np.mean(
                    absolute_errors[valid_mard]
                    / actual_glucose[valid_mard]
                )
            )
            if np.any(valid_mard)
            else None
        ),
        "persistence_MAE_mg_dL": float(
            np.mean(persistence_absolute_errors)
        ),
        "persistence_RMSE_mg_dL": float(
            np.sqrt(np.mean(persistence_errors ** 2))
        ),
        "mean_absolute_delta_error_mg_dL": float(
            np.mean(np.abs(predicted_deltas - true_deltas))
        ),
        "true_delta_outside_nominal_range_percent": float(
            100.0 * np.mean(outside_nominal_range)
        ),
        "predicted_delta_min_mg_dL": float(np.min(predicted_deltas)),
        "predicted_delta_max_mg_dL": float(np.max(predicted_deltas)),
    }

    predictions = metadata.copy()
    predictions["true_class"] = true_classes
    predictions["predicted_class"] = predicted_classes
    predictions["predicted_delta_mg_dL"] = predicted_deltas
    predictions["predicted_glucose_mg_dL"] = predicted_glucose
    predictions["persistence_glucose_mg_dL"] = persistence_predictions
    predictions["prediction_error_mg_dL"] = errors

    return metrics, predictions


def _get_logits_for_metrics(model, X):
    """Collect logits in original sequence order for test cross-entropy."""
    model.eval()
    results = []

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X)),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    with torch.no_grad():
        for (X_batch,) in loader:
            results.append(
                model(X_batch.to(DEVICE)).cpu().numpy()
            )

    return np.concatenate(results, axis=0)


def run_horizon(horizon):
    print("\n" + "=" * 72)
    print(f"PATIENT {PATIENT} | HORIZON {horizon} MINUTES")
    print("=" * 72)

    training_df = load_split(PATIENT, "training")
    testing_df = load_split(PATIENT, "testing")

    # Split the source training partition chronologically, before
    # generating sequences. This prevents windows or targets from
    # crossing the train/validation boundary.
    split_index = int(len(training_df) * (1.0 - VALIDATION_FRACTION))

    train_df = training_df.iloc[:split_index].copy()
    val_df = training_df.iloc[split_index:].copy()

    print(f"Training source rows:   {len(train_df)}")
    print(f"Validation source rows: {len(val_df)}")
    print(f"Testing source rows:    {len(testing_df)}")
    print(f"Device: {DEVICE}")

    X_train, y_train, _ = prepare_sequences(train_df, horizon)
    X_val, y_val, _ = prepare_sequences(val_df, horizon)
    X_test, y_test, metadata_test = prepare_sequences(
        testing_df,
        horizon,
    )

    print(f"Training sequences:   {len(y_train)}")
    print(f"Validation sequences: {len(y_val)}")
    print(f"Testing sequences:    {len(y_test)}")

    if len(y_train) == 0 or len(y_val) == 0 or len(y_test) == 0:
        raise RuntimeError("A split generated no sequences.")

    set_seed(SEED)

    model, best_epoch, best_val_loss = train_model(
        X_train,
        y_train,
        X_val,
        y_val,
        horizon,
    )

    # The held-out testing split is evaluated only after model selection.
    metrics, predictions = calculate_metrics(
        model,
        X_test,
        y_test,
        metadata_test,
        horizon,
    )

    metrics["best_epoch"] = int(best_epoch)
    metrics["best_validation_cross_entropy"] = float(best_val_loss)

    metrics_path = OUTPUT_DIR / f"patient{PATIENT}_{horizon}min_test_metrics.json"
    predictions_path = (
        OUTPUT_DIR / f"patient{PATIENT}_{horizon}min_test_predictions.csv"
    )

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    predictions.to_csv(predictions_path, index=False)

    print("\nHELD-OUT TEST RESULTS")
    print("-" * 48)

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print(f"\nMetrics saved:     {metrics_path}")
    print(f"Predictions saved: {predictions_path}")

    return metrics


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "patient": PATIENT,
        "horizons_minutes": HORIZONS,
        "features": FEATURES,
        "lookback": LOOKBACK,
        "n_classes": N_CLASSES,
        "seed": SEED,
        "validation_fraction": VALIDATION_FRACTION,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "device": str(DEVICE),
        "note": (
            "GluNet-inspired pilot. Expected-value delta decoding; "
            "not an exact reproduction of every paper detail."
        ),
    }

    with (OUTPUT_DIR / "experiment_config.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(config, f, indent=2)

    all_metrics = []

    for horizon in HORIZONS:
        all_metrics.append(run_horizon(horizon))

    summary_path = OUTPUT_DIR / "pilot_summary.csv"
    pd.DataFrame(all_metrics).to_csv(summary_path, index=False)

    print("\n" + "=" * 72)
    print("PILOT COMPLETE")
    print(f"Summary: {summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
