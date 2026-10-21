"""Experiment B: GluNet with square-root inverse-frequency class weights.

This is a separate experiment. It does not modify the original baseline.

Patient: 540
Horizons: 30 and 60 minutes
Training/validation: chronological split of the predefined training partition
Testing: predefined testing partition, held out until evaluation

Class weights are calculated exclusively from the training labels.
Absent classes receive zero target weight.
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
    build_sequences,
    load_split,
)
from experiments.glunet.glunet_model import GluNet


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

PATIENT = "540"
HORIZONS = [30, 60]

SEED = 42
VALIDATION_FRACTION = 0.20

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
MAX_EPOCHS = 30
PATIENCE = 5

OUTPUT_DIR = Path(
    "experiments/glunet/runs/pilot_patient540_weighted"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(X, y, shuffle=False):
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
        raise RuntimeError(
            "Sequence, label, and metadata lengths differ."
        )

    if not np.isfinite(X).all():
        raise RuntimeError(
            "Input sequences contain non-finite values."
        )

    return X, y, metadata


def make_class_weights(y):
    """Compute square-root inverse-frequency weights.

    Only classes represented in y receive nonzero weights.
    The weights are normalized to have sample-weighted mean 1.
    """
    counts = np.bincount(y, minlength=N_CLASSES).astype(np.float64)
    present = counts > 0

    weights = np.zeros(N_CLASSES, dtype=np.float64)
    weights[present] = 1.0 / np.sqrt(counts[present])

    # Normalize so that sum(count[c] * weight[c]) / N == 1.
    weights /= np.sum(counts * weights) / len(y)

    if not np.isfinite(weights).all():
        raise RuntimeError("Non-finite class weights generated.")

    if np.any(weights < 0):
        raise RuntimeError("Negative class weights generated.")

    if np.any(weights[~present] != 0):
        raise RuntimeError("Absent classes must have zero weight.")

    print("\nClass-weight audit")
    print(f"  Training samples: {len(y)}")
    print(f"  Represented classes: {present.sum()}")
    print(f"  Absent classes: {(~present).sum()}")
    print(f"  Minimum nonzero weight: {weights[present].min():.4f}")
    print(f"  Maximum nonzero weight: {weights[present].max():.4f}")
    print(
        "  Sample-weighted mean: "
        f"{np.sum(counts * weights) / len(y):.4f}"
    )

    print("  Class counts and weights:")
    for cls in np.flatnonzero(present):
        print(
            f"    class={cls:3d} "
            f"count={int(counts[cls]):5d} "
            f"weight={weights[cls]:.4f}"
        )

    return torch.tensor(weights, dtype=torch.float32)


def calculate_loss(model, loader, criterion):
    """Return sample-weighted mean loss for a data loader."""
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


def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    horizon,
):
    class_weights = make_class_weights(y_train).to(DEVICE)

    model = GluNet(
        input_features=len(FEATURES),
        n_classes=N_CLASSES,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    train_loader = make_loader(
        X_train,
        y_train,
        shuffle=True,
    )
    val_loader = make_loader(
        X_val,
        y_val,
        shuffle=False,
    )

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

        # Select checkpoints using weighted validation cross-entropy.
        val_loss = calculate_loss(
            model,
            val_loader,
            criterion,
        )

        history.append({
            "epoch": epoch,
            "train_weighted_cross_entropy": train_loss,
            "validation_weighted_cross_entropy": val_loss,
        })

        print(
            f"  Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"train weighted CE={train_loss:.5f} | "
            f"validation weighted CE={val_loss:.5f}"
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
        raise RuntimeError(
            "Training did not produce a valid checkpoint."
        )

    model.load_state_dict(best_state)

    history_path = (
        OUTPUT_DIR / f"patient{PATIENT}_{horizon}min_history.csv"
    )
    pd.DataFrame(history).to_csv(
        history_path,
        index=False,
    )

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
        "best_validation_weighted_loss": best_val_loss,
        "class_weights": class_weights.cpu(),
        "model_state_dict": best_state,
        "model_config": {
            "input_features": len(FEATURES),
            "channels": 32,
            "n_classes": N_CLASSES,
        },
        "training_method": "sqrt_inverse_frequency_weighted_cross_entropy",
    }, checkpoint_path)

    print(f"  Best epoch: {best_epoch}")
    print(f"  Best validation weighted CE: {best_val_loss:.5f}")
    print(f"  Checkpoint: {checkpoint_path}")

    return model, best_epoch, best_val_loss


def evaluate_test(model, X, y, metadata, horizon):
    """Evaluate the held-out test split using unweighted metrics."""
    model.eval()

    loader = make_loader(X, y, shuffle=False)

    all_logits = []
    all_probabilities = []

    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(DEVICE)

            logits = model(X_batch)
            probabilities = torch.softmax(logits, dim=1)

            all_logits.append(logits.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    probabilities = np.concatenate(all_probabilities, axis=0)

    predicted_classes = logits.argmax(axis=1)

    class_values = (
        np.arange(N_CLASSES, dtype=np.float64) + 1.0
    ) / 10.0 - 12.8

    # Preserve the same expected-value decoder as the baseline.
    predicted_deltas = probabilities @ class_values

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

    persistence_errors = current_glucose - actual_glucose
    persistence_absolute_errors = np.abs(persistence_errors)

    # Use unweighted CE for comparison with the original baseline.
    test_ce = nn.functional.cross_entropy(
        torch.from_numpy(logits.astype(np.float32)),
        torch.from_numpy(true_classes),
    ).item()

    outside_nominal_range = (
        (true_deltas < -12.75)
        | (true_deltas > 12.85)
    )

    positive_actual = actual_glucose > 0

    metrics = {
        "patient": PATIENT,
        "split": "testing",
        "horizon_minutes": horizon,
        "n_sequences": int(len(y)),
        "unweighted_cross_entropy": float(test_ce),
        "class_accuracy": float(
            np.mean(predicted_classes == true_classes)
        ),
        "predicted_unique_classes": int(
            np.unique(predicted_classes).size
        ),
        "predicted_class_0_percent": float(
            100.0 * np.mean(predicted_classes == 0)
        ),
        "predicted_class_255_percent": float(
            100.0 * np.mean(predicted_classes == 255)
        ),
        "mean_maximum_probability": float(
            probabilities.max(axis=1).mean()
        ),
        "glucose_MAE_mg_dL": float(
            absolute_errors.mean()
        ),
        "glucose_RMSE_mg_dL": float(
            np.sqrt(np.mean(errors ** 2))
        ),
        "glucose_MARD_percent": (
            float(
                100.0 * np.mean(
                    absolute_errors[positive_actual]
                    / actual_glucose[positive_actual]
                )
            )
            if np.any(positive_actual)
            else None
        ),
        "mean_signed_error_mg_dL": float(errors.mean()),
        "persistence_MAE_mg_dL": float(
            persistence_absolute_errors.mean()
        ),
        "persistence_RMSE_mg_dL": float(
            np.sqrt(np.mean(persistence_errors ** 2))
        ),
        "mean_absolute_delta_error_mg_dL": float(
            np.mean(np.abs(predicted_deltas - true_deltas))
        ),
        "true_delta_outside_nominal_range_percent": float(
            100.0 * outside_nominal_range.mean()
        ),
        "predicted_delta_min_mg_dL": float(
            predicted_deltas.min()
        ),
        "predicted_delta_max_mg_dL": float(
            predicted_deltas.max()
        ),
    }

    predictions = metadata.copy()
    predictions["true_class"] = true_classes
    predictions["predicted_class"] = predicted_classes
    predictions["predicted_delta_mg_dL"] = predicted_deltas
    predictions["predicted_glucose_mg_dL"] = predicted_glucose
    predictions["persistence_glucose_mg_dL"] = current_glucose
    predictions["prediction_error_mg_dL"] = errors

    return metrics, predictions


def run_horizon(horizon):
    print("\n" + "=" * 72)
    print(f"WEIGHTED GLUNET | PATIENT {PATIENT} | {horizon} MINUTES")
    print("=" * 72)

    training_df = load_split(PATIENT, "training")
    testing_df = load_split(PATIENT, "testing")

    # Same chronological split strategy as the baseline.
    split_index = int(
        len(training_df) * (1.0 - VALIDATION_FRACTION)
    )

    train_df = training_df.iloc[:split_index].copy()
    val_df = training_df.iloc[split_index:].copy()

    X_train, y_train, _ = prepare_sequences(
        train_df,
        horizon,
    )
    X_val, y_val, _ = prepare_sequences(
        val_df,
        horizon,
    )
    X_test, y_test, metadata_test = prepare_sequences(
        testing_df,
        horizon,
    )

    print(f"Device: {DEVICE}")
    print(f"Training sequences: {len(y_train)}")
    print(f"Validation sequences: {len(y_val)}")
    print(f"Testing sequences: {len(y_test)}")

    set_seed(SEED)

    model, best_epoch, best_val_loss = train_model(
        X_train,
        y_train,
        X_val,
        y_val,
        horizon,
    )

    metrics, predictions = evaluate_test(
        model,
        X_test,
        y_test,
        metadata_test,
        horizon,
    )

    metrics["best_epoch"] = int(best_epoch)
    metrics["best_validation_weighted_cross_entropy"] = float(
        best_val_loss
    )

    metrics_path = (
        OUTPUT_DIR / f"patient{PATIENT}_{horizon}min_test_metrics.json"
    )
    predictions_path = (
        OUTPUT_DIR / f"patient{PATIENT}_{horizon}min_test_predictions.csv"
    )

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    print("\nHELD-OUT TEST RESULTS")
    print("-" * 48)

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print(f"\nMetrics saved: {metrics_path}")
    print(f"Predictions saved: {predictions_path}")

    return metrics


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "experiment": "GluNet weighted cross-entropy",
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
        "class_weighting": "normalized square-root inverse frequency",
        "checkpoint_selection": "weighted validation cross-entropy",
        "test_cross_entropy": "unweighted",
        "device": str(DEVICE),
        "note": (
            "Separate experimental variant; original baseline is preserved."
        ),
    }

    with (
        OUTPUT_DIR / "experiment_config.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    results = []

    for horizon in HORIZONS:
        results.append(run_horizon(horizon))

    summary_path = OUTPUT_DIR / "weighted_pilot_summary.csv"

    pd.DataFrame(results).to_csv(
        summary_path,
        index=False,
    )

    print("\n" + "=" * 72)
    print("WEIGHTED EXPERIMENT COMPLETE")
    print(f"Summary: {summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
