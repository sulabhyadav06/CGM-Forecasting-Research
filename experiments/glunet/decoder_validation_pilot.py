"""Validation-only comparison of provisional GluNet decoders.

Trains the isolated candidate model on patient 540, horizon 30 min.
Uses chronological validation and never loads the held-out test partition.
This is an experimental comparison, not a claim of exact GluNet reproduction.
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
from experiments.glunet.glunet_model_candidate import GluNetCandidate


PATIENT = "540"
HORIZON = 30
SEED = 42
VALIDATION_FRACTION = 0.20
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
MAX_EPOCHS = 30
PATIENCE = 5

OUTPUT_DIR = Path(
    "experiments/glunet/runs/decoder_validation_pilot_candidate"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(X, y, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(X, dtype=np.float32)),
        torch.from_numpy(np.asarray(y, dtype=np.int64)),
    )
    generator = torch.Generator()
    generator.manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
    )


def validation_logits(model, X, y):
    model.eval()
    loader = make_loader(X, y, shuffle=False)
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(DEVICE))
            all_logits.append(logits.cpu())
            all_labels.append(yb)

    return torch.cat(all_logits), torch.cat(all_labels)


def glucose_metrics(name, prediction, actual, current):
    error = prediction - actual
    abs_error = np.abs(error)
    valid = actual > 0

    return {
        "decoder": name,
        "n_validation_sequences": int(len(actual)),
        "MAE_mg_dL": float(abs_error.mean()),
        "RMSE_mg_dL": float(np.sqrt(np.mean(error ** 2))),
        "MARD_percent": (
            float(100.0 * np.mean(abs_error[valid] / actual[valid]))
            if valid.any() else None
        ),
        "mean_signed_error_mg_dL": float(error.mean()),
        "persistence_MAE_mg_dL": float(np.mean(np.abs(current - actual))),
        "prediction_min_mg_dL": float(prediction.min()),
        "prediction_max_mg_dL": float(prediction.max()),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    # Deliberately load ONLY the predefined training partition.
    training_df = load_split(PATIENT, "training")
    split_index = int(len(training_df) * (1.0 - VALIDATION_FRACTION))
    train_df = training_df.iloc[:split_index].copy()
    val_df = training_df.iloc[split_index:].copy()

    X_train, y_train, train_meta = build_sequences(
        train_df, horizon_minutes=HORIZON
    )
    X_val, y_val, val_meta = build_sequences(
        val_df, horizon_minutes=HORIZON
    )

    if not len(y_train) or not len(y_val):
        raise RuntimeError("Training or validation split has no sequences.")
    if not np.isfinite(X_train).all() or not np.isfinite(X_val).all():
        raise RuntimeError("Non-finite input values found.")
    if not (len(X_val) == len(y_val) == len(val_meta)):
        raise RuntimeError("Validation sequence/label/metadata mismatch.")

    model = GluNetCandidate(
        input_features=len(FEATURES),
        lookback=LOOKBACK,
        n_classes=N_CLASSES,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader = make_loader(X_val, y_val, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    history = []

    print(f"Patient: {PATIENT} | Horizon: {HORIZON} min")
    print(f"Device: {DEVICE}")
    print(f"Features: {FEATURES}")
    print(f"Training source rows: {len(train_df)}")
    print(f"Validation source rows: {len(val_df)}")
    print(f"Training sequences: {len(y_train)}")
    print(f"Validation sequences: {len(y_val)}")
    print("Held-out testing partition: NOT LOADED")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_count = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            total_count += len(yb)

        train_loss = total_loss / total_count
        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb.to(DEVICE))
                loss = criterion(logits, yb.to(DEVICE))
                val_loss_sum += loss.item() * len(yb)
                val_count += len(yb)

        val_loss = val_loss_sum / val_count
        history.append({
            "epoch": epoch,
            "training_cross_entropy": train_loss,
            "validation_cross_entropy": val_loss,
        })
        print(
            f"Epoch {epoch:02d}: train CE={train_loss:.5f}, "
            f"validation CE={val_loss:.5f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("No valid checkpoint was selected.")

    model.load_state_dict(best_state)
    model.eval()

    # Both decoders use the exact same selected checkpoint and validation X.
    X_tensor = torch.from_numpy(np.asarray(X_val, dtype=np.float32))
    all_logits = []
    with torch.no_grad():
        for start in range(0, len(X_tensor), BATCH_SIZE):
            xb = X_tensor[start:start + BATCH_SIZE].to(DEVICE)
            all_logits.append(model(xb).cpu())
    logits = torch.cat(all_logits, dim=0)

    probabilities = torch.softmax(logits, dim=-1)
    class_ids = logits.argmax(dim=-1).numpy()
    class_values = (
        torch.arange(N_CLASSES, dtype=probabilities.dtype) + 1
    ) / 10.0 - 12.8

    # Provisional decoder A: recover the most probable class.
    delta_argmax = (class_ids.astype(np.float64) + 1) / 10.0 - 12.8

    # Provisional decoder B: expected value across the class distribution.
    delta_expected = (
        probabilities * class_values.unsqueeze(0)
    ).sum(dim=-1).numpy()

    current = val_meta["current_glucose"].to_numpy(dtype=np.float64)
    actual = val_meta["target_glucose"].to_numpy(dtype=np.float64)
    true_classes = np.asarray(y_val, dtype=np.int64)

    glucose_argmax = current + delta_argmax
    glucose_expected = current + delta_expected

    metrics = [
        glucose_metrics("argmax_provisional", glucose_argmax, actual, current),
        glucose_metrics(
            "expected_value_provisional", glucose_expected, actual, current
        ),
    ]

    for row in metrics:
        row["patient"] = PATIENT
        row["horizon_minutes"] = HORIZON
        row["best_epoch"] = best_epoch
        row["best_validation_cross_entropy"] = best_val_loss
        row["validation_class_accuracy"] = float(
            np.mean(class_ids == true_classes)
        )

    predictions = val_meta.copy()
    predictions["true_class"] = true_classes
    predictions["argmax_class"] = class_ids
    predictions["argmax_delta_mg_dL"] = delta_argmax
    predictions["expected_delta_mg_dL"] = delta_expected
    predictions["argmax_glucose_mg_dL"] = glucose_argmax
    predictions["expected_glucose_mg_dL"] = glucose_expected
    predictions["actual_glucose_mg_dL"] = actual
    predictions["persistence_glucose_mg_dL"] = current

    predictions.to_csv(OUTPUT_DIR / "validation_predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(OUTPUT_DIR / "decoder_metrics.csv", index=False)
    pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history.csv", index=False)

    torch.save({
        "patient": PATIENT,
        "horizon_minutes": HORIZON,
        "seed": SEED,
        "features": FEATURES,
        "lookback": LOOKBACK,
        "best_epoch": best_epoch,
        "best_validation_cross_entropy": best_val_loss,
        "model_state_dict": best_state,
        "note": "Candidate architecture; provisional decoders; validation only.",
    }, OUTPUT_DIR / "best_validation_checkpoint.pt")

    config = {
        "patient": PATIENT,
        "horizon_minutes": HORIZON,
        "seed": SEED,
        "features": FEATURES,
        "lookback": LOOKBACK,
        "n_classes": N_CLASSES,
        "validation_fraction": VALIDATION_FRACTION,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "device": str(DEVICE),
        "training_sequences": int(len(y_train)),
        "validation_sequences": int(len(y_val)),
        "best_epoch": best_epoch,
        "best_validation_cross_entropy": best_val_loss,
        "testing_partition_loaded": False,
        "decoder_status": "Both methods provisional; paper postprocessing not verified.",
    }
    (OUTPUT_DIR / "experiment_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print("\nVALIDATION-ONLY DECODER COMPARISON")
    print(pd.DataFrame(metrics).to_string(index=False))
    print(f"\nOutputs saved under: {OUTPUT_DIR}")
    print("No held-out test data was loaded or evaluated.")


if __name__ == "__main__":
    main()
