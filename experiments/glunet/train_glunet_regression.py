"""Continuous-regression pilot for OhioT1DM patient 540.

Uses the existing GluNet-inspired causal convolutional backbone,
but replaces 256-class classification with continuous glucose-change
regression. This is a controlled experimental variant, not an exact
reproduction of the published GluNet model.
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
    build_sequences,
    load_split,
)
from experiments.glunet.glunet_regression_model import GluNetRegressor


PATIENT = "540"
HORIZONS = (30, 60)

SEED = 42
VALIDATION_FRACTION = 0.20
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
MAX_EPOCHS = 30
PATIENCE = 5

OUTPUT_DIR = Path("experiments/glunet/runs/pilot_patient540_regression")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_sequences(df, horizon):
    X, _, metadata = build_sequences(
        df,
        horizon_minutes=horizon,
    )

    if len(X) == 0:
        raise RuntimeError(f"No sequences generated for {horizon} minutes")

    if not np.isfinite(X).all():
        raise RuntimeError("Inputs contain non-finite values")

    current = metadata["current_glucose"].to_numpy(dtype=np.float32)
    target = metadata["target_glucose"].to_numpy(dtype=np.float32)
    delta = target - current

    if not np.isfinite(delta).all():
        raise RuntimeError("Regression targets contain non-finite values")

    # Regression targets are continuous, unclipped glucose changes.
    return X, delta, metadata


def make_loader(X, y=None, shuffle=False):
    tensors = [torch.from_numpy(np.asarray(X, dtype=np.float32))]

    if y is not None:
        tensors.append(
            torch.from_numpy(np.asarray(y, dtype=np.float32))
        )

    return DataLoader(
        TensorDataset(*tensors),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )


def predict_delta(model, X):
    model.eval()
    results = []

    loader = make_loader(X, shuffle=False)

    with torch.no_grad():
        for (xb,) in loader:
            results.append(model(xb.to(DEVICE)).cpu().numpy())

    return np.concatenate(results)


def train_model(X_train, y_train, X_val, y_val, horizon):
    model = GluNetRegressor(
        input_features=len(FEATURES),
        channels=32,
    ).to(DEVICE)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader = make_loader(X_val, y_val, shuffle=False)

    best_val_mae = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        total_squared_error = 0.0
        total_count = 0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            predictions = model(xb)
            loss = criterion(predictions, yb)

            loss.backward()
            optimizer.step()

            total_squared_error += loss.item() * len(yb)
            total_count += len(yb)

        train_rmse = np.sqrt(total_squared_error / total_count)

        val_predictions = predict_delta(model, X_val)
        val_errors = val_predictions - y_val

        val_mae = float(np.mean(np.abs(val_errors)))
        val_rmse = float(np.sqrt(np.mean(val_errors ** 2)))

        history.append({
            "epoch": epoch,
            "train_rmse_mg_dL": float(train_rmse),
            "validation_mae_mg_dL": val_mae,
            "validation_rmse_mg_dL": val_rmse,
        })

        print(
            f"  Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"train RMSE={train_rmse:.4f} | "
            f"validation MAE={val_mae:.4f} | "
            f"validation RMSE={val_rmse:.4f}"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")

    model.load_state_dict(best_state)

    checkpoint_path = (
        OUTPUT_DIR / f"patient{PATIENT}_{horizon}min_best.pt"
    )

    torch.save({
        "patient": PATIENT,
        "horizon_minutes": horizon,
        "features": FEATURES,
        "lookback": LOOKBACK,
        "seed": SEED,
        "best_epoch": best_epoch,
        "best_validation_mae_mg_dL": best_val_mae,
        "model_state_dict": best_state,
        "model_config": {
            "input_features": len(FEATURES),
            "channels": 32,
            "output_type": "continuous_glucose_change_mg_dL",
        },
        "training_method": "MSE_with_validation_MAE_checkpoint_selection",
    }, checkpoint_path)

    pd.DataFrame(history).to_csv(
        OUTPUT_DIR / f"patient{PATIENT}_{horizon}min_history.csv",
        index=False,
    )

    print(f"  Best epoch: {best_epoch}")
    print(f"  Best validation MAE: {best_val_mae:.4f} mg/dL")
    print(f"  Checkpoint: {checkpoint_path}")

    return model, best_epoch, best_val_mae


def evaluate(model, X, y_delta, metadata, horizon):
    predicted_delta = predict_delta(model, X)

    current = metadata["current_glucose"].to_numpy(dtype=float)
    actual = metadata["target_glucose"].to_numpy(dtype=float)

    predicted_glucose = current + predicted_delta
    errors = predicted_glucose - actual
    absolute_errors = np.abs(errors)

    persistence_errors = current - actual

    valid = actual > 0
    mard = (
        float(
            100.0 * np.mean(
                absolute_errors[valid] / actual[valid]
            )
        )
        if valid.any()
        else None
    )

    metrics = {
        "patient": PATIENT,
        "horizon_minutes": horizon,
        "n_test_sequences": int(len(y_delta)),
        "glucose_MAE_mg_dL": float(absolute_errors.mean()),
        "glucose_RMSE_mg_dL": float(
            np.sqrt(np.mean(errors ** 2))
        ),
        "glucose_MARD_percent": mard,
        "mean_signed_error_mg_dL": float(errors.mean()),
        "persistence_MAE_mg_dL": float(
            np.mean(np.abs(persistence_errors))
        ),
        "persistence_RMSE_mg_dL": float(
            np.sqrt(np.mean(persistence_errors ** 2))
        ),
        "MAE_improvement_vs_persistence_percent": float(
            100.0 * (
                1.0
                - absolute_errors.mean()
                / np.mean(np.abs(persistence_errors))
            )
        ),
        "predicted_delta_min_mg_dL": float(predicted_delta.min()),
        "predicted_delta_median_mg_dL": float(
            np.median(predicted_delta)
        ),
        "predicted_delta_max_mg_dL": float(predicted_delta.max()),
        "actual_delta_min_mg_dL": float(y_delta.min()),
        "actual_delta_median_mg_dL": float(np.median(y_delta)),
        "actual_delta_max_mg_dL": float(y_delta.max()),
    }

    predictions = metadata.copy()
    predictions["actual_delta_mg_dL"] = y_delta
    predictions["predicted_delta_mg_dL"] = predicted_delta
    predictions["predicted_glucose_mg_dL"] = predicted_glucose
    predictions["persistence_glucose_mg_dL"] = current
    predictions["prediction_error_mg_dL"] = errors

    return metrics, predictions


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_source = load_split(PATIENT, "training")
    test_source = load_split(PATIENT, "testing")

    split_index = int(
        len(train_source) * (1.0 - VALIDATION_FRACTION)
    )

    train_df = train_source.iloc[:split_index].copy()
    val_df = train_source.iloc[split_index:].copy()

    all_results = []

    for horizon in HORIZONS:
        print("\n" + "=" * 72)
        print(f"CONTINUOUS REGRESSION | PATIENT {PATIENT} | {horizon} MIN")
        print("=" * 72)

        X_train, y_train, _ = prepare_sequences(train_df, horizon)
        X_val, y_val, _ = prepare_sequences(val_df, horizon)
        X_test, y_test, metadata_test = prepare_sequences(
            test_source, horizon
        )

        print(f"Device: {DEVICE}")
        print(f"Training sequences: {len(y_train)}")
        print(f"Validation sequences: {len(y_val)}")
        print(f"Testing sequences: {len(y_test)}")

        set_seed(SEED)

        model, best_epoch, best_val_mae = train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            horizon,
        )

        metrics, predictions = evaluate(
            model,
            X_test,
            y_test,
            metadata_test,
            horizon,
        )

        metrics["best_epoch"] = int(best_epoch)
        metrics["best_validation_mae_mg_dL"] = float(best_val_mae)

        metrics_path = (
            OUTPUT_DIR
            / f"patient{PATIENT}_{horizon}min_test_metrics.json"
        )
        predictions_path = (
            OUTPUT_DIR
            / f"patient{PATIENT}_{horizon}min_test_predictions.csv"
        )

        metrics_path.write_text(json.dumps(metrics, indent=2))
        predictions.to_csv(predictions_path, index=False)

        all_results.append(metrics)

        print("\nHELD-OUT TEST RESULTS")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

        print(f"Metrics saved: {metrics_path}")
        print(f"Predictions saved: {predictions_path}")

    summary_path = OUTPUT_DIR / "regression_pilot_summary.csv"
    pd.DataFrame(all_results).to_csv(summary_path, index=False)

    print("\n" + "=" * 72)
    print("REGRESSION PILOT COMPLETE")
    print(f"Summary: {summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
