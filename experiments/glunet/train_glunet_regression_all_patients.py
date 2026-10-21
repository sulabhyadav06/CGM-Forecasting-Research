"""All-patient continuous glucose-change regression experiment.

Patients:
    540, 544, 552, 567, 584, 596, 559, 563, 570, 575, 588, 591

Horizons:
    30 and 60 minutes

Uses existing OhioT1DM training/testing partitions.
The last 20% of each training partition is reserved chronologically
for validation. Test partitions are not used for model selection.

This is a GluNet-inspired regression experiment, not an exact
reproduction of the published GluNet classification method.
"""

import copy
import json
import random
import traceback
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


PATIENTS = (
    "540", "544", "552", "567", "584", "596",
    "559", "563", "570", "575", "588", "591",
)
HORIZONS = (30, 60)

SEED = 42
VALIDATION_FRACTION = 0.20
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
MAX_EPOCHS = 30
PATIENCE = 5

OUTPUT_DIR = Path(
    "experiments/glunet/runs/all_patients_regression"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(X, y=None, shuffle=False):
    tensors = [
        torch.from_numpy(np.asarray(X, dtype=np.float32))
    ]

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


def prepare_sequences(df, horizon):
    X, _, metadata = build_sequences(
        df,
        horizon_minutes=horizon,
    )

    if len(X) == 0:
        raise RuntimeError(
            f"No sequences generated for horizon={horizon}"
        )

    if not np.isfinite(X).all():
        raise RuntimeError("Input sequences contain non-finite values")

    current = metadata["current_glucose"].to_numpy(
        dtype=np.float32
    )
    target = metadata["target_glucose"].to_numpy(
        dtype=np.float32
    )
    delta = target - current

    if not np.isfinite(delta).all():
        raise RuntimeError("Regression targets contain non-finite values")

    return X, delta, metadata


def predict_delta(model, X):
    model.eval()
    predictions = []

    loader = make_loader(X, shuffle=False)

    with torch.no_grad():
        for (xb,) in loader:
            output = model(xb.to(DEVICE))
            predictions.append(output.cpu().numpy())

    return np.concatenate(predictions)


def train_model(
    patient,
    horizon,
    X_train,
    y_train,
    X_val,
    y_val,
    patient_output,
):
    set_seed(SEED)

    model = GluNetRegressor(
        input_features=len(FEATURES),
        channels=32,
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    train_loader = make_loader(
        X_train, y_train, shuffle=True
    )

    best_val_mae = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        squared_error_sum = 0.0
        sample_count = 0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            predictions = model(xb)
            loss = criterion(predictions, yb)

            loss.backward()
            optimizer.step()

            squared_error_sum += loss.item() * len(yb)
            sample_count += len(yb)

        train_rmse = float(
            np.sqrt(squared_error_sum / sample_count)
        )

        val_predictions = predict_delta(model, X_val)
        val_errors = val_predictions - y_val

        val_mae = float(np.mean(np.abs(val_errors)))
        val_rmse = float(np.sqrt(np.mean(val_errors ** 2)))

        history.append({
            "patient": patient,
            "horizon_minutes": horizon,
            "epoch": epoch,
            "train_rmse_mg_dL": train_rmse,
            "validation_mae_mg_dL": val_mae,
            "validation_rmse_mg_dL": val_rmse,
        })

        print(
            f"  Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"train RMSE={train_rmse:.3f} | "
            f"validation MAE={val_mae:.3f} | "
            f"validation RMSE={val_rmse:.3f}",
            flush=True,
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
        raise RuntimeError(
            f"Patient {patient}, horizon {horizon}: "
            "no valid checkpoint produced"
        )

    model.load_state_dict(best_state)

    checkpoint_path = (
        patient_output
        / f"patient{patient}_{horizon}min_best.pt"
    )

    torch.save({
        "patient": patient,
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
        "training_method": (
            "MSE_with_validation_MAE_checkpoint_selection"
        ),
    }, checkpoint_path)

    pd.DataFrame(history).to_csv(
        patient_output
        / f"patient{patient}_{horizon}min_history.csv",
        index=False,
    )

    print(
        f"  Best epoch: {best_epoch}; "
        f"validation MAE={best_val_mae:.3f}",
        flush=True,
    )

    return model, best_epoch, best_val_mae


def evaluate(model, patient, horizon, X, y_delta, metadata):
    predicted_delta = predict_delta(model, X)

    current = metadata["current_glucose"].to_numpy(dtype=float)
    actual = metadata["target_glucose"].to_numpy(dtype=float)

    predicted_glucose = current + predicted_delta
    errors = predicted_glucose - actual
    abs_errors = np.abs(errors)

    # Persistence predicts future glucose = current glucose.
    persistence_errors = current - actual
    persistence_abs = np.abs(persistence_errors)

    valid = actual > 0

    mard = (
        float(
            100.0 * np.mean(
                abs_errors[valid] / actual[valid]
            )
        )
        if np.any(valid)
        else None
    )

    mae = float(abs_errors.mean())
    persistence_mae = float(persistence_abs.mean())

    metrics = {
        "patient": patient,
        "horizon_minutes": horizon,
        "n_test_sequences": int(len(y_delta)),
        "glucose_MAE_mg_dL": mae,
        "glucose_RMSE_mg_dL": float(
            np.sqrt(np.mean(errors ** 2))
        ),
        "glucose_MARD_percent": mard,
        "mean_signed_error_mg_dL": float(errors.mean()),
        "persistence_MAE_mg_dL": persistence_mae,
        "persistence_RMSE_mg_dL": float(
            np.sqrt(np.mean(persistence_errors ** 2))
        ),
        "MAE_improvement_vs_persistence_percent": (
            float(100.0 * (1.0 - mae / persistence_mae))
            if persistence_mae > 0
            else None
        ),
        "predicted_delta_min_mg_dL": float(
            predicted_delta.min()
        ),
        "predicted_delta_median_mg_dL": float(
            np.median(predicted_delta)
        ),
        "predicted_delta_max_mg_dL": float(
            predicted_delta.max()
        ),
        "actual_delta_min_mg_dL": float(y_delta.min()),
        "actual_delta_median_mg_dL": float(
            np.median(y_delta)
        ),
        "actual_delta_max_mg_dL": float(y_delta.max()),
    }

    predictions = metadata.copy()
    predictions["actual_delta_mg_dL"] = y_delta
    predictions["predicted_delta_mg_dL"] = predicted_delta
    predictions["predicted_glucose_mg_dL"] = predicted_glucose
    predictions["persistence_glucose_mg_dL"] = current
    predictions["prediction_error_mg_dL"] = errors

    return metrics, predictions


def run_one(patient, horizon):
    patient_output = OUTPUT_DIR / f"patient{patient}"
    patient_output.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 76)
    print(
        f"ALL-PATIENT REGRESSION | PATIENT {patient} | "
        f"{horizon} MINUTES"
    )
    print("=" * 76, flush=True)

    training_source = load_split(patient, "training")
    testing_source = load_split(patient, "testing")

    split_index = int(
        len(training_source) * (1.0 - VALIDATION_FRACTION)
    )

    train_df = training_source.iloc[:split_index].copy()
    val_df = training_source.iloc[split_index:].copy()

    X_train, y_train, _ = prepare_sequences(train_df, horizon)
    X_val, y_val, _ = prepare_sequences(val_df, horizon)
    X_test, y_test, metadata_test = prepare_sequences(
        testing_source, horizon
    )

    print(f"Device: {DEVICE}")
    print(f"Training sequences: {len(y_train)}")
    print(f"Validation sequences: {len(y_val)}")
    print(f"Testing sequences: {len(y_test)}")

    model, best_epoch, best_val_mae = train_model(
        patient,
        horizon,
        X_train,
        y_train,
        X_val,
        y_val,
        patient_output,
    )

    metrics, predictions = evaluate(
        model,
        patient,
        horizon,
        X_test,
        y_test,
        metadata_test,
    )

    metrics["best_epoch"] = int(best_epoch)
    metrics["best_validation_mae_mg_dL"] = float(best_val_mae)

    metrics_path = (
        patient_output
        / f"patient{patient}_{horizon}min_test_metrics.json"
    )
    predictions_path = (
        patient_output
        / f"patient{patient}_{horizon}min_test_predictions.csv"
    )

    metrics_path.write_text(json.dumps(metrics, indent=2))
    predictions.to_csv(predictions_path, index=False)

    print("\nHELD-OUT TEST RESULTS")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    return metrics


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "patients": PATIENTS,
        "horizons_minutes": HORIZONS,
        "features": FEATURES,
        "lookback": LOOKBACK,
        "seed": SEED,
        "validation_fraction": VALIDATION_FRACTION,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "device": str(DEVICE),
        "target": "unclipped continuous future glucose change in mg/dL",
        "model": "GluNet-inspired causal convolutional regression",
        "note": (
            "Not an exact reproduction of the published GluNet "
            "classification method. Existing CRNN pipeline untouched."
        ),
    }

    (OUTPUT_DIR / "experiment_config.json").write_text(
        json.dumps(config, indent=2)
    )

    results = []
    failures = []

    total = len(PATIENTS) * len(HORIZONS)
    completed = 0

    for patient in PATIENTS:
        for horizon in HORIZONS:
            completed += 1
            print(
                f"\nJOB {completed}/{total}: "
                f"patient={patient}, horizon={horizon}",
                flush=True,
            )

            try:
                metrics = run_one(patient, horizon)
                results.append(metrics)

            except Exception as exc:
                failure = {
                    "patient": patient,
                    "horizon_minutes": horizon,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)

                print("\nERROR:", repr(exc), flush=True)
                print(failure["traceback"], flush=True)

                failure_path = (
                    OUTPUT_DIR
                    / f"failure_patient{patient}_{horizon}min.json"
                )
                failure_path.write_text(
                    json.dumps(failure, indent=2)
                )

            # Save incremental progress so completed jobs are preserved.
            pd.DataFrame(results).to_csv(
                OUTPUT_DIR / "all_patients_regression_results.csv",
                index=False,
            )
            (OUTPUT_DIR / "failures.json").write_text(
                json.dumps(failures, indent=2)
            )

    if results:
        results_df = pd.DataFrame(results)

        results_df.to_csv(
            OUTPUT_DIR / "all_patients_regression_results.csv",
            index=False,
        )

        aggregate_rows = []

        for horizon, group in results_df.groupby("horizon_minutes"):
            for metric in (
                "glucose_MAE_mg_dL",
                "glucose_RMSE_mg_dL",
                "glucose_MARD_percent",
                "mean_signed_error_mg_dL",
                "persistence_MAE_mg_dL",
                "persistence_RMSE_mg_dL",
                "MAE_improvement_vs_persistence_percent",
            ):
                values = pd.to_numeric(
                    group[metric], errors="coerce"
                ).dropna()

                aggregate_rows.append({
                    "horizon_minutes": int(horizon),
                    "metric": metric,
                    "patients_completed": int(len(values)),
                    "mean_across_patients": (
                        float(values.mean()) if len(values) else None
                    ),
                    "median_across_patients": (
                        float(values.median()) if len(values) else None
                    ),
                    "std_across_patients": (
                        float(values.std(ddof=1))
                        if len(values) > 1
                        else None
                    ),
                })

        pd.DataFrame(aggregate_rows).to_csv(
            OUTPUT_DIR / "aggregate_patient_summary.csv",
            index=False,
        )

    print("\n" + "=" * 76)
    print("ALL-PATIENT REGRESSION EXPERIMENT FINISHED")
    print(f"Successful jobs: {len(results)}/{total}")
    print(f"Failed jobs: {len(failures)}/{total}")
    print(f"Results: {OUTPUT_DIR / 'all_patients_regression_results.csv'}")
    print(f"Failures: {OUTPUT_DIR / 'failures.json'}")
    print("=" * 76)

    if failures:
        print(
            "Some jobs failed. Inspect failures.json and the individual "
            "failure files before interpreting the aggregate results."
        )


if __name__ == "__main__":
    main()
