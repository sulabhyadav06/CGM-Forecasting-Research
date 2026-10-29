"""Compare saved GluNet candidate forecasts against same-split baselines.

This reads existing output files only; it does not retrain or modify data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.glunet.glunet_data import quantize_delta


def metrics(actual: np.ndarray, forecast: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    err = forecast - actual
    return {
        "MAE_mg_dL": float(np.mean(np.abs(err))),
        "RMSE_mg_dL": float(np.sqrt(np.mean(err**2))),
        "MARD_pct": float(100 * np.mean(np.abs(err) / np.maximum(np.abs(actual), 1e-8))),
        "mean_signed_error_mg_dL": float(np.mean(err)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-run-dir", required=True,
                        help="Timestamped candidate run directory containing patient540_30min/")
    parser.add_argument("--audit-dir", required=True,
                        help="Directory containing baseline_metrics.csv from audit_glunet_targets.py")
    parser.add_argument("--patient", default="540")
    parser.add_argument("--horizon", type=int, default=30)
    args = parser.parse_args()

    candidate_dir = Path(args.candidate_run_dir).expanduser().resolve()
    audit_dir = Path(args.audit_dir).expanduser().resolve()
    pred_path = candidate_dir / f"patient{args.patient}_{args.horizon}min" / "test_predictions_and_diagnostics.csv"
    baseline_path = audit_dir / "baseline_metrics.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Saved predictions not found: {pred_path}")
    if not baseline_path.exists():
        raise FileNotFoundError(f"Audit baseline file not found: {baseline_path}")

    pred = pd.read_csv(pred_path)
    baseline = pd.read_csv(baseline_path)
    required = {"target_glucose", "current_glucose", "delta_glucose",
                "prediction_expected_class_value", "prediction_argmax_class_value",
                "prediction_persistence", "predicted_class_argmax_zero_based"}
    missing = sorted(required - set(pred.columns))
    if missing:
        raise ValueError(f"Prediction file missing columns: {missing}")
    if pred.empty:
        raise ValueError("Prediction file contains no rows")

    # Recompute targets with the repository's canonical quantizer.
    true_classes = quantize_delta(pred["delta_glucose"].to_numpy(dtype=float))
    predicted_classes = pred["predicted_class_argmax_zero_based"].to_numpy(dtype=int)
    counts = np.bincount(predicted_classes, minlength=256)
    true_counts = np.bincount(true_classes, minlength=256)

    rows = []
    forecasts = {
        "candidate_expected_class_value": pred["prediction_expected_class_value"].to_numpy(),
        "candidate_argmax_class_value": pred["prediction_argmax_class_value"].to_numpy(),
        "persistence_from_saved_predictions": pred["prediction_persistence"].to_numpy(),
    }
    actual = pred["target_glucose"].to_numpy(dtype=float)
    for name, forecast in forecasts.items():
        rows.append({"model": name, **metrics(actual, forecast)})

    audit_test = baseline[
        (baseline["patient"].astype(str) == str(args.patient))
        & (baseline["horizon_minutes"] == args.horizon)
        & (baseline["evaluation_split"] == "test")
    ]
    for _, row in audit_test.iterrows():
        rows.append({
            "model": f"baseline_{row['baseline']}",
            "MAE_mg_dL": float(row["MAE_mg_dL"]),
            "RMSE_mg_dL": float(row["RMSE_mg_dL"]),
            "MARD_pct": float(row["MARD_pct"]),
            "mean_signed_error_mg_dL": np.nan,
        })

    result = pd.DataFrame(rows).sort_values("MAE_mg_dL")
    print(f"Patient {args.patient} | horizon {args.horizon} min | test windows: {len(pred)}")
    print("\nFORECAST METRICS (lower is better)")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nCLASS COLLAPSE")
    print(f"True classes present: {np.count_nonzero(true_counts)}")
    print(f"Predicted classes present: {np.count_nonzero(counts)}")
    print(f"True endpoint share: {100 * (true_counts[0] + true_counts[-1]) / len(true_classes):.2f}%")
    print(f"Predicted endpoint share: {100 * (counts[0] + counts[-1]) / len(predicted_classes):.2f}%")
    print(f"Most common predicted class: {int(np.argmax(counts))} ({100 * counts.max() / len(predicted_classes):.2f}%)")
    print("\nInterpretation: compare the two candidate decoders with persistence and the training-only baselines. This is an existing-checkpoint evaluation, not a new training run.")


if __name__ == "__main__":
    main()
