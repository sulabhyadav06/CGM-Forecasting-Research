"""Error-stratified analysis of saved all-patient regression predictions."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("experiments/glunet/runs/all_patients_regression")
OUTPUT = ROOT / "error_stratified_analysis"
OUTPUT.mkdir(parents=True, exist_ok=True)

rows = []

for patient_dir in sorted(ROOT.glob("patient*")):
    if not patient_dir.is_dir():
        continue

    patient = patient_dir.name.replace("patient", "")

    for path in sorted(patient_dir.glob("*_test_predictions.csv")):
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"Could not read {path}: {exc}")
            continue

        required = {
            "target_glucose",
            "current_glucose",
            "actual_delta_mg_dL",
            "predicted_delta_mg_dL",
            "predicted_glucose_mg_dL",
        }

        if not required.issubset(df.columns):
            print(f"Skipping unexpected prediction file: {path}")
            continue

        horizon = int(
            path.name.split("_")[1].replace("min", "")
        )

        df = df.copy()
        df["patient"] = patient
        df["horizon_minutes"] = horizon

        df["glucose_error"] = (
            df["predicted_glucose_mg_dL"] - df["target_glucose"]
        )
        df["absolute_error"] = df["glucose_error"].abs()
        df["absolute_actual_change"] = (
            df["actual_delta_mg_dL"].abs()
        )

        # Actual glucose-change magnitude.
        df["change_group"] = pd.cut(
            df["absolute_actual_change"],
            bins=[-np.inf, 5, 10, 20, 30, 50, np.inf],
            labels=[
                "0-5",
                "5-10",
                "10-20",
                "20-30",
                "30-50",
                ">50",
            ],
            right=False,
        )

        # Current glucose categories. These describe current glucose,
        # not necessarily the glucose category at the forecast horizon.
        df["current_glucose_group"] = pd.cut(
            df["current_glucose"],
            bins=[-np.inf, 70, 180, np.inf],
            labels=["<70", "70-180", ">180"],
            right=False,
        )

        # Error magnitude categories.
        for threshold in (20, 30, 40, 50):
            df[f"abs_error_gt_{threshold}"] = (
                df["absolute_error"] > threshold
            )

        rows.append(df)

if not rows:
    raise SystemExit(
        "No usable saved prediction CSVs found. "
        f"Check directory: {ROOT}"
    )

all_predictions = pd.concat(rows, ignore_index=True)

all_predictions.to_csv(
    OUTPUT / "all_test_predictions_with_errors.csv",
    index=False,
)


def summarize(group, group_columns):
    result = (
        group.groupby(group_columns, observed=True)
        .agg(
            n=("absolute_error", "size"),
            MAE_mg_dL=("absolute_error", "mean"),
            RMSE_mg_dL=(
                "glucose_error",
                lambda x: float(np.sqrt(np.mean(np.square(x)))),
            ),
            mean_signed_error_mg_dL=("glucose_error", "mean"),
            median_absolute_error_mg_dL=("absolute_error", "median"),
            actual_change_mean_mg_dL=("actual_delta_mg_dL", "mean"),
            predicted_change_mean_mg_dL=("predicted_delta_mg_dL", "mean"),
        )
        .reset_index()
    )

    return result


by_change = summarize(
    all_predictions,
    ["horizon_minutes", "change_group"],
)

by_current_glucose = summarize(
    all_predictions,
    ["horizon_minutes", "current_glucose_group"],
)

by_patient_change = summarize(
    all_predictions,
    ["patient", "horizon_minutes", "change_group"],
)

by_patient_glucose = summarize(
    all_predictions,
    ["patient", "horizon_minutes", "current_glucose_group"],
)

by_change.to_csv(
    OUTPUT / "errors_by_actual_change_magnitude.csv",
    index=False,
)
by_current_glucose.to_csv(
    OUTPUT / "errors_by_current_glucose_range.csv",
    index=False,
)
by_patient_change.to_csv(
    OUTPUT / "patient_errors_by_actual_change.csv",
    index=False,
)
by_patient_glucose.to_csv(
    OUTPUT / "patient_errors_by_current_glucose.csv",
    index=False,
)

large_error_rows = []

for horizon, group in all_predictions.groupby("horizon_minutes"):
    for threshold in (20, 30, 40, 50):
        selected = group["absolute_error"] > threshold

        large_error_rows.append({
            "horizon_minutes": int(horizon),
            "absolute_error_threshold_mg_dL": threshold,
            "n_observations": int(len(group)),
            "n_errors_above_threshold": int(selected.sum()),
            "percent_errors_above_threshold": float(
                100 * selected.mean()
            ),
        })

pd.DataFrame(large_error_rows).to_csv(
    OUTPUT / "large_error_frequency.csv",
    index=False,
)

print("\nERROR BY ACTUAL GLUCOSE-CHANGE MAGNITUDE")
print(by_change.round(3).to_string(index=False))

print("\nERROR BY CURRENT GLUCOSE RANGE")
print(by_current_glucose.round(3).to_string(index=False))

print("\nFREQUENCY OF LARGE ERRORS")
print(pd.DataFrame(large_error_rows).round(3).to_string(index=False))

print(f"\nAll analysis files saved in: {OUTPUT}")
