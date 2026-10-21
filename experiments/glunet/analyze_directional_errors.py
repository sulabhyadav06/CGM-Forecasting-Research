"""Directional error analysis for all-patient CGM regression predictions."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("experiments/glunet/runs/all_patients_regression")
OUTPUT = ROOT / "directional_error_analysis"
OUTPUT.mkdir(parents=True, exist_ok=True)

frames = []

for patient_dir in sorted(ROOT.glob("patient*")):
    if not patient_dir.is_dir():
        continue

    patient = patient_dir.name.replace("patient", "")

    for path in patient_dir.glob("*_test_predictions.csv"):
        df = pd.read_csv(path)

        required = {
            "current_glucose",
            "target_glucose",
            "actual_delta_mg_dL",
            "predicted_delta_mg_dL",
            "predicted_glucose_mg_dL",
        }

        if not required.issubset(df.columns):
            print(f"Skipping incompatible file: {path}")
            continue

        horizon = int(
            path.name.split("_")[1].replace("min", "")
        )

        df = df.copy()
        df["patient"] = patient
        df["horizon_minutes"] = horizon

        # Positive error means overprediction; negative means underprediction.
        df["error_mg_dL"] = (
            df["predicted_glucose_mg_dL"] - df["target_glucose"]
        )
        df["absolute_error_mg_dL"] = df["error_mg_dL"].abs()

        # Direction of the actual change.
        df["actual_direction"] = np.select(
            [
                df["actual_delta_mg_dL"] > 0,
                df["actual_delta_mg_dL"] < 0,
            ],
            ["rise", "fall"],
            default="no_change",
        )

        # Direction of the predicted change.
        df["predicted_direction"] = np.select(
            [
                df["predicted_delta_mg_dL"] > 0,
                df["predicted_delta_mg_dL"] < 0,
            ],
            ["rise", "fall"],
            default="no_change",
        )

        df["actual_change_magnitude"] = (
            df["actual_delta_mg_dL"].abs()
        )

        df["magnitude_group"] = pd.cut(
            df["actual_change_magnitude"],
            bins=[-np.inf, 10, 20, 30, 50, np.inf],
            labels=["<10", "10-20", "20-30", "30-50", ">50"],
            right=False,
        )

        # Positive actual changes above 50 are large rises.
        # Negative actual changes below -50 are large falls.
        df["large_change_direction"] = np.select(
            [
                df["actual_delta_mg_dL"] > 50,
                df["actual_delta_mg_dL"] < -50,
            ],
            ["large_rise", "large_fall"],
            default="other",
        )

        frames.append(df)

if not frames:
    raise SystemExit(
        f"No usable prediction files found in {ROOT}"
    )

all_df = pd.concat(frames, ignore_index=True)

all_df.to_csv(
    OUTPUT / "all_predictions_directional.csv",
    index=False,
)


def summarize(group, columns):
    return (
        group.groupby(columns, observed=True)
        .agg(
            n=("error_mg_dL", "size"),
            MAE_mg_dL=("absolute_error_mg_dL", "mean"),
            RMSE_mg_dL=(
                "error_mg_dL",
                lambda x: float(np.sqrt(np.mean(np.square(x)))),
            ),
            mean_signed_error_mg_dL=("error_mg_dL", "mean"),
            mean_actual_change_mg_dL=("actual_delta_mg_dL", "mean"),
            mean_predicted_change_mg_dL=("predicted_delta_mg_dL", "mean"),
            direction_match_rate=(
                "actual_direction",
                lambda x: float(
                    np.mean(
                        x.to_numpy()
                        == group.loc[x.index, "predicted_direction"].to_numpy()
                    )
                ),
            ),
        )
        .reset_index()
    )


# Overall performance by the direction of the actual glucose change.
directional = summarize(
    all_df,
    ["horizon_minutes", "actual_direction"],
)

directional.to_csv(
    OUTPUT / "errors_by_actual_direction.csv",
    index=False,
)

# Separate large rises from large falls.
large = all_df[
    all_df["large_change_direction"].isin(
        ["large_rise", "large_fall"]
    )
].copy()

large_summary = summarize(
    large,
    ["horizon_minutes", "large_change_direction"],
)

large_summary.to_csv(
    OUTPUT / "large_rise_vs_fall.csv",
    index=False,
)

# Patient-level performance on large rises and large falls.
patient_large = summarize(
    large,
    ["patient", "horizon_minutes", "large_change_direction"],
)

patient_large.to_csv(
    OUTPUT / "patient_large_rise_vs_fall.csv",
    index=False,
)

# Error by actual change magnitude and direction.
by_magnitude_direction = summarize(
    all_df,
    ["horizon_minutes", "magnitude_group", "actual_direction"],
)

by_magnitude_direction.to_csv(
    OUTPUT / "errors_by_magnitude_and_direction.csv",
    index=False,
)

# How often does the predicted change have the wrong direction?
all_df["direction_correct"] = (
    all_df["actual_direction"] == all_df["predicted_direction"]
)

direction_accuracy = (
    all_df.groupby("horizon_minutes")
    .agg(
        n=("direction_correct", "size"),
        direction_match_rate=("direction_correct", "mean"),
    )
    .reset_index()
)

direction_accuracy["direction_match_percent"] = (
    100 * direction_accuracy["direction_match_rate"]
)

direction_accuracy.to_csv(
    OUTPUT / "direction_match_summary.csv",
    index=False,
)

print("\nERROR BY ACTUAL DIRECTION")
print(directional.round(3).to_string(index=False))

print("\nLARGE RISE VERSUS LARGE FALL")
print(large_summary.round(3).to_string(index=False))

print("\nDIRECTION MATCH RATE")
print(direction_accuracy.round(3).to_string(index=False))

print("\nERROR BY MAGNITUDE AND DIRECTION")
print(by_magnitude_direction.round(3).to_string(index=False))

print(f"\nSaved all results to: {OUTPUT}")
