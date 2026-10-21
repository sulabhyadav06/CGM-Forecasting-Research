"""Audit GluNet sequence generation and label distributions.

This script does not modify the CRNN pipeline or source CSV files.
It uses the existing GluNet data loader and sequence builder.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.glunet.glunet_data import (
    DATA_ROOT,
    FEATURES,
    N_CLASSES,
    build_sequences,
    load_split,
)

PATIENTS = [
    "540", "544", "552", "567", "584", "596",
    "559", "563", "570", "575", "588", "591",
]

HORIZONS = [30, 60]
SPLITS = ["training", "testing"]

OUTPUT_DIR = Path("experiments/glunet/reports")


def summarize_patient(patient, split, horizon):
    df = load_split(patient, split)

    X, labels, metadata = build_sequences(
        df,
        horizon_minutes=horizon,
    )

    if len(labels) == 0:
        return {
            "patient": patient,
            "split": split,
            "horizon_minutes": horizon,
            "input_rows": len(df),
            "sequences": 0,
            "features": len(FEATURES),
            "class_count": 0,
            "classes_used_pct": 0.0,
            "below_paper_range": np.nan,
            "above_paper_range": np.nan,
            "clipped_pct": np.nan,
            "delta_min": np.nan,
            "delta_max": np.nan,
            "delta_mean": np.nan,
            "delta_std": np.nan,
        }

    deltas = metadata["delta_glucose"].to_numpy()

    # Paper's stated clipping thresholds:
    # delta < -12.75 and delta > 12.85.
    # Values at the exact boundaries are not counted as outside.
    below = int(np.sum(deltas < -12.75))
    above = int(np.sum(deltas > 12.85))
    clipped = below + above

    return {
        "patient": patient,
        "split": split,
        "horizon_minutes": horizon,
        "input_rows": len(df),
        "sequences": len(labels),
        "features": X.shape[2],
        "class_count": len(np.unique(labels)),
        "classes_used_pct": (
            100.0 * len(np.unique(labels)) / N_CLASSES
        ),
        "below_paper_range": below,
        "above_paper_range": above,
        "clipped_pct": 100.0 * clipped / len(deltas),
        "delta_min": float(np.min(deltas)),
        "delta_max": float(np.max(deltas)),
        "delta_mean": float(np.mean(deltas)),
        "delta_std": float(np.std(deltas)),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for split in SPLITS:
        for horizon in HORIZONS:
            for patient in PATIENTS:
                print(
                    f"Auditing patient={patient}, "
                    f"split={split}, horizon={horizon} min",
                    flush=True,
                )

                try:
                    result = summarize_patient(
                        patient,
                        split,
                        horizon,
                    )
                    rows.append(result)
                except Exception as exc:
                    raise RuntimeError(
                        f"Audit failed for patient={patient}, "
                        f"split={split}, horizon={horizon}"
                    ) from exc

    results = pd.DataFrame(rows)

    detail_path = OUTPUT_DIR / "glunet_data_audit.csv"
    results.to_csv(detail_path, index=False)

    print("\n=== PER-PATIENT RESULTS ===")
    print(results.to_string(index=False))

    print("\n=== AGGREGATED RESULTS ===")

    summary = (
        results.groupby(["split", "horizon_minutes"], as_index=False)
        .agg(
            patients=("patient", "nunique"),
            input_rows=("input_rows", "sum"),
            sequences=("sequences", "sum"),
            below_paper_range=("below_paper_range", "sum"),
            above_paper_range=("above_paper_range", "sum"),
            delta_min=("delta_min", "min"),
            delta_max=("delta_max", "max"),
        )
    )

    print(summary.to_string(index=False))

    # Calculate the aggregate percentage using total counts,
    # rather than averaging the per-patient percentages.
    total_clipped = (
        results["below_paper_range"].fillna(0)
        + results["above_paper_range"].fillna(0)
    )

    results["clipped_count"] = total_clipped

    clipped_summary = (
        results.groupby(["split", "horizon_minutes"])
        .agg(
            sequences=("sequences", "sum"),
            clipped_count=("clipped_count", "sum"),
        )
        .reset_index()
    )

    clipped_summary["clipped_pct"] = (
        100.0
        * clipped_summary["clipped_count"]
        / clipped_summary["sequences"].replace(0, np.nan)
    )

    summary_path = OUTPUT_DIR / "glunet_data_audit_summary.csv"
    summary.to_csv(summary_path, index=False)

    clipped_summary_path = (
        OUTPUT_DIR / "glunet_clipping_summary.csv"
    )
    clipped_summary.to_csv(clipped_summary_path, index=False)

    print("\n=== CLIPPING SUMMARY ===")
    print(clipped_summary.to_string(index=False))

    print(f"\nDetailed report: {detail_path}")
    print(f"Summary report:  {summary_path}")
    print(f"Clipping report: {clipped_summary_path}")
    print(f"Data root:       {DATA_ROOT}")


if __name__ == "__main__":
    main()
