from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("output/crnn_initial")
HORIZONS = (30, 60)

# Clarke Error Grid zones:
# A: clinically accurate predictions
# B: errors unlikely to lead to inappropriate treatment
# C: errors that could lead to unnecessary treatment
# D: failure to detect clinically significant glucose levels
# E: erroneous treatment decisions in the opposite direction


def clarke_error_grid_zones(reference, prediction):
    """
    Python 3 translation of the zone-counting rules in the
    downloaded ClarkeErrorGrid.py reference implementation.

    Returns one zone label (A-E) per glucose pair.
    """
    reference = np.asarray(reference, dtype=float)
    prediction = np.asarray(prediction, dtype=float)

    if reference.shape != prediction.shape:
        raise ValueError("Reference and prediction arrays must match.")

    if not np.isfinite(reference).all() or not np.isfinite(prediction).all():
        raise ValueError("Glucose values must all be finite.")

    zones = []

    for ref, pred in zip(reference, prediction):
        if (
          (ref < 70 and pred < 70)
          or abs(ref - pred) < 0.2 * ref
        ):
            zone = "A"

        elif (
            (ref >= 180 and pred <= 70)
            or (ref <= 70 and pred >= 180)
        ):
            zone = "E"

        elif (
            ((ref >= 70 and ref <= 290) and pred >= ref + 110)
            or (
                (ref >= 130 and ref <= 180)
                and pred <= (7 / 5) * ref - 182
            )
        ):
            zone = "C"

        elif (
             (ref >= 240 and 70 <= pred <= 180)
              or (ref <= 175 / 3 and 70 <= pred <= 180)
              or (175 / 3 <= ref <= 70 and pred >= (6 / 5) * ref)
       ):
            zone = "D"

        else:
            zone = "B"

        zones.append(zone)

    return np.asarray(zones)


def evaluate_file(path):
    df = pd.read_csv(path)

    required = {"target_glucose", "predicted_glucose"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    actual = df["target_glucose"].to_numpy(dtype=float)
    predicted = df["predicted_glucose"].to_numpy(dtype=float)

    zones = clarke_error_grid_zones(actual, predicted)

    result = {
        "n": len(df),
        "MAE_mg_dL": np.mean(np.abs(predicted - actual)),
        "RMSE_mg_dL": np.sqrt(np.mean((predicted - actual) ** 2)),
    }

    for zone in "ABCDE":
        count = int(np.sum(zones == zone))
        result[f"CEG_{zone}_count"] = count
        result[f"CEG_{zone}_percent"] = 100 * count / len(zones)

    return result, zones


def main():
    rows = []

    for horizon in HORIZONS:
        patient_dirs = sorted(ROOT.glob("patient_*"))

        for patient_dir in patient_dirs:
            patient = patient_dir.name.replace("patient_", "")
            path = patient_dir / f"horizon_{horizon}min" / "test_predictions.csv"

            if not path.exists():
                print(f"SKIP: missing {path}")
                continue

            metrics, zones = evaluate_file(path)
            rows.append({
                "patient": patient,
                "horizon_minutes": horizon,
                **metrics,
            })

    if not rows:
        raise RuntimeError(f"No prediction files found under {ROOT}")

    results = pd.DataFrame(rows).sort_values(
        ["horizon_minutes", "patient"]
    )

    out = ROOT / "clinical_metrics_initial.csv"
    results.to_csv(out, index=False)

    print("\nPatient-level clinical metrics")
    print(results.to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
