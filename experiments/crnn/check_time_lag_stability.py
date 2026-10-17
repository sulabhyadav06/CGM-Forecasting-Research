import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = Path("experiments/crnn/time_lag_metrics.py")
spec = importlib.util.spec_from_file_location("time_lag_metrics", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

ROOT = Path("output/crnn_initial")
OUTPUT = ROOT / "time_lag_stability.csv"

STEP = module.STEP_MINUTES
MAX_STEPS = module.MAX_LAG_MINUTES // STEP


def run_correlations(runs, lag_steps):
    correlations = []

    for run in runs:
        actual = run["target_glucose"].to_numpy(dtype=float)
        predicted = run["predicted_glucose"].to_numpy(dtype=float)

        if lag_steps > 0:
            a, p = actual[:-lag_steps], predicted[lag_steps:]
        elif lag_steps < 0:
            shift = -lag_steps
            a, p = actual[shift:], predicted[:-shift]
        else:
            a, p = actual, predicted

        if len(a) < 3 or np.std(a) == 0 or np.std(p) == 0:
            continue

        r = float(np.corrcoef(a, p)[0, 1])

        if np.isfinite(r):
            correlations.append((r, len(a)))

    if not correlations:
        return np.nan, 0

    weights = np.array([n for _, n in correlations], dtype=float)
    values = np.array([r for r, _ in correlations], dtype=float)

    return float(np.average(values, weights=weights)), int(weights.sum())


def main():
    rows = []

    for horizon in (30, 60):
        for patient_dir in sorted(ROOT.glob("patient_*")):
            patient = patient_dir.name.replace("patient_", "")
            path = patient_dir / f"horizon_{horizon}min" / "test_predictions.csv"

            if not path.exists():
                continue

            runs = module.load_contiguous_runs(path)
            candidates = []

            for steps in range(-MAX_STEPS, MAX_STEPS + 1):
                r, n = run_correlations(runs, steps)

                if np.isfinite(r):
                    candidates.append((r, steps * STEP, n))

            if not candidates:
                print(f"Skipping patient {patient}, horizon {horizon}: no valid correlations")
                continue

            # Highest weighted mean run-level correlation; smallest absolute
            # lag breaks exact ties.
            best = sorted(candidates, key=lambda x: (-x[0], abs(x[1]), x[1]))[0]

            original = module.evaluate(path, patient, horizon)
            rows.append({
                "patient": patient,
                "horizon_min": horizon,
                "original_lag_min": original["time_lag_min"],
                "run_weighted_lag_min": best[1],
                "lag_difference_min": best[1] - original["time_lag_min"],
                "original_correlation": original["best_correlation"],
                "run_weighted_correlation": best[0],
                "effective_horizon_run_weighted_min": horizon - best[1],
                "n_contiguous_runs": len(runs),
                "status": "boundary" if abs(best[1]) == module.MAX_LAG_MINUTES else "ok",
            })

            print(
                f"Patient {patient}, {horizon} min: "
                f"original lag={original['time_lag_min']}, "
                f"run-weighted lag={best[1]}, "
                f"difference={best[1] - original['time_lag_min']} min"
            )

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT, index=False)

    print(f"\nSaved: {OUTPUT}")

    for horizon, group in result.groupby("horizon_min"):
        print(f"\nHorizon: {horizon} min")
        print("Mean absolute lag difference:",
              f"{group['lag_difference_min'].abs().mean():.2f} min")
        print("Patients with changed lag:",
              int((group["lag_difference_min"] != 0).sum()), "of", len(group))


if __name__ == "__main__":
    main()
