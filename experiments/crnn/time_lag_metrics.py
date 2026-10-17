from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("output/crnn_initial")
OUTPUT = ROOT / "time_lag_metrics_revised.csv"

STEP_MINUTES = 5
MAX_LAG_MINUTES = 120
GAP_THRESHOLD_SECONDS = 330


def load_contiguous_runs(path):
    df = pd.read_csv(path, parse_dates=["target_timestamp"])
    required = {"target_timestamp", "target_glucose", "predicted_glucose"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df = df.dropna(
        subset=["target_timestamp", "target_glucose", "predicted_glucose"]
    )
    df = df.sort_values("target_timestamp").reset_index(drop=True)

    if df["target_timestamp"].duplicated().any():
        raise ValueError(f"Duplicate timestamps in {path}")

    gap_seconds = df["target_timestamp"].diff().dt.total_seconds()
    run_id = (gap_seconds > GAP_THRESHOLD_SECONDS).cumsum()

    return [
        run.reset_index(drop=True)
        for _, run in df.groupby(run_id, sort=False)
        if len(run) >= 3
    ]


def correlation_at_lag(runs, lag_steps):
    actual_parts = []
    predicted_parts = []

    for run in runs:
        actual = run["target_glucose"].to_numpy(dtype=float)
        predicted = run["predicted_glucose"].to_numpy(dtype=float)

        if lag_steps > 0:
            # Positive lag: predicted trace is delayed relative to actual.
            a = actual[:-lag_steps]
            p = predicted[lag_steps:]
        elif lag_steps < 0:
            shift = -lag_steps
            a = actual[shift:]
            p = predicted[:-shift]
        else:
            a = actual
            p = predicted

        if len(a) >= 3:
            actual_parts.append(a)
            predicted_parts.append(p)

    if not actual_parts:
        return np.nan, 0

    actual_all = np.concatenate(actual_parts)
    predicted_all = np.concatenate(predicted_parts)

    if np.std(actual_all) == 0 or np.std(predicted_all) == 0:
        return np.nan, len(actual_all)

    return (
        float(np.corrcoef(actual_all, predicted_all)[0, 1]),
        len(actual_all),
    )


def evaluate(path, patient, horizon):
    runs = load_contiguous_runs(path)

    if not runs:
        return {
            "patient": patient,
            "horizon_min": horizon,
            "n_contiguous_runs": 0,
            "time_lag_min": np.nan,
            "best_correlation": np.nan,
            "effective_prediction_horizon_min": np.nan,
            "search_boundary": False,
            "status": "insufficient data",
        }

    max_steps = MAX_LAG_MINUTES // STEP_MINUTES
    candidates = []

    for lag_steps in range(-max_steps, max_steps + 1):
        corr, n_pairs = correlation_at_lag(runs, lag_steps)

        if np.isfinite(corr):
            candidates.append({
                "lag_min": lag_steps * STEP_MINUTES,
                "correlation": corr,
                "n_pairs": n_pairs,
            })

    if not candidates:
        best = None
    else:
        # Highest correlation; ties prefer the smallest absolute lag.
        best = sorted(
            candidates,
            key=lambda x: (
                -x["correlation"],
                abs(x["lag_min"]),
                x["lag_min"],
            ),
        )[0]

    if best is None:
        lag = corr = effective = np.nan
        boundary = False
        status = "correlation undefined"
        n_pairs = 0
    else:
        lag = best["lag_min"]
        corr = best["correlation"]
        effective = horizon - lag
        boundary = abs(lag) == MAX_LAG_MINUTES
        n_pairs = best["n_pairs"]
        status = "check search boundary" if boundary else "ok"

    return {
        "patient": patient,
        "horizon_min": horizon,
        "n_predictions": sum(len(run) for run in runs),
        "n_contiguous_runs": len(runs),
        "n_pairs_at_best_lag": n_pairs,
        "time_lag_min": lag,
        "best_correlation": corr,
        "effective_prediction_horizon_min": effective,
        "search_boundary": boundary,
        "status": status,
    }


def main():
    results = []

    for horizon in (30, 60):
        for patient_dir in sorted(ROOT.glob("patient_*")):
            patient = patient_dir.name.replace("patient_", "")
            path = patient_dir / f"horizon_{horizon}min" / "test_predictions.csv"

            if not path.exists():
                continue

            result = evaluate(path, patient, horizon)
            results.append(result)

            print(
                f"Patient {patient}, horizon {horizon}: "
                f"lag={result['time_lag_min']} min, "
                f"r={result['best_correlation']}, "
                f"effective horizon="
                f"{result['effective_prediction_horizon_min']} min, "
                f"{result['status']}"
            )

    if not results:
        raise RuntimeError(f"No prediction files found under {ROOT}")

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT, index=False)

    print(f"\nSaved to {OUTPUT}")
    print("\nSummary across patients (mean ± sample SD):")

    for horizon, group in out.groupby("horizon_min"):
        print(f"\n{horizon}-minute horizon")

        for col in [
            "time_lag_min",
            "effective_prediction_horizon_min",
            "best_correlation",
        ]:
            values = group[col].dropna()

            if len(values):
                print(
                    f"  {col}: {values.mean():.2f} ± "
                    f"{values.std(ddof=1):.2f}"
                )

        print(
            "  Boundary results:",
            int(group["search_boundary"].sum()),
            "of",
            len(group),
        )


if __name__ == "__main__":
    main()
