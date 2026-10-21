"""Clinical evaluation of saved GluNet-inspired regression predictions.

Reads the three existing regression runs and writes new reports only to:
experiments/glunet/reports/clinical_evaluation/

This evaluates saved predictions; it does not retrain models.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.crnn.clinical_metrics import clarke_error_grid_zones


RUNS = {
    "mse": Path("experiments/glunet/runs/all_patients_regression"),
    "huber": Path("experiments/glunet/runs/all_patients_regression_huber"),
    "mae": Path("experiments/glunet/runs/all_patients_regression_mae"),
}
OUTPUT = Path("experiments/glunet/reports/clinical_evaluation")

PATIENTS = (
    "540", "544", "552", "567", "584", "596",
    "559", "563", "570", "575", "588", "591",
)
HORIZONS = (30, 60)

STEP_MINUTES = 5
MAX_LAG_MINUTES = 120
GAP_THRESHOLD_SECONDS = 330

REQUIRED = {
    "target_timestamp",
    "current_glucose",
    "target_glucose",
    "predicted_glucose_mg_dL",
    "persistence_glucose_mg_dL",
}


def load_predictions(path):
    df = pd.read_csv(path, parse_dates=["target_timestamp"])
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    df = df.copy()
    for col in (
        "current_glucose",
        "target_glucose",
        "predicted_glucose_mg_dL",
        "persistence_glucose_mg_dL",
    ):
        df[col] = pd.to_numeric(df[col], errors="raise")

    if df["target_timestamp"].isna().any():
        raise ValueError(f"{path}: invalid target timestamps")

    if df["target_timestamp"].duplicated().any():
        raise ValueError(f"{path}: duplicate target timestamps")

    numeric = df[
        [
            "current_glucose",
            "target_glucose",
            "predicted_glucose_mg_dL",
            "persistence_glucose_mg_dL",
        ]
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric).all():
        raise ValueError(f"{path}: non-finite glucose values")

    df = df.sort_values("target_timestamp").reset_index(drop=True)
    return df


def glucose_metrics(actual, predicted):
    error = predicted - actual
    nonzero = np.abs(actual) > 0

    return {
        "n": int(len(actual)),
        "MAE_mg_dL": float(np.mean(np.abs(error))),
        "RMSE_mg_dL": float(np.sqrt(np.mean(error**2))),
        "MARD_percent": (
            float(100 * np.mean(np.abs(error[nonzero] / actual[nonzero])))
            if nonzero.any()
            else np.nan
        ),
        "mean_signed_error_mg_dL": float(np.mean(error)),
    }


def clinical_metrics(df):
    actual = df["target_glucose"].to_numpy(dtype=float)
    predicted = df["predicted_glucose_mg_dL"].to_numpy(dtype=float)
    persistence = df["persistence_glucose_mg_dL"].to_numpy(dtype=float)

    zones = clarke_error_grid_zones(actual, predicted)
    result = glucose_metrics(actual, predicted)
    baseline = glucose_metrics(actual, persistence)

    result["persistence_MAE_mg_dL"] = baseline["MAE_mg_dL"]
    result["persistence_RMSE_mg_dL"] = baseline["RMSE_mg_dL"]
    result["MAE_improvement_vs_persistence_percent"] = (
        100 * (baseline["MAE_mg_dL"] - result["MAE_mg_dL"])
        / baseline["MAE_mg_dL"]
        if baseline["MAE_mg_dL"] > 0
        else np.nan
    )

    for zone in "ABCDE":
        count = int(np.sum(zones == zone))
        result[f"CEG_{zone}_count"] = count
        result[f"CEG_{zone}_percent"] = (
            100 * count / len(zones) if len(zones) else np.nan
        )

    return result


def contiguous_runs(df):
    gap = df["target_timestamp"].diff().dt.total_seconds()
    run_id = (gap > GAP_THRESHOLD_SECONDS).cumsum()
    return [
        group.reset_index(drop=True)
        for _, group in df.groupby(run_id, sort=False)
        if len(group) >= 3
    ]


def correlation_at_lag(runs, lag_steps):
    actual_parts = []
    predicted_parts = []

    for run in runs:
        actual = run["target_glucose"].to_numpy(dtype=float)
        predicted = run["predicted_glucose_mg_dL"].to_numpy(dtype=float)

        if lag_steps > 0:
            # Positive lag means the predicted trace is delayed.
            a, p = actual[:-lag_steps], predicted[lag_steps:]
        elif lag_steps < 0:
            shift = -lag_steps
            a, p = actual[shift:], predicted[:-shift]
        else:
            a, p = actual, predicted

        if len(a) >= 3:
            actual_parts.append(a)
            predicted_parts.append(p)

    if not actual_parts:
        return np.nan, 0

    actual_all = np.concatenate(actual_parts)
    predicted_all = np.concatenate(predicted_parts)

    if np.std(actual_all) == 0 or np.std(predicted_all) == 0:
        return np.nan, len(actual_all)

    return float(np.corrcoef(actual_all, predicted_all)[0, 1]), len(actual_all)


def lag_metrics(df, patient, horizon):
    runs = contiguous_runs(df)
    base = {
        "patient": patient,
        "horizon_minutes": horizon,
        "n_predictions_in_usable_runs": int(sum(len(r) for r in runs)),
        "n_contiguous_runs": len(runs),
    }

    if not runs:
        return {
            **base,
            "n_pairs_at_best_lag": 0,
            "time_lag_minutes": np.nan,
            "best_correlation": np.nan,
            "effective_prediction_horizon_minutes": np.nan,
            "search_boundary": False,
            "status": "insufficient data",
        }

    max_steps = MAX_LAG_MINUTES // STEP_MINUTES
    candidates = []

    for steps in range(-max_steps, max_steps + 1):
        corr, n_pairs = correlation_at_lag(runs, steps)
        if np.isfinite(corr):
            candidates.append({
                "lag": steps * STEP_MINUTES,
                "correlation": corr,
                "n_pairs": n_pairs,
            })

    if not candidates:
        return {
            **base,
            "n_pairs_at_best_lag": 0,
            "time_lag_minutes": np.nan,
            "best_correlation": np.nan,
            "effective_prediction_horizon_minutes": np.nan,
            "search_boundary": False,
            "status": "correlation undefined",
        }

    best = sorted(
        candidates,
        key=lambda item: (
            -item["correlation"],
            abs(item["lag"]),
            item["lag"],
        ),
    )[0]
    boundary = abs(best["lag"]) == MAX_LAG_MINUTES

    return {
        **base,
        "n_pairs_at_best_lag": best["n_pairs"],
        "time_lag_minutes": best["lag"],
        "best_correlation": best["correlation"],
        "effective_prediction_horizon_minutes": horizon - best["lag"],
        "search_boundary": boundary,
        "status": "check search boundary" if boundary else "ok",
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    clinical_rows = []
    lag_rows = []
    audit_rows = []
    errors = []

    for variant, run_root in RUNS.items():
        for patient in PATIENTS:
            for horizon in HORIZONS:
                path = (
                    run_root
                    / f"patient{patient}"
                    / f"patient{patient}_{horizon}min_test_predictions.csv"
                )

                if not path.exists():
                    errors.append({
                        "variant": variant,
                        "patient": patient,
                        "horizon_minutes": horizon,
                        "path": str(path),
                        "error": "missing prediction file",
                    })
                    continue

                try:
                    df = load_predictions(path)
                    metrics = clinical_metrics(df)
                    clinical_rows.append({
                        "variant": variant,
                        "patient": patient,
                        "horizon_minutes": horizon,
                        "source_file": str(path),
                        **metrics,
                    })
                    lag_rows.append({
                        "variant": variant,
                        **lag_metrics(df, patient, horizon),
                    })
                    audit_rows.append({
                        "variant": variant,
                        "patient": patient,
                        "horizon_minutes": horizon,
                        "rows": len(df),
                        "finite_values": True,
                        "unique_target_timestamps": True,
                        "status": "ok",
                    })
                except Exception as exc:
                    errors.append({
                        "variant": variant,
                        "patient": patient,
                        "horizon_minutes": horizon,
                        "path": str(path),
                        "error": repr(exc),
                    })

    if not clinical_rows:
        raise RuntimeError("No prediction files were successfully evaluated.")

    clinical = pd.DataFrame(clinical_rows).sort_values(
        ["variant", "horizon_minutes", "patient"]
    )
    lag = pd.DataFrame(lag_rows).sort_values(
        ["variant", "horizon_minutes", "patient"]
    )
    audit = pd.DataFrame(audit_rows)
    error_df = pd.DataFrame(
        errors,
        columns=["variant", "patient", "horizon_minutes", "path", "error"],
    )

    clinical.to_csv(OUTPUT / "patient_clinical_metrics.csv", index=False)
    lag.to_csv(OUTPUT / "patient_time_lag_metrics.csv", index=False)
    audit.to_csv(OUTPUT / "prediction_audit.csv", index=False)
    error_df.to_csv(OUTPUT / "evaluation_errors.csv", index=False)

    summary_rows = []
    for (variant, horizon), group in clinical.groupby(
        ["variant", "horizon_minutes"]
    ):
        row = {
            "variant": variant,
            "horizon_minutes": int(horizon),
            "patients_evaluated": int(group["patient"].nunique()),
        }
        for metric in (
            "MAE_mg_dL",
            "RMSE_mg_dL",
            "MARD_percent",
            "mean_signed_error_mg_dL",
            "CEG_A_percent",
            "CEG_B_percent",
            "CEG_C_percent",
            "CEG_D_percent",
            "CEG_E_percent",
            "MAE_improvement_vs_persistence_percent",
        ):
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.nanmean(values))
            row[f"{metric}_sd"] = (
                float(np.nanstd(values, ddof=1))
                if np.isfinite(values).sum() > 1
                else np.nan
            )
        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(
        OUTPUT / "summary_mean_sd_across_patients.csv", index=False
    )

    print(f"Output directory: {OUTPUT}")
    print(f"Clinical rows: {len(clinical)}")
    print(f"Time-lag rows: {len(lag)}")
    print(f"Successfully audited prediction files: {len(audit)}")
    print(f"Errors or missing files: {len(error_df)}")
    print("\nClinical summary:")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    if not error_df.empty:
        print("\nErrors:")
        print(error_df.to_string(index=False))


if __name__ == "__main__":
    main()
