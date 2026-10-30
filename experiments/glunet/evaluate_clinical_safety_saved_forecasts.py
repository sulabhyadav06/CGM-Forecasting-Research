"""Clinical safety evaluation for saved GluNet forecasts (patient/horizon).

Evaluation only: decoder class mappings are estimated on the chronological
training segment. Test targets are used only for clinical scoring.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from clinical_metrics import clarke_error_grid_distribution
from experiments.glunet.analyze_train_only_class_decoders import (
    _load_test,
    _train_representatives,
)

RUNS = {
    "original_unweighted": "output/glunet_collapse_diagnostic/candidate_20261029_123854",
    "alpha_025": "output/glunet_weight_power_025/candidate_20261030_100153",
    "alpha_050": "output/glunet_weight_power_050/candidate_20261030_100339",
    "alpha_060": "output/glunet_weight_power_060/candidate_20261030_100851",
    "alpha_075": "output/glunet_weight_power_075/candidate_20261030_100549",
}


def binary_metrics(actual_event: np.ndarray, predicted_event: np.ndarray) -> dict:
    actual_event = np.asarray(actual_event, dtype=bool)
    predicted_event = np.asarray(predicted_event, dtype=bool)
    tp = int(np.sum(actual_event & predicted_event))
    fp = int(np.sum(~actual_event & predicted_event))
    fn = int(np.sum(actual_event & ~predicted_event))
    tn = int(np.sum(~actual_event & ~predicted_event))
    div = lambda a, b: float(a / b) if b else float("nan")
    return {
        "actual_events": int(actual_event.sum()),
        "predicted_events": int(predicted_event.sum()),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "sensitivity_pct": 100 * div(tp, tp + fn),
        "precision_pct": 100 * div(tp, tp + fp),
        "specificity_pct": 100 * div(tn, tn + fp),
        "false_negative_rate_pct": 100 * div(fn, tp + fn),
        "false_positive_rate_pct": 100 * div(fp, tn + fp),
    }


def score(actual: np.ndarray, forecast: np.ndarray) -> dict:
    error = np.asarray(forecast, float) - np.asarray(actual, float)
    zones = clarke_error_grid_distribution(actual, forecast)
    return {
        "n": int(len(actual)),
        "MAE_mg_dL": float(np.mean(np.abs(error))),
        "RMSE_mg_dL": float(np.sqrt(np.mean(error ** 2))),
        "mean_signed_error_mg_dL": float(np.mean(error)),
        "CEGA_A_pct": zones["A"], "CEGA_B_pct": zones["B"],
        "CEGA_C_pct": zones["C"], "CEGA_D_pct": zones["D"],
        "CEGA_E_pct": zones["E"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient", default="540")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--data-root", default="~/Desktop/OhioT1DM_data_backup")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--output-dir", default="output/glunet_clinical_safety")
    parser.add_argument("--run", action="append", help="Optional LABEL=RUN_DIR; repeat to override defaults")
    args = parser.parse_args()
    runs = dict(RUNS)
    if args.run:
        runs = {}
        for item in args.run:
            if "=" not in item:
                parser.error("--run must be LABEL=RUN_DIR")
            label, path = item.split("=", 1)
            if not label or label in runs:
                parser.error(f"empty or duplicate run label: {label!r}")
            runs[label] = path

    reps, mean_map, median_map, n_train = _train_representatives(
        Path(args.data_root).expanduser().resolve(),
        args.patient, args.horizon, args.validation_fraction,
    )
    fixed_map = reps["fixed_codebook_delta"].to_numpy(float)
    loaded = {}
    anchor = None
    actual = current = delta = None

    for label, run_dir in runs.items():
        df, prob, true_cls, pred_cls = _load_test(
            label, run_dir, args.patient, args.horizon
        )
        aligned = df[["target_timestamp", "current_glucose", "target_glucose", "delta_glucose"]]
        if anchor is None:
            anchor = aligned.copy()
            actual = df["target_glucose"].to_numpy(float)
            current = df["current_glucose"].to_numpy(float)
            delta = df["delta_glucose"].to_numpy(float)
        elif not anchor.equals(aligned):
            raise ValueError(f"{label}: test rows/order differ from the other runs")

        decoders = {
            "expected_fixed_codebook": current + prob @ fixed_map,
            "expected_train_conditional_mean": current + prob @ mean_map,
            "expected_train_conditional_median": current + prob @ median_map,
        }
        loaded[label] = decoders

    metric_rows, glycemic_rows, dynamic_rows = [], [], []
    for label, decoders in loaded.items():
        for decoder, forecast in decoders.items():
            metric_rows.append({"run": label, "decoder": decoder, **score(actual, forecast)})
            # Clinical event classification at the forecast horizon. Thresholds
            # follow conventional CGM categories: <70 low, >180 high.
            for event_name, actual_event, predicted_event in [
                ("hypoglycemia_lt70", actual < 70, forecast < 70),
                ("hyperglycemia_gt180", actual > 180, forecast > 180),
            ]:
                glycemic_rows.append({
                    "run": label, "decoder": decoder, "event": event_name,
                    **binary_metrics(actual_event, predicted_event),
                })

            # Error strata use the TRUE target glucose category, not current glucose.
            categories = {
                "target_low_lt70": actual < 70,
                "target_in_range_70_180": (actual >= 70) & (actual <= 180),
                "target_high_gt180": actual > 180,
            }
            for group, mask in categories.items():
                if mask.any():
                    metric_rows.append({
                        "run": label, "decoder": decoder, "stratum": group,
                        **score(actual[mask], forecast[mask]),
                    })
            for group, mask in [
                ("rapid_fall_delta_lt_minus20", delta < -20),
                ("rapid_rise_delta_gt20", delta > 20),
                ("all_falling_delta", delta < 0),
                ("all_rising_delta", delta > 0),
            ]:
                if mask.any():
                    dynamic_rows.append({
                        "run": label, "decoder": decoder, "stratum": group,
                        **score(actual[mask], forecast[mask]),
                    })

    out = Path(args.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(out / "clinical_metrics_overall_and_glucose_strata.csv", index=False)
    pd.DataFrame(glycemic_rows).to_csv(out / "hypo_hyper_event_metrics.csv", index=False)
    pd.DataFrame(dynamic_rows).to_csv(out / "rapid_dynamics_metrics.csv", index=False)
    (out / "README.txt").write_text(
        "Clinical safety analysis of saved GluNet forecasts\n"
        f"Patient={args.patient}; horizon={args.horizon}; test n={len(actual)}; "
        f"training windows used for mappings={n_train}.\n"
        "Forecasts include fixed-codebook, training-conditional mean and "
        "training-conditional median probability expectations. All conditional "
        "mappings are estimated from the chronological training segment only.\n"
        "Clarke Error Grid zones are calculated pointwise using the repository's "
        "clinical_metrics.clarke_error_grid_distribution implementation.\n"
        "Hypoglycemia event classification uses target glucose <70 mg/dL; "
        "hyperglycemia uses target glucose >180 mg/dL. These are forecast-horizon "
        "classification diagnostics, not alarm-policy validation.\n"
        "Glucose strata use TRUE target glucose. Rapid dynamics are defined as "
        "actual target-minus-current delta below -20 or above +20 mg/dL.\n"
        "This is a single-patient, single-horizon retrospective test analysis and "
        "does not establish clinical safety or generalization.\n"
    )
    print(f"Training windows used for mappings: {n_train}; test samples: {len(actual)}")
    print("\nOVERALL AND TRUE-TARGET GLUCOSE-STRATIFIED METRICS")
    print(pd.DataFrame(metric_rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nHYPOGLYCEMIA / HYPERGLYCEMIA EVENT METRICS")
    print(pd.DataFrame(glycemic_rows).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nRAPID-DYNAMICS METRICS")
    print(pd.DataFrame(dynamic_rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved reports to {out.resolve()}")


if __name__ == "__main__":
    main()
