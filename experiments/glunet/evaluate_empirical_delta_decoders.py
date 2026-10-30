"""Evaluate train-derived glucose-delta decoders for saved GluNet class probabilities.

All class representatives are estimated from the training portion only. Validation
and test targets are never used to fit the decoder. This is a post-hoc decoder
comparison, not retraining or a change to the 256-class target definition.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.glunet.glunet_data import quantize_delta, recover_delta
from experiments.glunet.train_glunet_candidate_all_patients import (
    load_partition,
    make_sequences,
)

DEFAULT_RUNS = {
    "alpha_025": "output/glunet_weight_power_025/candidate_20261030_100153",
    "alpha_050": "output/glunet_weight_power_050/candidate_20261030_100339",
    "alpha_060": "output/glunet_weight_power_060/candidate_20261030_100851",
    "alpha_075": "output/glunet_weight_power_075/candidate_20261030_100549",
}


def parse_run(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=RUN_DIRECTORY")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Run label and directory must be non-empty")
    return label.strip(), Path(path).expanduser()


def metric_row(actual, forecast):
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    error = forecast - actual
    return {
        "n": int(len(actual)),
        "MAE_mg_dL": float(np.mean(np.abs(error))),
        "RMSE_mg_dL": float(np.sqrt(np.mean(error ** 2))),
        "MARD_pct": float(100 * np.mean(np.abs(error) / np.maximum(np.abs(actual), 1e-8))),
        "mean_signed_error_mg_dL": float(np.mean(error)),
    }


def build_training_codebook(run_dir, patient, horizon, data_root_override=None):
    config_path = run_dir / "experiment_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing run configuration: {config_path}")
    config = json.loads(config_path.read_text())
    data_root = Path(data_root_override or config.get("data_root", "~/Desktop/OhioT1DM_data_backup")).expanduser()
    validation_fraction = float(config.get("validation_fraction", 0.2))
    if not 0.05 <= validation_fraction <= 0.4:
        raise ValueError(f"Unexpected validation_fraction in {config_path}: {validation_fraction}")

    train_all = load_partition(data_root, str(patient), "training")
    cut = int(len(train_all) * (1.0 - validation_fraction))
    if cut <= 0:
        raise ValueError("Chronological training segment is empty")
    train_segment = train_all.iloc[:cut].copy()
    _, y_train, meta_train = make_sequences(train_segment, int(horizon))
    if len(y_train) == 0:
        raise ValueError("No training sequences available to construct decoder codebook")

    train_delta = meta_train["delta_glucose"].to_numpy(dtype=float)
    fixed_centers = recover_delta(np.arange(256, dtype=np.int64)).astype(float)
    means = fixed_centers.copy()
    medians = fixed_centers.copy()
    counts = np.bincount(y_train, minlength=256)
    for cls in np.flatnonzero(counts):
        values = train_delta[y_train == cls]
        means[cls] = float(np.mean(values))
        medians[cls] = float(np.median(values))

    codebook = pd.DataFrame({
        "class_index": np.arange(256),
        "training_count": counts,
        "fixed_representative_delta": fixed_centers,
        "training_mean_delta": means,
        "training_median_delta": medians,
        "training_class_observed": counts > 0,
    })
    return codebook, means, medians, fixed_centers, data_root, validation_fraction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient", default="540")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--data-root", default=None,
                        help="Optional override; otherwise each run's saved config is used")
    parser.add_argument("--output-dir", default="output/glunet_empirical_decoder_comparison")
    parser.add_argument("--run", action="append", type=parse_run,
                        help="Optional LABEL=RUN_DIR; repeat to override the four known defaults")
    args = parser.parse_args()

    runs = dict(args.run) if args.run else {
        label: Path(path) for label, path in DEFAULT_RUNS.items()
    }
    if len(runs) < 1:
        parser.error("At least one run is required")

    output = Path(args.output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    summary_rows, strata_rows, codebook_rows = [], [], []
    anchor = None
    shared_codebook = None

    for label, run_dir in runs.items():
        run_dir = Path(run_dir).expanduser()
        folder = run_dir / f"patient{args.patient}_{args.horizon}min"
        csv_path = folder / "test_predictions_and_diagnostics.csv"
        npz_path = folder / "test_probabilities.npz"
        if not csv_path.exists() or not npz_path.exists():
            raise FileNotFoundError(f"{label}: expected {csv_path} and {npz_path}")
        frame = pd.read_csv(csv_path)
        artifacts = np.load(npz_path)
        prob = artifacts["probabilities"].astype(float)
        true_class = artifacts["true_classes"].astype(int)
        pred_class = artifacts["predicted_classes"].astype(int)
        if prob.shape != (len(frame), 256) or len(true_class) != len(frame):
            raise ValueError(f"{label}: CSV and probability artifact dimensions differ")
        if not np.array_equal(prob.argmax(axis=1), pred_class):
            raise ValueError(f"{label}: probability argmax and saved predicted classes differ")
        canonical_true_class = quantize_delta(frame["delta_glucose"].to_numpy(dtype=float))
        if not np.array_equal(canonical_true_class, true_class):
            raise ValueError(f"{label}: saved true classes disagree with canonical quantization")

        alignment = frame[["target_timestamp", "current_glucose", "target_glucose", "delta_glucose"]]
        if anchor is None:
            anchor = alignment.copy()
        elif not anchor.equals(alignment):
            raise ValueError(f"{label}: test observations/order differ from first run")

        codebook, train_mean, train_median, fixed, data_root, val_frac = build_training_codebook(
            run_dir, args.patient, args.horizon, args.data_root
        )
        codebook["run"] = label
        codebook_rows.append(codebook)

        # Fixed centers reproduce the existing decoders; empirical representatives
        # use only training-label conditional means/medians.
        decoders = {
            "fixed_argmax": fixed[pred_class],
            "fixed_probability_expectation": prob @ fixed,
            "training_mean_argmax": train_mean[pred_class],
            "training_median_argmax": train_median[pred_class],
            "training_mean_probability_expectation": prob @ train_mean,
            "training_median_probability_expectation": prob @ train_median,
        }
        current = frame["current_glucose"].to_numpy(dtype=float)
        actual = frame["target_glucose"].to_numpy(dtype=float)
        delta = frame["delta_glucose"].to_numpy(dtype=float)
        true_endpoint = (true_class == 0) | (true_class == 255)
        direction_groups = {
            "all": np.ones(len(actual), dtype=bool),
            "true_endpoint": true_endpoint,
            "true_interior": ~true_endpoint,
            "falling_delta": delta < 0,
            "rising_delta": delta > 0,
            "rapid_fall_delta_lt_minus20": delta < -20,
            "rapid_rise_delta_gt20": delta > 20,
        }

        # Oracle decoders estimate the representational floor using training-only
        # class means/medians, with the true class supplied only for this diagnostic.
        oracle = {
            "oracle_fixed_true_class": current + fixed[true_class],
            "oracle_training_mean_true_class": current + train_mean[true_class],
            "oracle_training_median_true_class": current + train_median[true_class],
        }
        for name, forecast in oracle.items():
            summary_rows.append({
                "run": label, "decoder": name, "uses_true_class_oracle": True,
                **metric_row(actual, forecast),
            })

        for name, delta_forecast in decoders.items():
            forecast = current + delta_forecast
            summary_rows.append({
                "run": label, "decoder": name, "uses_true_class_oracle": False,
                **metric_row(actual, forecast),
            })
            for group, mask in direction_groups.items():
                if mask.any():
                    strata_rows.append({
                        "run": label, "decoder": name, "group": group,
                        **metric_row(actual[mask], forecast[mask]),
                    })

        # Persistence comparator.
        summary_rows.append({
            "run": label, "decoder": "persistence", "uses_true_class_oracle": False,
            **metric_row(actual, current),
        })

    codebooks = pd.concat(codebook_rows, ignore_index=True)
    codebooks.to_csv(output / "training_only_class_codebooks.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output / "decoder_forecast_metrics.csv", index=False)
    pd.DataFrame(strata_rows).to_csv(output / "decoder_stratified_metrics.csv", index=False)
    (output / "README.txt").write_text(
        "Train-only empirical decoder comparison\\n"
        f"Patient {args.patient}; horizon {args.horizon} minutes.\\n"
        "Class means and medians are fitted from the chronological training segment only, "
        "using the validation fraction recorded in each run's experiment_config.json.\\n"
        "The true-class oracle rows are diagnostic representation estimates, not deployable "
        "forecasts. They use the known test class only to estimate class-decoding error.\\n"
        "For unseen-in-training classes, fixed quantizer representatives are used as a fallback.\\n"
        "No model weights are changed and no test labels are used to fit the codebook.\\n"
    )
    result = pd.DataFrame(summary_rows)
    print(f"Saved empirical decoder comparison to: {output.resolve()}")
    print("\\nFORECAST METRICS")
    cols = ["run", "decoder", "uses_true_class_oracle", "MAE_mg_dL",
            "RMSE_mg_dL", "MARD_pct", "mean_signed_error_mg_dL"]
    print(result[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\\nORACLE REPRESENTATION COMPARISON")
    print(result[result["uses_true_class_oracle"]][cols].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
