"""Measure decoder error and train-only class-conditional reconstruction.

This is an evaluation-only analysis. Class representative values are estimated
from the training partition only; test labels are used only for final scoring.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.glunet.glunet_data import quantize_delta, recover_delta
from experiments.glunet.train_glunet_candidate_all_patients import (
    apply_preprocessor, fit_preprocessor, load_partition, make_sequences,
)

RUNS = {
    "alpha_025": "output/glunet_weight_power_025/candidate_20261030_100153",
    "alpha_050": "output/glunet_weight_power_050/candidate_20261030_100339",
    "alpha_060": "output/glunet_weight_power_060/candidate_20261030_100851",
    "alpha_075": "output/glunet_weight_power_075/candidate_20261030_100549",
}
DECODERS = {
    "argmax": "prediction_argmax_class_value",
    "expected": "prediction_expected_class_value",
}


def _score(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    return {
        "n": int(len(y)),
        "MAE_mg_dL": float(np.mean(np.abs(e))),
        "RMSE_mg_dL": float(np.sqrt(np.mean(e ** 2))),
        "mean_signed_error_mg_dL": float(np.mean(e)),
    }


def _load_test(label, run_dir, patient, horizon):
    folder = Path(run_dir).expanduser() / f"patient{patient}_{horizon}min"
    csv_path, npz_path = folder / "test_predictions_and_diagnostics.csv", folder / "test_probabilities.npz"
    if not csv_path.exists() or not npz_path.exists():
        raise FileNotFoundError(f"{label}: expected prediction CSV and probability NPZ in {folder}")
    df = pd.read_csv(csv_path)
    z = np.load(npz_path)
    prob = z["probabilities"]
    y_cls = z["true_classes"].astype(int)
    pred_cls = z["predicted_classes"].astype(int)
    if prob.shape != (len(df), 256) or len(y_cls) != len(df):
        raise ValueError(f"{label}: inconsistent test artifact dimensions")
    if not np.array_equal(y_cls, quantize_delta(df["delta_glucose"].to_numpy(float))):
        raise ValueError(f"{label}: true classes disagree with canonical quantizer")
    if not np.array_equal(pred_cls, prob.argmax(axis=1)):
        raise ValueError(f"{label}: saved predictions disagree with probability argmax")
    return df, prob, y_cls, pred_cls


def _train_representatives(data_root, patient, horizon, validation_fraction):
    raw = load_partition(data_root, patient, "training")
    cut = int(len(raw) * (1.0 - validation_fraction))
    train_segment = raw.iloc[:cut].copy()
    prep = fit_preprocessor(train_segment)
    train_segment = apply_preprocessor(train_segment, prep)
    _, y_train, meta = make_sequences(train_segment, horizon)
    if len(y_train) == 0:
        raise ValueError("No training sequences available for class representatives")
    delta_train = meta["delta_glucose"].to_numpy(float)
    if not np.array_equal(y_train, quantize_delta(delta_train)):
        raise ValueError("Training labels disagree with canonical quantizer")
    counts = np.bincount(y_train, minlength=256)
    means = np.full(256, np.nan)
    medians = np.full(256, np.nan)
    for cls in np.flatnonzero(counts):
        values = delta_train[y_train == cls]
        means[cls] = float(np.mean(values))
        medians[cls] = float(np.median(values))
    global_mean = float(np.mean(delta_train))
    global_median = float(np.median(delta_train))
    # Unseen classes receive the corresponding global training-only fallback.
    means_filled = np.where(np.isfinite(means), means, global_mean)
    medians_filled = np.where(np.isfinite(medians), medians, global_median)
    table = pd.DataFrame({
        "class_index": np.arange(256),
        "training_count": counts,
        "training_share_pct": 100 * counts / counts.sum(),
        "fixed_codebook_delta": recover_delta(np.arange(256)).astype(float),
        "train_conditional_mean_delta": means_filled,
        "train_conditional_median_delta": medians_filled,
        "used_global_mean_fallback": counts == 0,
        "used_global_median_fallback": counts == 0,
    })
    return table, means_filled, medians_filled, int(len(y_train))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--patient", default="540")
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--data-root", default="~/Desktop/OhioT1DM_data_backup")
    p.add_argument("--validation-fraction", type=float, default=0.2)
    p.add_argument("--output-dir", default="output/glunet_decoder_decomposition")
    p.add_argument("--run", action="append", help="Optional LABEL=RUN_DIR; repeat to override defaults")
    args = p.parse_args()
    if not 0.05 <= args.validation_fraction <= 0.4:
        p.error("validation-fraction must be between 0.05 and 0.4")
    runs = {}
    if args.run:
        for item in args.run:
            if "=" not in item:
                p.error("--run must be LABEL=PATH")
            label, path = item.split("=", 1)
            if not label or label in runs:
                p.error(f"empty or duplicate run label: {label!r}")
            runs[label] = path
    else:
        runs = RUNS

    data_root = Path(args.data_root).expanduser().resolve()
    reps, mean_map, median_map, n_train = _train_representatives(
        data_root, args.patient, args.horizon, args.validation_fraction
    )
    loaded = {}
    anchor = None
    for label, run_dir in runs.items():
        df, prob, y_cls, pred_cls = _load_test(label, run_dir, args.patient, args.horizon)
        alignment = df[["target_timestamp", "current_glucose", "target_glucose", "delta_glucose"]]
        if anchor is None:
            anchor = alignment.copy()
        elif not anchor.equals(alignment):
            raise ValueError(f"{label}: test rows/order differ from the first run")
        loaded[label] = (df, prob, y_cls, pred_cls)
    reps_out = Path(args.output_dir).expanduser()
    reps_out.mkdir(parents=True, exist_ok=True)
    reps.to_csv(reps_out / "train_only_class_representatives.csv", index=False)

    actual_delta = next(iter(loaded.values()))[0]["delta_glucose"].to_numpy(float)
    actual_glucose = next(iter(loaded.values()))[0]["target_glucose"].to_numpy(float)
    current = next(iter(loaded.values()))[0]["current_glucose"].to_numpy(float)
    true_cls = next(iter(loaded.values()))[2]
    true_fixed = recover_delta(true_cls).astype(float)
    rows, decomposition = [], []
    for label, (df, prob, y_cls, pred_cls) in loaded.items():
        argmax_fixed = recover_delta(pred_cls).astype(float)
        expected_fixed = prob @ recover_delta(np.arange(256)).astype(float)
        for decoder, model_delta in [("argmax", argmax_fixed), ("expected", expected_fixed)]:
            decomposition.append({
                "run": label, "decoder": decoder,
                "n": len(actual_delta),
                "true_target_delta_mae_vs_zero": float(np.mean(np.abs(actual_delta))),
                "target_quantization_floor_mae": float(np.mean(np.abs(true_fixed - actual_delta))),
                "model_forecast_delta_mae": float(np.mean(np.abs(model_delta - actual_delta))),
                "model_decoded_delta_vs_true_class_representative_mae": float(np.mean(np.abs(model_delta - true_fixed))),
                "model_decoded_delta_vs_true_class_representative_bias": float(np.mean(model_delta - true_fixed)),
                "target_quantization_floor_signed_bias": float(np.mean(true_fixed - actual_delta)),
                "mean_endpoint_probability_mass_pct": float(100 * np.mean(prob[:, 0] + prob[:, 255])),
            })
        # Decoder comparison using train-only conditional mean and median mappings.
        decoders = {
            "argmax_fixed_codebook": argmax_fixed,
            "expected_fixed_codebook": expected_fixed,
            "argmax_train_conditional_mean": mean_map[pred_cls],
            "argmax_train_conditional_median": median_map[pred_cls],
            "expected_train_conditional_mean": prob @ mean_map,
            "expected_train_conditional_median": prob @ median_map,
        }
        for decoder, predicted_delta in decoders.items():
            predicted_glucose = current + predicted_delta
            rows.append({
                "run": label, "decoder": decoder,
                **_score(actual_glucose, predicted_glucose),
                "delta_MAE_mg_dL": float(np.mean(np.abs(predicted_delta - actual_delta))),
                "delta_RMSE_mg_dL": float(np.sqrt(np.mean((predicted_delta - actual_delta) ** 2))),
            })
    pd.DataFrame(rows).to_csv(reps_out / "decoder_comparison_train_only_mappings.csv", index=False)
    pd.DataFrame(decomposition).to_csv(reps_out / "error_decomposition.csv", index=False)
    (reps_out / "README.txt").write_text(
        "Decoder decomposition and training-only class representatives\\n"
        f"Patient {args.patient}; horizon {args.horizon}; training windows used for mappings: {n_train}.\\n"
        "Class conditional means/medians are estimated only from the chronological training segment. "
        "The validation segment and test labels are not used to fit decoder mappings.\\n"
        "The conditional mean is the squared-error-optimal constant per class; the median is the "
        "absolute-error-optimal constant per class. Probability-weighted mappings average these "
        "training-only representatives using the saved test probabilities.\\n"
        "This is a post-hoc decoder experiment, not a new trained model. It does not alter saved runs.\\n"
    )
    print(f"Training windows used for class mappings: {n_train}")
    print("\\nDecoder comparison (test metrics)")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\\nError decomposition")
    print(pd.DataFrame(decomposition).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\\nSaved reports to {reps_out.resolve()}")


if __name__ == "__main__":
    main()
