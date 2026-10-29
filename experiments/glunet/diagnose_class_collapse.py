"""Audit GluNet target-class collapse against target distributions and simple baselines.

This is a read-only diagnostic: it does not train a model, alter labels, or
modify existing run artifacts. It reproduces the candidate trainer's split,
preprocessing fit, sequence construction, and quantization for each patient
and horizon, then compares target-class distributions and forecasting baselines.

Example:
python -m experiments.glunet.diagnose_class_collapse \
  --data-root "$HOME/Desktop/OhioT1DM_data_backup" \
  --output-dir "$HOME/Desktop/CGM-reproduction-test/output/glunet_label_audit" \
  --patients 540 --horizons 30 \
  --run-dir "$HOME/Desktop/CGM-reproduction-test/output/glunet_collapse_diagnostic/candidate_20261029_123854"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.glunet.train_glunet_candidate_all_patients import (
    PATIENTS,
    apply_preprocessor,
    fit_preprocessor,
    load_partition,
    make_sequences,
)

N_CLASSES = 256
CLASS_CENTRES = np.arange(N_CLASSES, dtype=np.float64) / 10.0 - 12.7


def distribution_summary(labels: np.ndarray) -> dict:
    """Return robust summary statistics for zero-based 256-class labels."""
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        raise ValueError("Cannot summarize an empty label array")
    counts = np.bincount(labels, minlength=N_CLASSES)
    probs = counts[counts > 0] / labels.size
    return {
        "n": int(labels.size),
        "unique_classes": int(np.count_nonzero(counts)),
        "endpoint_share_pct": float(100.0 * (counts[0] + counts[-1]) / labels.size),
        "class_0_share_pct": float(100.0 * counts[0] / labels.size),
        "class_255_share_pct": float(100.0 * counts[-1] / labels.size),
        "largest_class_share_pct": float(100.0 * counts.max() / labels.size),
        "entropy_nats": float(-np.sum(probs * np.log(probs))),
    }


def jensen_shannon_divergence(a: np.ndarray, b: np.ndarray) -> float:
    """Jensen-Shannon divergence (nats) between two categorical histograms."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.sum() <= 0 or b.sum() <= 0:
        raise ValueError("Histograms must have positive mass")
    p, q = a / a.sum(), b / b.sum()
    m = 0.5 * (p + q)

    def kl(x, y):
        mask = x > 0
        return float(np.sum(x[mask] * np.log(x[mask] / y[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def forecast_metrics(actual: np.ndarray, forecast: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=np.float64)
    forecast = np.asarray(forecast, dtype=np.float64)
    if actual.shape != forecast.shape or actual.size == 0:
        raise ValueError("actual and forecast must be nonempty arrays with equal shape")
    err = forecast - actual
    denom = np.maximum(np.abs(actual), 1e-8)
    return {
        "n": int(actual.size),
        "MAE_mg_dl": float(np.mean(np.abs(err))),
        "RMSE_mg_dl": float(np.sqrt(np.mean(err**2))),
        "MARD_pct": float(100.0 * np.mean(np.abs(err) / denom)),
        "mean_signed_error_mg_dl": float(np.mean(err)),
    }


def _split_arrays(root: Path, patient: str, horizon: int):
    """Build exactly the current candidate's chronological train/val/test labels."""
    train_all = load_partition(root, patient, "training")
    test_df = load_partition(root, patient, "testing")
    cut = int(len(train_all) * 0.8)
    train_df, val_df = train_all.iloc[:cut].copy(), train_all.iloc[cut:].copy()
    prep = fit_preprocessor(train_df)
    train_df, val_df, test_df = [
        apply_preprocessor(x, prep) for x in (train_df, val_df, test_df)
    ]
    Xtr, ytr, mtr = make_sequences(train_df, horizon)
    Xv, yv, mv = make_sequences(val_df, horizon)
    Xte, yte, mte = make_sequences(test_df, horizon)
    if min(len(ytr), len(yv), len(yte)) == 0:
        raise RuntimeError(
            f"Patient {patient}, horizon {horizon}: empty split; cannot audit"
        )
    return (ytr, mtr, yv, mv, yte, mte)


def audit_one(root: Path, patient: str, horizon: int, run_dir: Path | None):
    ytr, mtr, yv, mv, yte, mte = _split_arrays(root, patient, horizon)
    split_labels = {"train": ytr, "validation": yv, "test": yte}
    summary_rows, histograms = [], {}
    for split, labels in split_labels.items():
        summary_rows.append({
            "patient": patient, "horizon_minutes": horizon, "split": split,
            **distribution_summary(labels),
        })
        histograms[split] = np.bincount(labels, minlength=N_CLASSES)
    divergence_rows = [
        {
            "patient": patient, "horizon_minutes": horizon,
            "comparison": "train_vs_validation",
            "Jensen_Shannon_divergence_nats": jensen_shannon_divergence(
                histograms["train"], histograms["validation"]
            ),
        },
        {
            "patient": patient, "horizon_minutes": horizon,
            "comparison": "train_vs_test",
            "Jensen_Shannon_divergence_nats": jensen_shannon_divergence(
                histograms["train"], histograms["test"]
            ),
        },
        {
            "patient": patient, "horizon_minutes": horizon,
            "comparison": "validation_vs_test",
            "Jensen_Shannon_divergence_nats": jensen_shannon_divergence(
                histograms["validation"], histograms["test"]
            ),
        },
    ]
    class_rows = []
    for split, counts in histograms.items():
        for cls, count in enumerate(counts):
            class_rows.append({
                "patient": patient, "horizon_minutes": horizon,
                "split": split, "class_zero_based": cls,
                "delta_centre_mg_dl": float(CLASS_CENTRES[cls]),
                "count": int(count), "share_pct": float(100.0 * count / counts.sum()),
            })

    # Targets are raw glucose values preserved by the trainer as Glucose_raw.
    train_delta = mtr["delta_glucose"].to_numpy(dtype=np.float64)
    train_current = mtr["current_glucose"].to_numpy(dtype=np.float64)
    val_delta = mv["delta_glucose"].to_numpy(dtype=np.float64)
    test_delta = mte["delta_glucose"].to_numpy(dtype=np.float64)
    test_current = mte["current_glucose"].to_numpy(dtype=np.float64)
    test_actual = mte["target_glucose"].to_numpy(dtype=np.float64)

    # Distributional class-prior decoder: expectation over training label frequencies.
    prior_counts = np.bincount(ytr, minlength=N_CLASSES).astype(np.float64)
    prior_probs = prior_counts / prior_counts.sum()
    prior_expected_delta = float(prior_probs @ CLASS_CENTRES)
    modal_class_delta = float(CLASS_CENTRES[int(np.argmax(prior_counts))])
    baselines = {
        "persistence": np.zeros_like(test_current),
        "train_mean_delta": np.full_like(test_current, float(np.mean(train_delta))),
        "train_median_delta": np.full_like(test_current, float(np.median(train_delta))),
        "empirical_class_prior_expected_delta": np.full_like(test_current, prior_expected_delta),
        "empirical_most_common_class_delta": np.full_like(test_current, modal_class_delta),
    }
    baseline_rows = []
    for name, delta_forecast in baselines.items():
        row = {
            "patient": patient, "horizon_minutes": horizon, "model": name,
            "training_delta_mean_mg_dl": float(np.mean(train_delta)),
            "training_delta_median_mg_dl": float(np.median(train_delta)),
            "class_prior_expected_delta_mg_dl": prior_expected_delta,
            "most_common_class_delta_mg_dl": modal_class_delta,
            **forecast_metrics(test_actual, test_current + delta_forecast),
        }
        baseline_rows.append(row)

    # Flag how much of the raw target lies outside the representable class-centre range.
    clip_rows = []
    for split, meta in (("train", mtr), ("validation", mv), ("test", mte)):
        delta = meta["delta_glucose"].to_numpy(dtype=np.float64)
        labels = split_labels[split]
        clip_rows.append({
            "patient": patient, "horizon_minutes": horizon, "split": split,
            "raw_delta_below_lowest_centre_pct": float(100 * np.mean(delta < CLASS_CENTRES[0])),
            "raw_delta_above_highest_centre_pct": float(100 * np.mean(delta > CLASS_CENTRES[-1])),
            "raw_delta_outside_centre_range_pct": float(
                100 * np.mean((delta < CLASS_CENTRES[0]) | (delta > CLASS_CENTRES[-1]))
            ),
            "quantized_endpoint_labels_pct": float(
                100 * np.mean((labels == 0) | (labels == N_CLASSES - 1))
            ),
            "delta_min_mg_dl": float(np.min(delta)),
            "delta_max_mg_dl": float(np.max(delta)),
            "delta_mean_mg_dl": float(np.mean(delta)),
            "delta_median_mg_dl": float(np.median(delta)),
        })

    model_rows = []
    if run_dir is not None:
        artifact = run_dir / f"patient{patient}_{horizon}min" / "test_probabilities.npz"
        if artifact.exists():
            with np.load(artifact) as data:
                pred = np.asarray(data["predicted_classes"], dtype=np.int64)
                probs = np.asarray(data["probabilities"], dtype=np.float64)
            if len(pred) == len(yte):
                model_rows.append({
                    "patient": patient, "horizon_minutes": horizon,
                    "source": str(artifact), "n_test": len(pred),
                    "target_unique_classes": int(np.unique(yte).size),
                    "predicted_unique_classes": int(np.unique(pred).size),
                    "target_endpoint_share_pct": float(100 * np.mean((yte == 0) | (yte == 255))),
                    "predicted_endpoint_share_pct": float(100 * np.mean((pred == 0) | (pred == 255))),
                    "predicted_mode_share_pct": float(
                        100 * np.bincount(pred, minlength=N_CLASSES).max() / len(pred)
                    ),
                    "mean_probability_entropy_nats": float(
                        np.mean(-np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1))
                    ),
                    "mean_max_probability": float(np.mean(probs.max(axis=1))),
                    "test_label_alignment": "matches current reconstructed test sequence count",
                })
            else:
                model_rows.append({
                    "patient": patient, "horizon_minutes": horizon,
                    "source": str(artifact), "n_test": len(pred),
                    "target_unique_classes": int(np.unique(yte).size),
                    "predicted_unique_classes": int(np.unique(pred).size),
                    "test_label_alignment": (
                        f"WARNING: saved predictions n={len(pred)} != reconstructed labels n={len(yte)}; "
                        "do not compare directly"
                    ),
                })
        else:
            model_rows.append({
                "patient": patient, "horizon_minutes": horizon,
                "source": str(artifact), "test_label_alignment": "no model artifact supplied/found",
            })

    return summary_rows, divergence_rows, class_rows, baseline_rows, clip_rows, model_rows


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="~/Desktop/OhioT1DM_data_backup")
    p.add_argument("--output-dir", default="experiments/glunet/runs/class_collapse_audit")
    p.add_argument("--patients", nargs="+", default=["540"])
    p.add_argument("--horizons", nargs="+", type=int, default=[30])
    p.add_argument("--run-dir", default=None, help="Optional timestamped candidate run directory")
    args = p.parse_args()
    if any(patient not in PATIENTS for patient in args.patients):
        p.error(f"patients must be selected from {PATIENTS}")
    if any(h <= 0 or h % 5 for h in args.horizons):
        p.error("horizons must be positive multiples of 5 minutes")
    return args


def main():
    args = parse_args()
    root = Path(args.data_root).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else None
    summaries, divergences, classes, baselines, clips, models = [], [], [], [], [], []
    for patient in args.patients:
        for horizon in args.horizons:
            results = audit_one(root, patient, horizon, run_dir)
            summaries.extend(results[0])
            divergences.extend(results[1])
            classes.extend(results[2])
            baselines.extend(results[3])
            clips.extend(results[4])
            models.extend(results[5])
            print(
                f"patient={patient} horizon={horizon}: "
                f"train/val/test unique classes="
                f"{results[0][0]['unique_classes']}/"
                f"{results[0][1]['unique_classes']}/"
                f"{results[0][2]['unique_classes']}; "
                f"test endpoint labels={results[0][2]['endpoint_share_pct']:.1f}%",
                flush=True,
            )
    pd.DataFrame(summaries).to_csv(out / "label_distribution_summary.csv", index=False)
    pd.DataFrame(divergences).to_csv(out / "split_distribution_divergence.csv", index=False)
    pd.DataFrame(classes).to_csv(out / "class_distribution.csv", index=False)
    pd.DataFrame(baselines).to_csv(out / "baseline_metrics.csv", index=False)
    pd.DataFrame(clips).to_csv(out / "raw_delta_and_endpoint_audit.csv", index=False)
    pd.DataFrame(models).to_csv(out / "model_prediction_audit.csv", index=False)
    (out / "README.txt").write_text(
        "Read-only GluNet class-collapse audit. No model training or source data changes.\n"
        "Inspect label_distribution_summary.csv, split_distribution_divergence.csv, "
        "baseline_metrics.csv, raw_delta_and_endpoint_audit.csv, and model_prediction_audit.csv.\n"
        "A high endpoint-label share can arise from the paper's narrow 256-class delta range; "
        "do not widen the quantizer or rebalance classes without a separately justified ablation.\n"
        "This audit reproduces the current candidate preprocessing; it does not make that "
        "preprocessing paper-faithful. The paper describes outlier handling, CGM gap filling, "
        "and input alignment, which still require a separately verified implementation.\n"
    )
    print(f"Audit complete. Files written to: {out}", flush=True)


if __name__ == "__main__":
    main()
