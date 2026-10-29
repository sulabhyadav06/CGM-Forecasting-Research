"""Audit GluNet target saturation, split distributions, and non-neural baselines.

This is an analysis-only script: it does not train a model, modify source data,
or change the quantization scheme. It mirrors the candidate trainer's
chronological train/validation split and sequence builder.
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
from experiments.glunet.glunet_data import quantize_delta

N_CLASSES = 256
LOWER_SATURATION_THRESHOLD = -12.75
UPPER_SATURATION_THRESHOLD = 12.75
CLASS_CENTRES = np.arange(N_CLASSES, dtype=np.float64) / 10.0 - 12.7


def class_distribution(labels: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        raise ValueError("Cannot summarize an empty label array")
    counts = np.bincount(labels, minlength=N_CLASSES)
    p = counts / counts.sum()
    nonzero = p[p > 0]
    return {
        "n": int(labels.size),
        "unique_classes": int(np.count_nonzero(counts)),
        "dominant_class_zero_based": int(np.argmax(counts)),
        "dominant_class_share_pct": float(100 * counts.max() / labels.size),
        "endpoint_class_share_pct": float(100 * (counts[0] + counts[-1]) / labels.size),
        "lower_endpoint_share_pct": float(100 * counts[0] / labels.size),
        "upper_endpoint_share_pct": float(100 * counts[-1] / labels.size),
        "entropy_nats": float(-np.sum(nonzero * np.log(nonzero))),
        "counts": counts.tolist(),
    }


def js_divergence(p_counts: np.ndarray, q_counts: np.ndarray) -> float:
    """Jensen-Shannon divergence in nats between two nonnegative histograms."""
    p = np.asarray(p_counts, dtype=np.float64)
    q = np.asarray(q_counts, dtype=np.float64)
    p = p / max(p.sum(), 1.0)
    q = q / max(q.sum(), 1.0)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def glucose_metrics(actual: np.ndarray, forecast: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=np.float64)
    forecast = np.asarray(forecast, dtype=np.float64)
    err = forecast - actual
    denom = np.maximum(np.abs(actual), 1e-8)
    return {
        "MAE_mg_dL": float(np.mean(np.abs(err))),
        "RMSE_mg_dL": float(np.sqrt(np.mean(err**2))),
        "mean_signed_error_mg_dL": float(np.mean(err)),
        "MARD_pct": float(100 * np.mean(np.abs(err) / denom)),
        "n": int(actual.size),
    }


def baseline_rows(train_meta: pd.DataFrame, eval_meta: pd.DataFrame,
                  patient: str, horizon: int, split: str) -> list[dict]:
    train_delta = train_meta["delta_glucose"].to_numpy(dtype=np.float64)
    current = eval_meta["current_glucose"].to_numpy(dtype=np.float64)
    actual = eval_meta["target_glucose"].to_numpy(dtype=np.float64)
    labels = quantize_delta(train_delta)
    counts = np.bincount(labels, minlength=N_CLASSES)
    mode_class = int(np.argmax(counts))
    expected_class_delta = float(np.sum(counts * CLASS_CENTRES) / counts.sum())
    forecasts = {
        "persistence_zero_delta": current,
        "training_mean_delta": current + float(np.mean(train_delta)),
        "training_median_delta": current + float(np.median(train_delta)),
        "training_class_prior_expected_delta": current + expected_class_delta,
        "training_class_prior_mode_delta": current + CLASS_CENTRES[mode_class],
    }
    rows = []
    for name, forecast in forecasts.items():
        row = {"patient": patient, "horizon_minutes": horizon,
               "evaluation_split": split, "baseline": name}
        row.update(glucose_metrics(actual, forecast))
        rows.append(row)
    return rows


def audit_patient(root: Path, patient: str, horizon: int,
                  validation_fraction: float) -> tuple[list[dict], list[dict], dict]:
    train_all = load_partition(root, patient, "training")
    test_df = load_partition(root, patient, "testing")
    cut = int(len(train_all) * (1.0 - validation_fraction))
    train_df = train_all.iloc[:cut].copy()
    val_df = train_all.iloc[cut:].copy()

    prep = fit_preprocessor(train_df)
    train_df, val_df, test_df = [
        apply_preprocessor(df, prep) for df in (train_df, val_df, test_df)
    ]
    splits = {}
    for name, df in (("train", train_df), ("validation", val_df), ("test", test_df)):
        _, labels, meta = make_sequences(df, horizon)
        if len(labels) == 0:
            raise ValueError(f"Patient {patient}, {name}, horizon {horizon}: no sequences")
        splits[name] = {"labels": labels, "meta": meta}

    train_counts = np.bincount(splits["train"]["labels"], minlength=N_CLASSES)
    distribution_rows = []
    summary = {}
    for name, values in splits.items():
        labels = values["labels"]
        meta = values["meta"]
        stats = class_distribution(labels)
        deltas = meta["delta_glucose"].to_numpy(dtype=np.float64)
        stats.update({
            "raw_delta_min_mg_dL": float(deltas.min()),
            "raw_delta_max_mg_dL": float(deltas.max()),
            "raw_delta_mean_mg_dL": float(deltas.mean()),
            "raw_delta_median_mg_dL": float(np.median(deltas)),
            "raw_delta_std_mg_dL": float(deltas.std()),
            "raw_delta_at_or_below_lower_saturation_threshold_pct": float(
                100 * np.mean(deltas <= LOWER_SATURATION_THRESHOLD)),
            "raw_delta_at_or_above_saturation_threshold_pct": float(
                100 * np.mean(deltas >= UPPER_SATURATION_THRESHOLD)),
            "JS_divergence_from_train_nats": (
                0.0 if name == "train" else js_divergence(
                    train_counts, np.bincount(labels, minlength=N_CLASSES))
            ),
        })
        summary[name] = stats
        for cls, count in enumerate(stats["counts"]):
            if count:
                distribution_rows.append({
                    "patient": patient, "horizon_minutes": horizon,
                    "split": name, "class_zero_based": cls,
                    "class_delta_centre_mg_dL": float(CLASS_CENTRES[cls]),
                    "count": int(count), "share_pct": float(100 * count / len(labels)),
                })

    baseline_results = []
    for split in ("validation", "test"):
        baseline_results.extend(baseline_rows(
            splits["train"]["meta"], splits[split]["meta"],
            patient, horizon, split,
        ))

    result = {
        "patient": patient,
        "horizon_minutes": horizon,
        "n_train_windows": len(splits["train"]["labels"]),
        "n_validation_windows": len(splits["validation"]["labels"]),
        "n_test_windows": len(splits["test"]["labels"]),
        "split_summaries": summary,
        "interpretation_notes": [
            "Endpoint class share is not identical to the fraction strictly outside the saturation thresholds: values inside the boundary bins also map to endpoint classes.",
            "Jensen-Shannon divergence compares each split's 256-class target distribution with the training distribution; it is descriptive, not a hypothesis test.",
            "Baselines are fitted using training-window deltas only. Validation and test data are not used to choose baseline parameters.",
            "This audit mirrors the current candidate preprocessing; it does not implement or certify fidelity to the paper's full preprocessing pipeline.",
        ],
    }
    return distribution_rows, baseline_results, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="~/Desktop/OhioT1DM_data_backup")
    parser.add_argument("--output-dir", default="output/glunet_target_audit")
    parser.add_argument("--patients", nargs="+", default=["540"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[30])
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()

    if any(p not in PATIENTS for p in args.patients):
        parser.error(f"patients must be selected from {PATIENTS}")
    if any(h <= 0 or h % 5 for h in args.horizons):
        parser.error("horizons must be positive multiples of 5")
    if not 0.05 <= args.validation_fraction <= 0.4:
        parser.error("validation-fraction must be between 0.05 and 0.4")

    root = Path(args.data_root).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    distributions, baselines, summaries = [], [], []
    for patient in args.patients:
        for horizon in args.horizons:
            d, b, s = audit_patient(root, patient, horizon, args.validation_fraction)
            distributions.extend(d)
            baselines.extend(b)
            summaries.append(s)
            print(
                f"{patient} h{horizon}: "
                f"endpoint share train/val/test = "
                f"{s['split_summaries']['train']['endpoint_class_share_pct']:.1f}%/"
                f"{s['split_summaries']['validation']['endpoint_class_share_pct']:.1f}%/"
                f"{s['split_summaries']['test']['endpoint_class_share_pct']:.1f}%; "
                f"unique classes train/val/test = "
                f"{s['split_summaries']['train']['unique_classes']}/"
                f"{s['split_summaries']['validation']['unique_classes']}/"
                f"{s['split_summaries']['test']['unique_classes']}",
                flush=True,
            )
    pd.DataFrame(distributions).to_csv(out / "target_class_distributions.csv", index=False)
    pd.DataFrame(baselines).to_csv(out / "baseline_metrics.csv", index=False)
    (out / "audit_summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"Audit complete. Results: {out}")


if __name__ == "__main__":
    main()
