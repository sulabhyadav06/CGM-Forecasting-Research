"""Consolidated paired evaluation of saved GluNet class-weighting ablations.

Reads existing checkpoints' prediction/probability files only. No retraining,
no test-informed fitting, and no mutation of experiment outputs.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from clinical_metrics import clarke_error_grid_distribution
from experiments.glunet.compare_candidate_baselines import metrics

DEFAULT_RUNS = {
    "alpha_025": "output/glunet_weight_power_025/candidate_20261030_100153",
    "alpha_050": "output/glunet_weight_power_050/candidate_20261030_100339",
    "alpha_060": "output/glunet_weight_power_060/candidate_20261030_100851",
    "alpha_075": "output/glunet_weight_power_075/candidate_20261030_100549",
}
KEYS = ["target_timestamp", "current_glucose", "target_glucose"]


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Each run must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Run label and path must be non-empty")
    return label.strip(), Path(path).expanduser()


def forecast_metrics(actual: np.ndarray, forecast: np.ndarray) -> dict:
    out = metrics(actual, forecast)
    out.update({f"CEGA_{k}_pct": v for k, v in
                clarke_error_grid_distribution(actual, forecast).items()})
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    out["hypo_sensitivity_pct"] = _event_scores(actual < 70, forecast < 70)["sensitivity_pct"]
    out["hypo_precision_pct"] = _event_scores(actual < 70, forecast < 70)["precision_pct"]
    out["hyper_sensitivity_pct"] = _event_scores(actual > 180, forecast > 180)["sensitivity_pct"]
    out["hyper_precision_pct"] = _event_scores(actual > 180, forecast > 180)["precision_pct"]
    return out


def _event_scores(true_event: np.ndarray, pred_event: np.ndarray) -> dict:
    true_event = np.asarray(true_event, dtype=bool)
    pred_event = np.asarray(pred_event, dtype=bool)
    tp = int(np.sum(true_event & pred_event))
    fp = int(np.sum(~true_event & pred_event))
    fn = int(np.sum(true_event & ~pred_event))
    precision = tp / (tp + fp) if tp + fp else np.nan
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision_pct": 100 * precision if np.isfinite(precision) else np.nan,
        "sensitivity_pct": 100 * sensitivity if np.isfinite(sensitivity) else np.nan,
    }


def moving_block_ci(actual: np.ndarray, base: np.ndarray, candidate: np.ndarray,
                    block_size: int, repetitions: int, seed: int) -> dict:
    """Paired non-circular moving-block bootstrap; candidate minus base."""
    rng = np.random.default_rng(seed)
    n = len(actual)
    n_blocks = int(np.ceil(n / block_size))
    samples = np.empty((repetitions, 4), dtype=float)
    for i in range(repetitions):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        a = metrics(actual[idx], base[idx])
        b = metrics(actual[idx], candidate[idx])
        samples[i] = [
            b["MAE_mg_dL"] - a["MAE_mg_dL"],
            b["RMSE_mg_dL"] - a["RMSE_mg_dL"],
            b["MARD_pct"] - a["MARD_pct"],
            b["mean_signed_error_mg_dL"] - a["mean_signed_error_mg_dL"],
        ]
    names = ["MAE", "RMSE", "MARD_pct", "mean_signed_error"]
    result = []
    for j, name in enumerate(names):
        lo, med, hi = np.percentile(samples[:, j], [2.5, 50, 97.5])
        result.append({
            "metric": name, "block_size": block_size,
            "median_difference_candidate_minus_base": med,
            "ci95_low": lo, "ci95_high": hi,
        })
    return result


def probability_diagnostics(prob: np.ndarray, true_classes: np.ndarray,
                            pred_classes: np.ndarray, label: str) -> dict:
    if prob.ndim != 2 or prob.shape[0] != len(true_classes):
        raise ValueError(f"{label}: probability array has incompatible shape {prob.shape}")
    if prob.shape[1] != 256:
        raise ValueError(f"{label}: expected 256 probability columns, got {prob.shape[1]}")
    if not np.isfinite(prob).all() or np.any(prob < -1e-7):
        raise ValueError(f"{label}: probabilities contain invalid values")
    row_sums = prob.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=2e-3):
        raise ValueError(f"{label}: probability rows do not sum to one")
    ranks = np.argsort(np.argsort(-prob, axis=1), axis=1)
    true_rank = ranks[np.arange(len(true_classes)), true_classes] + 1
    endpoint = (true_classes == 0) | (true_classes == 255)
    interior = ~endpoint
    pred_endpoint = (pred_classes == 0) | (pred_classes == 255)
    entropy = -np.sum(prob * np.log(np.maximum(prob, 1e-12)), axis=1)
    confidence = prob.max(axis=1)
    correct = (pred_classes == true_classes).astype(float)
    ece = 0.0
    for lo in np.linspace(0, 1, 16)[:-1]:
        hi = lo + 1 / 15
        mask = (confidence >= lo) & (confidence < hi if hi < 1 else confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    out = {
        "probability_entropy_mean": float(entropy.mean()),
        "max_probability_mean": float(confidence.mean()),
        "endpoint_probability_mass_mean_pct": float(100 * np.mean(prob[:, 0] + prob[:, 255])),
        "predicted_unique_classes": int(np.unique(pred_classes).size),
        "predicted_endpoint_share_pct": float(100 * pred_endpoint.mean()),
        "largest_predicted_class_share_pct": float(100 * pd.Series(pred_classes).value_counts(normalize=True).iloc[0]),
        "top1_accuracy_pct": float(100 * np.mean(true_rank <= 1)),
        "top5_accuracy_pct": float(100 * np.mean(true_rank <= 5)),
        "top10_accuracy_pct": float(100 * np.mean(true_rank <= 10)),
        "top20_accuracy_pct": float(100 * np.mean(true_rank <= 20)),
        "mean_true_class_probability": float(np.mean(prob[np.arange(len(true_classes)), true_classes])),
        "multiclass_brier_score": float(np.mean(np.sum((prob - np.eye(256)[true_classes]) ** 2, axis=1))),
        "top1_ece_15_bins": float(ece),
        "endpoint_top1_accuracy_pct": float(100 * np.mean(true_rank[endpoint] <= 1)) if endpoint.any() else np.nan,
        "endpoint_top5_accuracy_pct": float(100 * np.mean(true_rank[endpoint] <= 5)) if endpoint.any() else np.nan,
        "interior_top1_accuracy_pct": float(100 * np.mean(true_rank[interior] <= 1)) if interior.any() else np.nan,
        "interior_top5_accuracy_pct": float(100 * np.mean(true_rank[interior] <= 5)) if interior.any() else np.nan,
        "endpoint_mean_true_class_probability": float(np.mean(prob[np.arange(len(true_classes))[endpoint], true_classes[endpoint]])) if endpoint.any() else np.nan,
        "interior_mean_true_class_probability": float(np.mean(prob[np.arange(len(true_classes))[interior], true_classes[interior]])) if interior.any() else np.nan,
    }
    return out


def load_run(label: str, run_dir: Path, patient: str, horizon: int):
    folder = run_dir / f"patient{patient}_{horizon}min"
    pred_path = folder / "test_predictions_and_diagnostics.csv"
    prob_path = folder / "test_probabilities.npz"
    if not pred_path.exists() or not prob_path.exists():
        raise FileNotFoundError(f"{label}: expected both {pred_path} and {prob_path}")
    df = pd.read_csv(pred_path).sort_values("target_timestamp").reset_index(drop=True)
    required = set(KEYS + ["delta_glucose", "prediction_argmax_class_value",
                           "prediction_expected_class_value", "predicted_class_argmax_zero_based"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{label}: prediction CSV missing columns {missing}")
    z = np.load(prob_path)
    prob = z["probabilities"]
    y_class = z["true_classes"].astype(int)
    pred_class = z["predicted_classes"].astype(int)
    if len(df) != len(y_class) or len(df) != len(pred_class) or prob.shape[0] != len(df):
        raise ValueError(f"{label}: CSV/NPZ sample counts disagree")
    if not np.array_equal(pred_class, df["predicted_class_argmax_zero_based"].to_numpy(dtype=int)):
        raise ValueError(f"{label}: predicted classes disagree between CSV and NPZ")
    return df, prob, y_class, pred_class


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient", default="540")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--output-dir", default="output/glunet_weighting_comparison")
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run", action="append", type=parse_run,
                        help="Optional LABEL=RUN_DIR; can be supplied repeatedly to override defaults")
    args = parser.parse_args()
    if args.bootstrap_repetitions < 100:
        parser.error("bootstrap-repetitions must be at least 100")
    runs = dict(args.run) if args.run else DEFAULT_RUNS
    if len(runs) < 2:
        parser.error("At least two runs are required")

    loaded = {}
    reference = None
    for label, path in runs.items():
        df, prob, y_class, pred_class = load_run(label, Path(path), args.patient, args.horizon)
        if reference is None:
            reference = df[KEYS].copy()
        elif not reference.equals(df[KEYS]):
            raise ValueError(f"{label}: test observations do not match the first run exactly")
        loaded[label] = (df, prob, y_class, pred_class)

    actual = next(iter(loaded.values()))[0]["target_glucose"].to_numpy(dtype=float)
    summary_rows, probability_rows, strata_rows, forecast_arrays = [], [], [], {}
    for label, (df, prob, y_class, pred_class) in loaded.items():
        for decoder, col in [
            ("argmax", "prediction_argmax_class_value"),
            ("expected", "prediction_expected_class_value"),
        ]:
            forecast = df[col].to_numpy(dtype=float)
            forecast_arrays[(label, decoder)] = forecast
            summary_rows.append({
                "run": label, "decoder": decoder, **forecast_metrics(actual, forecast),
                "sample_wins_vs_persistence_pct": float(100 * np.mean(np.abs(forecast - actual) < np.abs(df["prediction_persistence"].to_numpy(dtype=float) - actual))),
            })
            for group, mask in [
                ("all", np.ones(len(actual), dtype=bool)),
                ("true_hypoglycemia_lt70", actual < 70),
                ("true_target_range_70_180", (actual >= 70) & (actual <= 180)),
                ("true_hyperglycemia_gt180", actual > 180),
            ]:
                if mask.any():
                    strata_rows.append({
                        "run": label, "decoder": decoder, "stratum": group,
                        "n": int(mask.sum()), **metrics(actual[mask], forecast[mask]),
                    })
        probability_rows.append({"run": label, **probability_diagnostics(prob, y_class, pred_class, label)})

    comparisons = []
    for decoder in ["argmax", "expected"]:
        for (base_label, _), (candidate_label, _) in combinations(
            [(label, forecast_arrays[(label, decoder)]) for label in loaded], 2
        ):
            base = forecast_arrays[(base_label, decoder)]
            candidate = forecast_arrays[(candidate_label, decoder)]
            wins = float(100 * np.mean(np.abs(candidate - actual) < np.abs(base - actual)))
            losses = float(100 * np.mean(np.abs(candidate - actual) > np.abs(base - actual)))
            ties = float(100 - wins - losses)
            for block_size in [12, 24]:
                for row in moving_block_ci(actual, base, candidate, block_size,
                                           args.bootstrap_repetitions, args.seed):
                    comparisons.append({
                        "decoder": decoder, "base_run": base_label,
                        "candidate_run": candidate_label, "candidate_sample_wins_pct": wins,
                        "candidate_sample_losses_pct": losses, "ties_pct": ties, **row,
                    })

    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(out / "forecast_metrics_and_clarke_grid.csv", index=False)
    pd.DataFrame(probability_rows).to_csv(out / "probability_and_class_diagnostics.csv", index=False)
    pd.DataFrame(strata_rows).to_csv(out / "glucose_stratified_metrics.csv", index=False)
    pd.DataFrame(comparisons).to_csv(out / "paired_block_bootstrap_comparisons.csv", index=False)
    (out / "README.txt").write_text(
        "Consolidated GluNet weighting evaluation\n"
        f"Patient: {args.patient}; horizon: {args.horizon} minutes\n"
        f"Runs: {', '.join(runs)}\n"
        f"Test windows: {len(actual)}\n"
        "Files: forecast_metrics_and_clarke_grid.csv, probability_and_class_diagnostics.csv, "
        "glucose_stratified_metrics.csv, paired_block_bootstrap_comparisons.csv\n"
        "Bootstrap differences are candidate minus base; negative is better for error metrics. "
        "Block bootstrap is a sensitivity analysis and not a substitute for external-patient validation.\n"
        "Hypo/hyper precision and sensitivity use pointwise forecast thresholds (<70 and >180 mg/dL), "
        "not event-level episode detection.\n"
    )

    print(f"Patient {args.patient} | horizon {args.horizon} min | {len(actual)} matched test windows")
    print("\nFORECAST METRICS + CLARKE ERROR GRID")
    print(pd.DataFrame(summary_rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nPROBABILITY / CLASS DIAGNOSTICS")
    print(pd.DataFrame(probability_rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nGLUCOSE-STRATIFIED METRICS")
    print(pd.DataFrame(strata_rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved four CSV reports and README to: {out}")


if __name__ == "__main__":
    main()
