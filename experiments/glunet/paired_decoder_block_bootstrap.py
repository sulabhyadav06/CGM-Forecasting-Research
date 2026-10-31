"""Paired moving-block bootstrap for saved GluNet decoder forecasts.

Evaluation only. All decoder mappings are fitted on the chronological training
segment; test labels are used only for paired scoring. Overlapping test windows
are handled with contiguous moving-block resampling rather than IID bootstrap.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.glunet.analyze_train_only_class_decoders import (
    _load_test,
    _train_representatives,
)

DEFAULT_RUNS = {
    "original_unweighted": "output/glunet_collapse_diagnostic/candidate_20261029_123854",
    "alpha_025": "output/glunet_weight_power_025/candidate_20261030_100153",
    "alpha_050": "output/glunet_weight_power_050/candidate_20261030_100339",
    "alpha_060": "output/glunet_weight_power_060/candidate_20261030_100851",
    "alpha_075": "output/glunet_weight_power_075/candidate_20261030_100549",
}

DECODERS = (
    "expected_fixed_codebook",
    "expected_train_conditional_mean",
    "expected_train_conditional_median",
)


def _metric_arrays(actual: np.ndarray, forecast: np.ndarray) -> dict[str, np.ndarray]:
    err = np.asarray(forecast, float) - np.asarray(actual, float)
    denom = np.maximum(np.abs(actual), 1e-8)
    return {
        "MAE": np.abs(err),
        "squared_error": err ** 2,
        "MARD_pct": 100.0 * np.abs(err) / denom,
    }


def _aggregate(values: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, float]:
    return {
        "MAE_mg_dL": float(np.mean(values["MAE"][indices])),
        "RMSE_mg_dL": float(np.sqrt(np.mean(values["squared_error"][indices]))),
        "MARD_pct": float(np.mean(values["MARD_pct"][indices])),
    }


def moving_block_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Non-circular moving-block bootstrap indices, sampled to length n."""
    if n < 1 or block_size < 1:
        raise ValueError("n and block_size must be positive")
    starts = rng.integers(0, max(1, n - block_size + 1),
                          size=int(np.ceil(n / block_size)))
    blocks = [np.arange(s, min(s + block_size, n)) for s in starts]
    return np.concatenate(blocks)[:n]


def paired_block_bootstrap(
    actual: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    block_size: int,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    """CI for candidate-minus-reference metric differences on paired rows."""
    actual = np.asarray(actual, float)
    candidate = np.asarray(candidate, float)
    reference = np.asarray(reference, float)
    if not (len(actual) == len(candidate) == len(reference)):
        raise ValueError("Actual and paired forecasts must have equal lengths")
    cand_values = _metric_arrays(actual, candidate)
    ref_values = _metric_arrays(actual, reference)
    point_c = _aggregate(cand_values, np.arange(len(actual)))
    point_r = _aggregate(ref_values, np.arange(len(actual)))
    point_diff = {
        "MAE": point_c["MAE_mg_dL"] - point_r["MAE_mg_dL"],
        "RMSE": point_c["RMSE_mg_dL"] - point_r["RMSE_mg_dL"],
        "MARD": point_c["MARD_pct"] - point_r["MARD_pct"],
    }
    rng = np.random.default_rng(seed)
    draws = {k: np.empty(n_bootstrap, dtype=float) for k in point_diff}
    for b in range(n_bootstrap):
        idx = moving_block_indices(len(actual), block_size, rng)
        c = _aggregate(cand_values, idx)
        r = _aggregate(ref_values, idx)
        draws["MAE"][b] = c["MAE_mg_dL"] - r["MAE_mg_dL"]
        draws["RMSE"][b] = c["RMSE_mg_dL"] - r["RMSE_mg_dL"]
        draws["MARD"][b] = c["MARD_pct"] - r["MARD_pct"]
    result = {}
    for metric, values in draws.items():
        lo, hi = np.quantile(values, [0.025, 0.975])
        result[f"{metric}_difference"] = point_diff[metric]
        result[f"{metric}_ci95_low"] = float(lo)
        result[f"{metric}_ci95_high"] = float(hi)
    return result


def parse_runs(items):
    if not items:
        return {k: Path(v).expanduser() for k, v in DEFAULT_RUNS.items()}
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError("--run must be LABEL=RUN_DIRECTORY")
        label, path = item.split("=", 1)
        if not label or label in result:
            raise ValueError(f"Empty or duplicate run label: {label!r}")
        result[label] = Path(path).expanduser()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient", default="540")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--data-root", default="~/Desktop/OhioT1DM_data_backup")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--output-dir", default="output/glunet_paired_decoder_bootstrap")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--blocks", type=int, nargs="+", default=[12, 24])
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--run", action="append", help="Optional LABEL=RUN_DIR; repeat to override defaults")
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least 100")
    if any(x < 1 for x in args.blocks):
        parser.error("All block sizes must be positive")

    runs = parse_runs(args.run)
    data_root = Path(args.data_root).expanduser().resolve()
    reps, mean_map, median_map, n_train = _train_representatives(
        data_root, args.patient, args.horizon, args.validation_fraction
    )
    fixed_map = reps["fixed_codebook_delta"].to_numpy(float)

    forecasts = {}
    anchor = None
    actual = current = None
    for label, run_dir in runs.items():
        df, prob, true_cls, pred_cls = _load_test(
            label, str(run_dir), args.patient, args.horizon
        )
        aligned = df[["target_timestamp", "current_glucose", "target_glucose", "delta_glucose"]]
        if anchor is None:
            anchor = aligned.copy()
            actual = df["target_glucose"].to_numpy(float)
            current = df["current_glucose"].to_numpy(float)
        elif not anchor.equals(aligned):
            raise ValueError(f"{label}: test timestamps/targets/order differ from other runs")
        forecasts[label] = {
            "expected_fixed_codebook": current + prob @ fixed_map,
            "expected_train_conditional_mean": current + prob @ mean_map,
            "expected_train_conditional_median": current + prob @ median_map,
        }

    if "original_unweighted" not in forecasts:
        raise ValueError("A run labelled original_unweighted is required as the reference")

    summary_rows, comparison_rows = [], []
    for label, decoder_forecasts in forecasts.items():
        for decoder, forecast in decoder_forecasts.items():
            vals = _metric_arrays(actual, forecast)
            summary_rows.append({
                "run": label, "decoder": decoder, "n": len(actual),
                "MAE_mg_dL": float(vals["MAE"].mean()),
                "RMSE_mg_dL": float(np.sqrt(vals["squared_error"].mean())),
                "MARD_pct": float(vals["MARD_pct"].mean()),
                "mean_signed_error_mg_dL": float(np.mean(forecast - actual)),
            })

    for label, decoder_forecasts in forecasts.items():
        if label == "original_unweighted":
            continue
        for decoder in DECODERS:
            for block_size in args.blocks:
                result = paired_block_bootstrap(
                    actual,
                    decoder_forecasts[decoder],
                    forecasts["original_unweighted"][decoder],
                    block_size,
                    args.bootstrap,
                    args.seed + block_size,
                )
                comparison_rows.append({
                    "candidate": label,
                    "reference": "original_unweighted",
                    "decoder": decoder,
                    "block_size": block_size,
                    "bootstrap_replicates": args.bootstrap,
                    **result,
                })

    output = Path(args.output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(output / "forecast_metric_summary.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(output / "paired_block_bootstrap_vs_original.csv", index=False)
    reps.to_csv(output / "training_only_class_representatives.csv", index=False)
    (output / "README.txt").write_text(
        "Paired moving-block bootstrap for saved GluNet predictions\n"
        f"Patient={args.patient}; horizon={args.horizon}; n={len(actual)}; "
        f"training windows for decoder mappings={n_train}.\n"
        f"Bootstrap replicates={args.bootstrap}; block sizes={args.blocks}; seed={args.seed}.\n"
        "Every model is evaluated on identical target timestamps, current glucose and targets. "
        "Class-conditional mappings are estimated from the chronological training segment only.\n"
        "Differences are candidate minus original_unweighted using the same decoder. "
        "Negative differences favor the candidate. Confidence intervals are percentile "
        "moving-block bootstrap intervals; they quantify within-this-test-sequence uncertainty, "
        "not between-patient generalization. Block sizes 12 and 24 are sensitivity analyses.\n"
        "No model is retrained and no validation/test labels are used to fit decoder mappings.\n"
    )
    print(f"Training windows used for mappings: {n_train}")
    print(f"Test samples: {len(actual)}")
    print("\nFORECAST METRICS")
    print(pd.DataFrame(summary_rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nPAIRED BLOCK BOOTSTRAP: candidate minus original_unweighted")
    print(pd.DataFrame(comparison_rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved reports to {output.resolve()}")


if __name__ == "__main__":
    main()
