"""Paired moving-block bootstrap for saved GluNet clinical-safety forecasts.

Evaluation only. Uses the same training-only conditional-mean decoder as
 evaluate_clinical_safety_saved_forecasts.py. Resamples paired chronological
 test rows in contiguous, non-circular blocks; no model is trained.
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


def _metrics(actual: np.ndarray, forecast: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    error = forecast - actual
    zones = clarke_error_grid_distribution(actual, forecast)
    low_true, low_pred = actual < 70, forecast < 70
    high_true, high_pred = actual > 180, forecast > 180

    def ratio(a: float, b: float) -> float:
        return float(a / b) if b else float("nan")

    tp_low = int(np.sum(low_true & low_pred))
    fp_low = int(np.sum(~low_true & low_pred))
    fn_low = int(np.sum(low_true & ~low_pred))
    tp_high = int(np.sum(high_true & high_pred))
    fp_high = int(np.sum(~high_true & high_pred))
    fn_high = int(np.sum(high_true & ~high_pred))
    return {
        "MAE_mg_dL": float(np.mean(np.abs(error))),
        "RMSE_mg_dL": float(np.sqrt(np.mean(error ** 2))),
        "MARD_pct": float(np.mean(np.abs(error) / np.maximum(np.abs(actual), 1e-6)) * 100),
        "mean_signed_error_mg_dL": float(np.mean(error)),
        **{f"CEGA_{z}_pct": float(zones[z]) for z in "ABCDE"},
        "hypo_sensitivity_pct": 100 * ratio(tp_low, tp_low + fn_low),
        "hypo_precision_pct": 100 * ratio(tp_low, tp_low + fp_low),
        "hypo_FNR_pct": 100 * ratio(fn_low, tp_low + fn_low),
        "hyper_sensitivity_pct": 100 * ratio(tp_high, tp_high + fn_high),
        "hyper_precision_pct": 100 * ratio(tp_high, tp_high + fp_high),
        "hyper_FNR_pct": 100 * ratio(fn_high, tp_high + fn_high),
    }


def _block_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample non-circular contiguous blocks until n paired rows are obtained."""
    if block_size < 1 or block_size > n:
        raise ValueError(f"block size must be between 1 and n={n}")
    pieces = []
    total = 0
    while total < n:
        start = int(rng.integers(0, n - block_size + 1))
        block = np.arange(start, start + block_size)
        pieces.append(block)
        total += block_size
    return np.concatenate(pieces)[:n]


def _bootstrap_comparison(
    actual: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    block_size: int,
    n_boot: int,
    seed: int,
    candidate_label: str,
) -> list[dict]:
    n = len(actual)
    rng = np.random.default_rng(seed)
    point_base = _metrics(actual, baseline)
    point_candidate = _metrics(actual, candidate)
    metric_names = list(point_base)
    samples = {name: [] for name in metric_names}

    for _ in range(n_boot):
        idx = _block_indices(n, block_size, rng)
        m_base = _metrics(actual[idx], baseline[idx])
        m_candidate = _metrics(actual[idx], candidate[idx])
        for name in metric_names:
            a, b = m_base[name], m_candidate[name]
            if np.isfinite(a) and np.isfinite(b):
                samples[name].append(b - a)

    rows = []
    for name in metric_names:
        values = np.asarray(samples[name], dtype=float)
        if len(values):
            low, high = np.percentile(values, [2.5, 97.5])
            mean = float(np.mean(values))
        else:
            low = high = mean = float("nan")
        rows.append({
            "comparison": f"{candidate_label}_vs_original",
            "block_size_samples": block_size,
            "block_duration_minutes": block_size * 5,
            "metric": name,
            "baseline_point": point_base[name],
            "candidate_point": point_candidate[name],
            "difference_candidate_minus_baseline": point_candidate[name] - point_base[name],
            "bootstrap_mean_difference": mean,
            "CI95_lower": float(low),
            "CI95_upper": float(high),
            "valid_bootstrap_replicates": int(len(values)),
            "requested_bootstrap_replicates": n_boot,
            "interpretation": (
                "lower_is_better" if name in {
                    "MAE_mg_dL", "RMSE_mg_dL", "MARD_pct",
                    "hypo_FNR_pct", "hyper_FNR_pct",
                } else "higher_is_better" if name in {
                    "hypo_sensitivity_pct", "hypo_precision_pct",
                    "hyper_sensitivity_pct", "hyper_precision_pct",
                    "CEGA_A_pct",
                } else "descriptive"
            ),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient", default="540")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--data-root", default="~/Desktop/OhioT1DM_data_backup")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--output-dir", default="output/glunet_clinical_safety_bootstrap")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[12, 24])
    args = parser.parse_args()
    if args.n_bootstrap < 100:
        parser.error("--n-bootstrap must be at least 100")
    if not 0.05 <= args.validation_fraction <= 0.4:
        parser.error("--validation-fraction must be between 0.05 and 0.4")

    _, mean_map, _, n_train = _train_representatives(
        Path(args.data_root).expanduser().resolve(),
        args.patient, args.horizon, args.validation_fraction,
    )
    loaded = {}
    anchor = None
    actual = current = None
    for label, run_dir in RUNS.items():
        df, prob, _, _ = _load_test(label, run_dir, args.patient, args.horizon)
        aligned = df[["target_timestamp", "current_glucose", "target_glucose", "delta_glucose"]]
        if anchor is None:
            anchor = aligned.copy()
            actual = df["target_glucose"].to_numpy(float)
            current = df["current_glucose"].to_numpy(float)
        elif not anchor.equals(aligned):
            raise ValueError(f"{label}: test rows/order differ from original run")
        loaded[label] = current + prob @ mean_map

    # Verify the test sequence is regular at 5-minute intervals. This is
    # necessary for interpreting block sizes as 60- and 120-minute blocks.
    timestamps = pd.to_datetime(anchor["target_timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Could not parse all target_timestamp values")
    gaps = timestamps.diff().dropna().dt.total_seconds().to_numpy() / 60
    if len(gaps) and not np.allclose(gaps, 5.0):
        unique_gaps = sorted(set(np.round(gaps, 3).tolist()))
        raise ValueError(
            "Test timestamps are not uniformly 5 minutes apart; "
            f"observed gaps (minutes): {unique_gaps[:20]}. Do not apply fixed-row blocks."
        )

    baseline = loaded["original_unweighted"]
    rows = []
    for label in ("alpha_025", "alpha_050", "alpha_060", "alpha_075"):
        for block_size in args.block_sizes:
            rows.extend(_bootstrap_comparison(
                actual, baseline, loaded[label], block_size,
                args.n_bootstrap, args.seed + block_size, label,
            ))

    out = Path(args.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(out / "paired_clinical_safety_bootstrap.csv", index=False)
    (out / "README.txt").write_text(
        "Paired moving-block bootstrap of saved GluNet forecasts\n"
        f"Patient={args.patient}; horizon={args.horizon}; n_test={len(actual)}; "
        f"training windows for decoder mapping={n_train}.\n"
        f"Bootstrap replicates per comparison/block size={args.n_bootstrap}; "
        f"block sizes={args.block_sizes} rows (5 minutes per row).\n"
        "All comparisons use the training-only conditional-mean decoder. "
        "The same paired test-row indices are applied to baseline and candidate.\n"
        "Intervals are percentile 95% intervals for candidate-minus-baseline "
        "differences. Lower is better for error metrics and false-negative rates; "
        "higher is better for sensitivity, precision and CEGA zone A.\n"
        "Bootstrap replicates with undefined event rates are omitted for that "
        "metric and the valid replicate count is reported. Non-circular blocks "
        "are sampled from the observed test sequence. Block sizes 12 and 24 "
        "are dependence sensitivity checks, not proof that all dependence is captured.\n"
        "This is a single-patient, single-horizon retrospective comparison. "
        "It does not establish clinical safety or generalization.\n"
    )
    print(f"Training windows for decoder mapping: {n_train}; test samples: {len(actual)}")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {(out / 'paired_clinical_safety_bootstrap.csv').resolve()}")


if __name__ == "__main__":
    main()
