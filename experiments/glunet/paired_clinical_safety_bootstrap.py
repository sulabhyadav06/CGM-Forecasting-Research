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


def _contiguous_segments(timestamps: pd.Series) -> list[np.ndarray]:
    """Return positional index arrays split wherever adjacent rows are not 5 min apart."""
    parsed = pd.to_datetime(timestamps, errors="coerce")
    if parsed.isna().any():
        raise ValueError("Could not parse all target_timestamp values")
    gap_minutes = parsed.diff().dt.total_seconds().to_numpy()[1:] / 60.0
    boundaries = np.flatnonzero(~np.isclose(gap_minutes, 5.0, atol=1e-3, rtol=0)) + 1
    edges = np.concatenate(([0], boundaries, [len(parsed)]))
    return [np.arange(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def _segment_aware_block_indices(
    segments: list[np.ndarray], block_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Resample each contiguous segment to its original length without crossing gaps.

    Each segment is resampled independently. If a segment is shorter than the
    requested block size, its entire segment is used as the maximum block.
    """
    if block_size < 1:
        raise ValueError("block size must be positive")
    sampled_segments = []
    for segment in segments:
        n_segment = len(segment)
        effective_block = min(block_size, n_segment)
        pieces = []
        total = 0
        while total < n_segment:
            start = int(rng.integers(0, n_segment - effective_block + 1))
            pieces.append(segment[start:start + effective_block])
            total += effective_block
        sampled_segments.append(np.concatenate(pieces)[:n_segment])
    return np.concatenate(sampled_segments)


def _bootstrap_comparison(
    actual: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    block_size: int,
    n_boot: int,
    seed: int,
    candidate_label: str,
    segments: list[np.ndarray],
) -> list[dict]:
    n = len(actual)
    rng = np.random.default_rng(seed)
    point_base = _metrics(actual, baseline)
    point_candidate = _metrics(actual, candidate)
    metric_names = list(point_base)
    samples = {name: [] for name in metric_names}

    for _ in range(n_boot):
        idx = _segment_aware_block_indices(segments, block_size, rng)
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

    # Split at missing-observation gaps. Bootstrap blocks are sampled only
    # inside each continuous 5-minute segment and never bridge a discontinuity.
    segments = _contiguous_segments(anchor["target_timestamp"])
    segment_lengths = np.asarray([len(s) for s in segments], dtype=int)
    print(
        f"Continuous 5-minute segments: {len(segments)}; "
        f"length min/median/max={segment_lengths.min()}/"
        f"{np.median(segment_lengths):.0f}/{segment_lengths.max()} rows; "
        f"segments shorter than 24 rows={int(np.sum(segment_lengths < 24))}"
    )

    baseline = loaded["original_unweighted"]
    rows = []
    for label in ("alpha_025", "alpha_050", "alpha_060", "alpha_075"):
        for block_size in args.block_sizes:
            rows.extend(_bootstrap_comparison(
                actual, baseline, loaded[label], block_size,
                args.n_bootstrap, args.seed + block_size, label, segments,
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
        f"requested block sizes={args.block_sizes} rows (5 minutes per row).\n"
        f"Continuous 5-minute segments={len(segments)}; segment lengths "
        f"min/median/max={segment_lengths.min()}/{np.median(segment_lengths):.0f}/"
        f"{segment_lengths.max()} rows; segments shorter than 24 rows="
        f"{int(np.sum(segment_lengths < 24))}.\n"
        "Segments are resampled independently; no sampled block crosses a "
        "timestamp discontinuity. For segments shorter than the requested "
        "block size, the whole segment is the effective block.\n"
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
