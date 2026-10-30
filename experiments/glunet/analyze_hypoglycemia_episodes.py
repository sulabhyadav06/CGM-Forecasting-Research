"""Episode-level retrospective hypoglycemia analysis for saved GluNet forecasts.

No model training occurs. Forecasts use the training-only conditional-mean
class decoder, matching evaluate_clinical_safety_saved_forecasts.py.

Episodes are contiguous low-glucose observations, allowing a configurable
maximum gap (default sensitivity grid: 5, 10, 15 minutes). Missing-data gaps
larger than the episode gap split episodes. A predicted episode detects an
actual episode when their intervals overlap the actual episode interval
expanded backwards by a configurable anticipation window. Matching is
one-to-one, choosing eligible pairs with the smallest absolute start-time
difference first. These are retrospective definitions, not validated alarm
rules.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

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


def extract_episodes(
    timestamps: pd.Series | np.ndarray,
    event_mask: np.ndarray,
    gap_tolerance_minutes: float = 10.0,
) -> list[dict]:
    """Extract event episodes; split whenever event samples are too far apart."""
    ts = pd.to_datetime(timestamps, errors="coerce")
    if pd.isna(ts).any():
        raise ValueError("timestamps contain unparseable values")
    times = pd.DatetimeIndex(ts)
    mask = np.asarray(event_mask, dtype=bool)
    if mask.ndim != 1 or len(mask) != len(times):
        raise ValueError("event_mask must be one-dimensional and match timestamps")
    if gap_tolerance_minutes <= 0:
        raise ValueError("gap_tolerance_minutes must be positive")
    if not times.is_monotonic_increasing:
        raise ValueError("timestamps must be sorted in ascending order")
    event_positions = np.flatnonzero(mask)
    if not len(event_positions):
        return []

    episodes = []
    current = [int(event_positions[0])]
    for pos in event_positions[1:]:
        previous = current[-1]
        gap = (times[pos] - times[previous]).total_seconds() / 60.0
        if gap <= gap_tolerance_minutes:
            current.append(int(pos))
        else:
            episodes.append(_episode_record(len(episodes) + 1, current, times))
            current = [int(pos)]
    episodes.append(_episode_record(len(episodes) + 1, current, times))
    return episodes


def _episode_record(episode_id: int, positions: list[int], times: pd.DatetimeIndex) -> dict:
    start, end = times[positions[0]], times[positions[-1]]
    return {
        "episode_id": episode_id,
        "start": start,
        "end": end,
        "n_low_observations": len(positions),
        "observed_span_minutes": (end - start).total_seconds() / 60.0,
    }


def match_episodes(
    actual_episodes: list[dict],
    predicted_episodes: list[dict],
    anticipation_minutes: float = 0.0,
) -> tuple[list[dict], list[int], list[int]]:
    """Greedy one-to-one interval matching; return matches, missed IDs, false IDs."""
    if anticipation_minutes < 0:
        raise ValueError("anticipation_minutes must be non-negative")
    candidates = []
    for ai, actual in enumerate(actual_episodes):
        window_start = actual["start"] - pd.Timedelta(minutes=anticipation_minutes)
        for pi, predicted in enumerate(predicted_episodes):
            overlaps_detection_window = (
                predicted["end"] >= window_start and predicted["start"] <= actual["end"]
            )
            if overlaps_detection_window:
                delta = (predicted["start"] - actual["start"]).total_seconds() / 60.0
                candidates.append((abs(delta), ai, pi, delta))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    used_actual: set[int] = set()
    used_predicted: set[int] = set()
    matches = []
    for _, ai, pi, delay in candidates:
        if ai in used_actual or pi in used_predicted:
            continue
        used_actual.add(ai)
        used_predicted.add(pi)
        matches.append({
            "actual_episode_id": actual_episodes[ai]["episode_id"],
            "predicted_episode_id": predicted_episodes[pi]["episode_id"],
            "detection_delay_minutes": float(delay),
            "detected_early": bool(delay < 0),
        })

    missed = [
        ep["episode_id"] for i, ep in enumerate(actual_episodes)
        if i not in used_actual
    ]
    false = [
        ep["episode_id"] for i, ep in enumerate(predicted_episodes)
        if i not in used_predicted
    ]
    return matches, missed, false


def evaluate_episode_setting(
    timestamps: pd.Series | np.ndarray,
    actual_glucose: np.ndarray,
    predicted_glucose: np.ndarray,
    gap_tolerance_minutes: float,
    anticipation_minutes: float,
    monitored_hours: float,
) -> tuple[dict, list[dict], list[dict]]:
    actual_glucose = np.asarray(actual_glucose, dtype=float)
    predicted_glucose = np.asarray(predicted_glucose, dtype=float)
    if actual_glucose.shape != predicted_glucose.shape:
        raise ValueError("actual and predicted glucose arrays must have equal shape")
    if not np.isfinite(actual_glucose).all() or not np.isfinite(predicted_glucose).all():
        raise ValueError("glucose arrays must contain finite values")

    actual_eps = extract_episodes(
        timestamps, actual_glucose < 70, gap_tolerance_minutes
    )
    predicted_eps = extract_episodes(
        timestamps, predicted_glucose < 70, gap_tolerance_minutes
    )
    matches, missed, false = match_episodes(
        actual_eps, predicted_eps, anticipation_minutes
    )
    n_actual, n_pred = len(actual_eps), len(predicted_eps)
    n_detected = len(matches)
    sensitivity = n_detected / n_actual * 100 if n_actual else float("nan")
    precision = n_detected / n_pred * 100 if n_pred else float("nan")
    delays = [m["detection_delay_minutes"] for m in matches]
    summary = {
        "actual_episode_count": n_actual,
        "predicted_episode_count": n_pred,
        "detected_episode_count": n_detected,
        "missed_episode_count": len(missed),
        "false_predicted_episode_count": len(false),
        "episode_sensitivity_pct": sensitivity,
        "episode_precision_pct": precision,
        "mean_detection_delay_minutes": float(np.mean(delays)) if delays else float("nan"),
        "median_detection_delay_minutes": float(np.median(delays)) if delays else float("nan"),
        "early_detection_count": int(sum(d < 0 for d in delays)),
        "late_or_onset_detection_count": int(sum(d >= 0 for d in delays)),
        "false_alarms_per_24h": (
            len(false) / monitored_hours * 24 if monitored_hours > 0 else float("nan")
        ),
        "monitored_hours_sample_equivalent": float(monitored_hours),
    }
    match_rows = [
        {
            **m,
            "actual_episode_start": actual_eps[m["actual_episode_id"] - 1]["start"],
            "actual_episode_end": actual_eps[m["actual_episode_id"] - 1]["end"],
            "predicted_episode_start": predicted_eps[m["predicted_episode_id"] - 1]["start"],
            "predicted_episode_end": predicted_eps[m["predicted_episode_id"] - 1]["end"],
        }
        for m in matches
    ]
    episode_rows = [
        {"episode_type": "actual", **ep, "matched": ep["episode_id"] not in missed}
        for ep in actual_eps
    ] + [
        {"episode_type": "predicted", **ep, "matched": ep["episode_id"] not in false}
        for ep in predicted_eps
    ]
    return summary, match_rows, episode_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient", default="540")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--data-root", default="~/Desktop/OhioT1DM_data_backup")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--output-dir", default="output/glunet_hypoglycemia_episodes")
    parser.add_argument("--gap-tolerances", type=float, nargs="+", default=[5, 10, 15])
    parser.add_argument("--anticipation-windows", type=float, nargs="+", default=[0, 15, 30])
    args = parser.parse_args()
    if any(x <= 0 for x in args.gap_tolerances):
        parser.error("all gap tolerances must be positive")
    if any(x < 0 for x in args.anticipation_windows):
        parser.error("anticipation windows must be non-negative")

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

    timestamps = pd.to_datetime(anchor["target_timestamp"], errors="coerce")
    if timestamps.isna().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("test timestamps must be parseable and sorted")
    # Sample-equivalent monitoring duration avoids pretending long missing-data
    # gaps were monitored. One observed CGM row represents one 5-minute slot.
    monitored_hours = len(timestamps) * 5.0 / 60.0

    summary_rows, match_rows_all, episode_rows_all = [], [], []
    for gap in args.gap_tolerances:
        for anticipation in args.anticipation_windows:
            for label, forecast in loaded.items():
                summary, matches, episodes = evaluate_episode_setting(
                    timestamps, actual, forecast, gap, anticipation, monitored_hours
                )
                setting = {
                    "run": label,
                    "gap_tolerance_minutes": gap,
                    "anticipation_window_minutes": anticipation,
                }
                summary_rows.append({**setting, **summary})
                match_rows_all.extend([{**setting, **m} for m in matches])
                episode_rows_all.extend([{**setting, **ep} for ep in episodes])

    out = Path(args.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(out / "hypoglycemia_episode_summary.csv", index=False)
    pd.DataFrame(match_rows_all).to_csv(out / "hypoglycemia_episode_matches.csv", index=False)
    pd.DataFrame(episode_rows_all).to_csv(out / "hypoglycemia_episode_inventory.csv", index=False)
    (out / "README.txt").write_text(
        "Retrospective episode-level hypoglycemia analysis of saved forecasts\n"
        f"Patient={args.patient}; horizon={args.horizon}; test rows={len(timestamps)}; "
        f"training windows for decoder mapping={n_train}.\\n"
        "Forecasts use the training-only conditional-mean class decoder. "
        "No model training occurs.\\n"
        "Low glucose is defined as <70 mg/dL. Episodes group low observations "
        "when consecutive low timestamps are no more than the configured gap "
        "tolerance apart. The sensitivity grid varies gap tolerance and "
        "anticipation window.\\n"
        "One-to-one matching requires predicted-episode interval overlap with "
        "the actual episode interval expanded backwards by the anticipation "
        "window. Detection delay is predicted episode start minus actual "
        "episode start; negative means predicted early. Unmatched predicted "
        "episodes are false alarms; unmatched actual episodes are missed.\\n"
        "False alarms per 24 hours use sample-equivalent monitoring time "
        "(test rows x 5 minutes), excluding unobserved gaps. These definitions "
        "are analysis conventions, not validated clinical alarm criteria.\\n"
        "Single patient and horizon; retrospective results do not establish "
        "clinical safety, real-time alarm performance, or generalization.\\n"
    )
    print(f"Training windows for decoder mapping: {n_train}; test rows: {len(timestamps)}")
    print(f"Sample-equivalent monitoring duration: {monitored_hours:.2f} hours")
    print("\\nEPISODE-LEVEL SUMMARY")
    print(pd.DataFrame(summary_rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\\nSaved reports to {out.resolve()}")


if __name__ == "__main__":
    main()
