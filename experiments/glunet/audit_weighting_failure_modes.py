"""Audit quantization, probability mass, and temporal failure modes for saved runs.

This is post-hoc analysis of existing test predictions only. It does not train,
select checkpoints, or use test labels to fit model parameters.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.glunet.glunet_data import quantize_delta

DEFAULT_RUNS = {
    "alpha_025": "output/glunet_weight_power_025/candidate_20261030_100153",
    "alpha_050": "output/glunet_weight_power_050/candidate_20261030_100339",
    "alpha_060": "output/glunet_weight_power_060/candidate_20261030_100851",
    "alpha_075": "output/glunet_weight_power_075/candidate_20261030_100549",
}
DECODERS = {
    "argmax": "prediction_argmax_class_value",
    "expected": "prediction_expected_class_value",
}


def _metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    return {
        "n": len(y),
        "MAE": float(np.mean(np.abs(e))),
        "RMSE": float(np.sqrt(np.mean(e ** 2))),
        "mean_error_prediction_minus_target": float(np.mean(e)),
    }


def _load(label, root, patient, horizon):
    folder = Path(root).expanduser() / f"patient{patient}_{horizon}min"
    csv_path, npz_path = folder / "test_predictions_and_diagnostics.csv", folder / "test_probabilities.npz"
    if not csv_path.exists() or not npz_path.exists():
        raise FileNotFoundError(f"{label}: expected {csv_path} and {npz_path}")
    df = pd.read_csv(csv_path)
    required = {"target_timestamp", "current_glucose", "target_glucose", "delta_glucose",
                "predicted_class_argmax_zero_based", *DECODERS.values()}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{label}: missing CSV columns {sorted(missing)}")
    z = np.load(npz_path)
    prob, true_cls, pred_cls = z["probabilities"], z["true_classes"].astype(int), z["predicted_classes"].astype(int)
    if len(df) != len(true_cls) or prob.shape != (len(df), 256):
        raise ValueError(f"{label}: incompatible CSV/NPZ dimensions")
    if not np.array_equal(pred_cls, df["predicted_class_argmax_zero_based"].to_numpy(int)):
        raise ValueError(f"{label}: predicted classes differ between CSV and NPZ")
    if not np.array_equal(prob.argmax(axis=1), pred_cls):
        raise ValueError(f"{label}: probability argmax differs from saved class predictions")
    df["target_timestamp_parsed"] = pd.to_datetime(df["target_timestamp"], errors="coerce", utc=True)
    if df["target_timestamp_parsed"].isna().any():
        raise ValueError(f"{label}: target_timestamp contains unparseable values")
    df = df.assign(_true_class=true_cls, _pred_class=pred_cls)
    return df, prob


def _summarize_group(df, forecast_col, label, decoder, group_name, mask):
    if not np.any(mask):
        return None
    y = df["target_glucose"].to_numpy(float)[mask]
    p = df[forecast_col].to_numpy(float)[mask]
    return {"run": label, "decoder": decoder, "group": group_name, **_metrics(y, p)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patient", default="540")
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--output-dir", default="output/glunet_failure_mechanism_audit")
    ap.add_argument("--run", action="append", help="Optional LABEL=RUN_DIR; repeat to override defaults")
    args = ap.parse_args()
    runs = {}
    if args.run:
        for item in args.run:
            if "=" not in item:
                ap.error("--run must be LABEL=PATH")
            label, path = item.split("=", 1)
            if not label or label in runs:
                ap.error(f"empty or duplicate run label: {label!r}")
            runs[label] = path
    else:
        runs = DEFAULT_RUNS

    loaded = {}
    anchor = None
    for label, path in runs.items():
        df, prob = _load(label, path, args.patient, args.horizon)
        align = df[["target_timestamp", "current_glucose", "target_glucose", "delta_glucose"]]
        if anchor is None:
            anchor = align.copy()
        elif not anchor.equals(align):
            raise ValueError(f"{label}: test rows/order differ from first run")
        loaded[label] = (df, prob)

    base_df = next(iter(loaded.values()))[0]
    delta = base_df["delta_glucose"].to_numpy(float)
    true_cls = base_df["_true_class"].to_numpy(int)
    if not np.array_equal(true_cls, quantize_delta(delta)):
        raise ValueError("saved true classes disagree with the repository canonical quantizer")
    current = base_df["current_glucose"].to_numpy(float)
    target = base_df["target_glucose"].to_numpy(float)
    if not np.allclose(current + delta, target, atol=1e-4, rtol=0):
        raise ValueError("delta_glucose is inconsistent with target_glucose - current_glucose")

    out = Path(args.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    # Canonical class occupancy and endpoint saturation.
    counts = np.bincount(true_cls, minlength=256)
    quant_rows = [{
        "class_index": i, "sample_count": int(counts[i]),
        "sample_share_pct": float(100 * counts[i] / len(true_cls)),
        "endpoint_class": bool(i in (0, 255)),
    } for i in range(256)]
    pd.DataFrame(quant_rows).to_csv(out / "true_quantized_class_occupancy.csv", index=False)

    # Understand the continuous delta distribution before class weighting.
    delta_bins = [-np.inf, -40, -20, -10, -5, 0, 5, 10, 20, 40, np.inf]
    delta_labels = ["<=-40", "-40:-20", "-20:-10", "-10:-5", "-5:0", "0:5",
                    "5:10", "10:20", "20:40", ">40"]
    delta_group = pd.cut(delta, bins=delta_bins, labels=delta_labels, right=False)
    delta_df = pd.DataFrame({"delta_glucose": delta, "delta_bin": delta_group.astype(str),
                             "true_class": true_cls})
    delta_df.groupby(["delta_bin", "true_class"], observed=False).size().rename("count").reset_index().to_csv(
        out / "continuous_delta_to_class_mapping.csv", index=False
    )
    pd.DataFrame([{
        "n": len(delta), "delta_min": float(np.min(delta)), "delta_max": float(np.max(delta)),
        "delta_mean": float(np.mean(delta)), "delta_sd": float(np.std(delta)),
        "delta_median": float(np.median(delta)),
        "negative_delta_share_pct": float(100 * np.mean(delta < 0)),
        "zero_delta_share_pct": float(100 * np.mean(delta == 0)),
        "positive_delta_share_pct": float(100 * np.mean(delta > 0)),
        "true_endpoint_class_share_pct": float(100 * np.mean((true_cls == 0) | (true_cls == 255))),
    }]).to_csv(out / "quantization_summary.csv", index=False)

    # Temporal dynamics and model residuals by direction and magnitude of observed change.
    magnitude_bins = [-np.inf, 5, 10, 20, np.inf]
    magnitude_labels = ["abs_delta_0_5", "abs_delta_5_10", "abs_delta_10_20", "abs_delta_gt20"]
    dynamics = pd.cut(np.abs(delta), bins=magnitude_bins, labels=magnitude_labels, right=False)
    direction = np.where(delta < 0, "falling", np.where(delta > 0, "rising", "unchanged"))
    temporal_rows, probability_rows = [], []
    for label, (df, prob) in loaded.items():
        for decoder, col in DECODERS.items():
            for group in magnitude_labels:
                mask = np.asarray(dynamics == group)
                row = _summarize_group(df, col, label, decoder, group, mask)
                if row:
                    temporal_rows.append(row)
            for group in ["falling", "unchanged", "rising"]:
                mask = direction == group
                row = _summarize_group(df, col, label, decoder, group, mask)
                if row:
                    temporal_rows.append(row)
            # Joint direction/magnitude groups expose rapid rise/fall behavior.
            for group in ["rapid_fall", "rapid_rise"]:
                mask = ((delta < -20) if group == "rapid_fall" else (delta > 20))
                row = _summarize_group(df, col, label, decoder, group, mask)
                if row:
                    temporal_rows.append(row)

        # Endpoint mass conditioned on the true class region and observed delta direction.
        endpoint_mass = prob[:, 0] + prob[:, 255]
        for group, mask in [
            ("true_endpoint_classes", (true_cls == 0) | (true_cls == 255)),
            ("true_interior_classes", (true_cls != 0) & (true_cls != 255)),
            ("falling_delta", delta < 0), ("unchanged_delta", delta == 0),
            ("rising_delta", delta > 0), ("rapid_fall_lt_minus20", delta < -20),
            ("rapid_rise_gt20", delta > 20),
        ]:
            if mask.any():
                probability_rows.append({
                    "run": label, "conditioning_group": group, "n": int(mask.sum()),
                    "mean_endpoint_probability_mass_pct": float(100 * np.mean(endpoint_mass[mask])),
                    "mean_max_class_probability": float(np.mean(prob[mask].max(axis=1))),
                    "mean_entropy_nats": float(np.mean(-np.sum(prob[mask] * np.log(np.maximum(prob[mask], 1e-12)), axis=1))),
                })

    pd.DataFrame(temporal_rows).to_csv(out / "temporal_error_strata.csv", index=False)
    pd.DataFrame(probability_rows).to_csv(out / "probability_mass_by_true_dynamics.csv", index=False)

    # Hour-of-day pattern can reveal circadian/time-of-day concentration; timestamps are UTC
    # as parsed above, so this is explicitly UTC and not local patient time.
    hours = base_df["target_timestamp_parsed"].dt.hour.to_numpy()
    hour_rows = []
    for label, (df, _) in loaded.items():
        for decoder, col in DECODERS.items():
            forecast = df[col].to_numpy(float)
            for hour in range(24):
                mask = hours == hour
                if mask.any():
                    hour_rows.append({
                        "run": label, "decoder": decoder, "hour_utc": hour,
                        **_metrics(target[mask], forecast[mask]),
                    })
    pd.DataFrame(hour_rows).to_csv(out / "error_by_hour_utc.csv", index=False)

    # Residual autocorrelation is descriptive, not an inferential test.
    ac_rows = []
    for label, (df, _) in loaded.items():
        for decoder, col in DECODERS.items():
            residual = df[col].to_numpy(float) - target
            for lag in [1, 2, 3, 6, 12, 24]:
                if len(residual) > lag and np.std(residual[:-lag]) > 0 and np.std(residual[lag:]) > 0:
                    corr = np.corrcoef(residual[:-lag], residual[lag:])[0, 1]
                else:
                    corr = np.nan
                ac_rows.append({"run": label, "decoder": decoder,
                                "lag_samples": lag, "lag_minutes": lag * 5,
                                "residual_autocorrelation": float(corr)})
    pd.DataFrame(ac_rows).to_csv(out / "residual_autocorrelation.csv", index=False)

    (out / "README.txt").write_text(
        "GluNet saved-run failure-mechanism audit\\n"
        f"Patient {args.patient}; horizon {args.horizon} minutes; {len(target)} matched test windows.\\n"
        "Outputs: true_quantized_class_occupancy.csv, continuous_delta_to_class_mapping.csv, "
        "quantization_summary.csv, temporal_error_strata.csv, probability_mass_by_true_dynamics.csv, "
        "error_by_hour_utc.csv, residual_autocorrelation.csv.\\n"
        "Rapid changes are defined descriptively as absolute 30-minute delta >20 mg/dL. "
        "Hour-of-day is UTC because patient timezone metadata is not available in these artifacts. "
        "Residual autocorrelation is descriptive. No test-set quantities are used to fit model parameters.\\n"
    )
    print(f"Saved quantization and temporal failure audit: {out}")
    print("\\nQuantization summary")
    print(pd.read_csv(out / "quantization_summary.csv").to_string(index=False))
    print("\\nTemporal error strata")
    temporal = pd.read_csv(out / "temporal_error_strata.csv")
    print(temporal[temporal["group"].isin(["rapid_fall", "rapid_rise", "falling", "rising"])].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))
    print("\\nProbability mass by observed dynamics")
    print(pd.read_csv(out / "probability_mass_by_true_dynamics.csv").to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
