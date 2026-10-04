from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PATIENTS = {
    "ohio2018": [559, 563, 570, 575, 588, 591],
    "ohio2020": [540, 544, 552, 567, 584, 596],
}

COMMON = [
    "timestamp",
    "glucose",
    "gap_from_previous_min",
    "valid_5min_interval",
    "carbs_last_60min",
    "bolus_last_60min",
    "basal_rate",
]

FEATURES_2018 = [
    "heart_rate_mean_5m",
    "heart_rate_std_5m",
    "heart_rate_min_5m",
    "heart_rate_max_5m",
    "heart_rate_count_5m",
    "heart_rate_coverage_5m",
    "heart_rate_missing_5m",
    "steps_mean_5m",
    "steps_std_5m",
    "steps_min_5m",
    "steps_max_5m",
    "steps_count_5m",
    "steps_coverage_5m",
    "steps_missing_5m",
]

FEATURES_2020 = [
    "accel_mean_5m",
    "accel_std_5m",
    "accel_min_5m",
    "accel_max_5m",
    "accel_count_5m",
    "accel_coverage_5m",
    "accel_missing_5m",
]


def parse_args():
    p = argparse.ArgumentParser(description="Audit Phase-2 OhioT1DM CSVs.")
    p.add_argument("--data-root", default="data/phase2")
    p.add_argument("--output-root", default="output/phase2_data_audit")
    p.add_argument("--max-gap-min", type=float, default=30.0)
    p.add_argument("--expected-step-min", type=float, default=5.0)
    return p.parse_args()


def audit_file(path, cohort, split, patient, max_gap, expected_step):
    df = pd.read_csv(path)
    rows = len(df)
    issues = []

    expected = COMMON + (
        FEATURES_2018 if cohort == "ohio2018" else FEATURES_2020
    )

    missing = [c for c in expected if c not in df.columns]
    extra = [c for c in df.columns if c not in expected]

    if missing:
        issues.append("missing_columns:" + "|".join(missing))
    if extra:
        issues.append("extra_columns:" + "|".join(extra))

    if "timestamp" not in df.columns:
        return {
            "cohort": cohort,
            "patient": patient,
            "split": split,
            "rows": rows,
            "issues": "missing_timestamp",
        }

    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    bad_ts = int(ts.isna().sum())
    dup = int(ts.duplicated().sum())

    dmins = ts.diff().dt.total_seconds().div(60)
    nonpositive = int((dmins.dropna() <= 0).sum())
    gaps = dmins.dropna()

    large = int((gaps > max_gap).sum())
    non5 = int(((gaps - expected_step).abs() > 1e-6).sum())

    glucose_bad = (
        int(pd.to_numeric(df["glucose"], errors="coerce").isna().sum())
        if "glucose" in df.columns
        else rows
    )

    numeric = df.select_dtypes(include=[np.number])
    nan_pct = (numeric.isna().mean() * 100).to_dict()
    constant = [
        c for c in numeric.columns
        if numeric[c].nunique(dropna=True) <= 1
    ]

    valid_mismatch = -1
    if "valid_5min_interval" in df.columns:
        expected_valid = dmins.eq(expected_step) | dmins.isna()
        valid_mismatch = int(
            (expected_valid != df["valid_5min_interval"].fillna(False)).sum()
        )

    suspicious_constant_observed = []
    for c in ("carbs_observed_60min", "bolus_observed_60min"):
        if c in df.columns and df[c].nunique(dropna=True) == 1:
            suspicious_constant_observed.append(c)

    if bad_ts:
        issues.append(f"bad_timestamps:{bad_ts}")
    if dup:
        issues.append(f"duplicate_timestamps:{dup}")
    if nonpositive:
        issues.append(f"nonpositive_time_deltas:{nonpositive}")
    if large:
        issues.append(f"gaps_over_{max_gap:g}min:{large}")
    if glucose_bad:
        issues.append(f"missing_glucose:{glucose_bad}")
    if valid_mismatch > 0:
        issues.append(f"valid_interval_mismatch:{valid_mismatch}")
    if suspicious_constant_observed:
        issues.append(
            "constant_observed_flags:" + "|".join(suspicious_constant_observed)
        )

    return {
        "cohort": cohort,
        "patient": patient,
        "split": split,
        "file": str(path),
        "rows": rows,
        "start": ts.min(),
        "end": ts.max(),
        "median_step_min": float(gaps.median()) if len(gaps) else np.nan,
        "mean_step_min": float(gaps.mean()) if len(gaps) else np.nan,
        "min_step_min": float(gaps.min()) if len(gaps) else np.nan,
        "max_step_min": float(gaps.max()) if len(gaps) else np.nan,
        "non5_intervals": non5,
        "nonpositive_time_deltas": nonpositive,
        "gaps_over_max": large,
        "duplicate_timestamps": dup,
        "bad_timestamps": bad_ts,
        "missing_glucose": glucose_bad,
        "valid_5min_pct": (
            float(df["valid_5min_interval"].mean() * 100)
            if "valid_5min_interval" in df.columns
            else np.nan
        ),
        "valid_interval_mismatch": valid_mismatch,
        "constant_numeric_features": "|".join(constant),
        "issues": ";".join(issues) if issues else "",
        **{f"nan_pct__{k}": v for k, v in nan_pct.items()},
    }


def main():
    args = parse_args()
    root = Path(args.data_root)
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)

    summaries = []
    missing_files = []

    for cohort, patients in PATIENTS.items():
        for split in ("train", "test"):
            for patient in patients:
                path = root / cohort / split / f"{patient}.csv"

                if not path.exists():
                    missing_files.append(str(path))
                    continue

                summaries.append(
                    audit_file(
                        path,
                        cohort,
                        split,
                        patient,
                        args.max_gap_min,
                        args.expected_step_min,
                    )
                )

    summary = pd.DataFrame(summaries)
    summary.to_csv(
        out / "phase2_data_quality_summary.csv",
        index=False,
    )

    miss_cols = [
        c for c in summary.columns
        if c.startswith("nan_pct__")
    ]

    if miss_cols:
        miss = summary[
            ["cohort", "patient", "split"] + miss_cols
        ].melt(
            id_vars=["cohort", "patient", "split"],
            var_name="feature",
            value_name="missing_pct",
        )
        miss["feature"] = miss["feature"].str.replace(
            "nan_pct__",
            "",
            regex=False,
        )
        miss.to_csv(
            out / "phase2_missingness_summary.csv",
            index=False,
        )

    # Train/test timestamp overlap.
    overlap = []

    for cohort, patients in PATIENTS.items():
        for patient in patients:
            paths = {
                split: root / cohort / split / f"{patient}.csv"
                for split in ("train", "test")
            }

            if not all(p.exists() for p in paths.values()):
                continue

            info = {}

            for split, path in paths.items():
                ts = pd.to_datetime(
                    pd.read_csv(path, usecols=["timestamp"])["timestamp"],
                    utc=True,
                    errors="coerce",
                ).dropna()

                info[split] = (
                    ts.min(),
                    ts.max(),
                    set(ts.astype("int64")),
                )

            intersection = len(
                info["train"][2] & info["test"][2]
            )

            date_range_overlap = not (
                info["train"][1] < info["test"][0]
                or info["test"][1] < info["train"][0]
            )

            overlap.append(
                {
                    "cohort": cohort,
                    "patient": patient,
                    "train_start": info["train"][0],
                    "train_end": info["train"][1],
                    "test_start": info["test"][0],
                    "test_end": info["test"][1],
                    "exact_timestamp_overlap": intersection,
                    "date_range_overlap": date_range_overlap,
                }
            )

    pd.DataFrame(overlap).to_csv(
        out / "phase2_train_test_overlap.csv",
        index=False,
    )

    # Expected schema table.
    feature_rows = []

    for cohort, patients in PATIENTS.items():
        expected = COMMON + (
            FEATURES_2018 if cohort == "ohio2018" else FEATURES_2020
        )

        for feature in expected:
            feature_rows.append(
                {
                    "cohort": cohort,
                    "feature": feature,
                    "expected": True,
                }
            )

    pd.DataFrame(feature_rows).to_csv(
        out / "phase2_feature_schema.csv",
        index=False,
    )

    report = [
        "PHASE-2 DATA QUALITY AUDIT",
        "=" * 30,
        f"Files checked: {len(summary)}",
        f"Missing files: {len(missing_files)}",
    ]

    if missing_files:
        report.append("MISSING FILES:")
        report.extend("  " + x for x in missing_files)

    if len(summary):
        report.extend(
            [
                f"Files with issues: "
                f"{(summary['issues'].astype(str).str.len() > 0).sum()}",
                f"Files with >{args.max_gap_min:g} min gaps: "
                f"{(summary['gaps_over_max'] > 0).sum()}",
                f"Files with duplicate timestamps: "
                f"{(summary['duplicate_timestamps'] > 0).sum()}",
                f"Files with missing glucose: "
                f"{(summary['missing_glucose'] > 0).sum()}",
            ]
        )

    report.extend(
        [
            "",
            "IMPORTANT INTERPRETATION",
            "- This audit validates stored Phase-2 CSV integrity and schema.",
            "- It cannot prove train-only scaling or model-level target leakage from CSVs alone.",
            "- Sequence construction must separately enforce continuous windows and future-target exclusion.",
            "- Constant carbs_observed_60min/bolus_observed_60min flags are reported as suspicious because the current preprocessor writes them as 1 for every row.",
        ]
    )

    (out / "phase2_data_quality_report.txt").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print("\n".join(report))

    structural_failure = (
        missing_files
        or (
            len(summary)
            and (
                (summary["bad_timestamps"] > 0)
                | (summary["duplicate_timestamps"] > 0)
                | (summary["nonpositive_time_deltas"] > 0)
                | (summary["missing_glucose"] > 0)
            ).any()
        )
    )

    if structural_failure:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
