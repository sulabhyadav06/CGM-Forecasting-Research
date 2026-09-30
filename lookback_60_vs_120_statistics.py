#!/usr/bin/env python3
"""
Paired 60-vs-120 minute Transformer lookback comparison.

This script reads the per-patient Transformer result CSVs produced by:

    output/phase2_transformer_lookback_study/lookback_60min/
    output/phase2_transformer_lookback_study/lookback_120min/

It performs paired patient-level Wilcoxon signed-rank tests for:
    - MAE
    - RMSE

It also reports:
    - per-patient errors
    - mean / SD for each lookback
    - mean and median paired improvement
    - rank-biserial effect size
    - optional Holm correction for the two primary tests

Positive improvement means:
    60-minute error - 120-minute error > 0
i.e. the 120-minute lookback has lower error.

Run from the repository root:

    python lookback_60_vs_120_statistics.py
"""

from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

try:
    from statsmodels.stats.multitest import multipletests
except ImportError:
    multipletests = None


def normalize_columns(df):
    mapping = {}
    for col in df.columns:
        mapping[col] = re.sub(
            r"[^a-z0-9]+",
            "_",
            str(col).strip().lower(),
        ).strip("_")
    return df.rename(columns=mapping)


def find_result_file(directory):
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found:\n{directory}"
        )

    csvs = list(directory.glob("*.csv"))

    if not csvs:
        raise FileNotFoundError(
            f"No CSV files found in:\n{directory}"
        )

    preferred_names = [
        "transformer_results.csv",
        "results.csv",
    ]

    for name in preferred_names:
        path = directory / name
        if path.exists():
            return path

    result_files = [
        p for p in csvs
        if "result" in p.name.lower()
    ]

    if result_files:
        return result_files[0]

    return csvs[0]


def load_results(path, lookback):
    df = pd.read_csv(path)
    df = normalize_columns(df)

    patient_col = next(
        (
            c for c in
            ["patient", "patient_id", "id"]
            if c in df.columns
        ),
        None,
    )

    horizon_col = next(
        (
            c for c in
            ["horizon_min", "horizon", "horizon_minutes"]
            if c in df.columns
        ),
        None,
    )

    mae_col = next(
        (
            c for c in
            [
                "mae",
                "test_mae_mgdl",
                "test_mae",
                "mean_absolute_error",
            ]
            if c in df.columns
        ),
        None,
    )

    rmse_col = next(
        (
            c for c in
            [
                "rmse",
                "test_rmse_mgdl",
                "test_rmse",
                "root_mean_squared_error",
            ]
            if c in df.columns
        ),
        None,
    )

    missing = [
        name
        for name, col in [
            ("patient", patient_col),
            ("horizon", horizon_col),
            ("mae", mae_col),
            ("rmse", rmse_col),
        ]
        if col is None
    ]

    if missing:
        raise ValueError(
            f"{path} is missing required fields: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    out = df[
        [
            patient_col,
            horizon_col,
            mae_col,
            rmse_col,
        ]
    ].copy()

    out.columns = [
        "patient",
        "horizon_min",
        "mae",
        "rmse",
    ]

    out["patient"] = out["patient"].astype(str)
    out["horizon_min"] = pd.to_numeric(
        out["horizon_min"],
        errors="coerce",
    )
    out["mae"] = pd.to_numeric(
        out["mae"],
        errors="coerce",
    )
    out["rmse"] = pd.to_numeric(
        out["rmse"],
        errors="coerce",
    )

    out = out.dropna(
        subset=[
            "patient",
            "horizon_min",
            "mae",
            "rmse",
        ]
    )

    # The requested study is the 30-minute horizon.
    out = out[
        out["horizon_min"] == 30
    ].copy()

    if out.empty:
        raise ValueError(
            f"No 30-minute horizon results found in {path}."
        )

    # Avoid duplicated patient rows.
    out = out.drop_duplicates(
        subset=["patient"],
        keep="last",
    )

    out["lookback_min"] = lookback

    return out


def rank_biserial(differences):
    """
    Rank-biserial correlation for paired differences.

    differences = error_60 - error_120

    Positive values mean the 120-minute lookback has lower error.
    """
    d = np.asarray(
        differences,
        dtype=float,
    )

    d = d[
        np.isfinite(d)
        & (d != 0)
    ]

    if len(d) == 0:
        return np.nan

    ranks = (
        pd.Series(np.abs(d))
        .rank(method="average")
        .to_numpy()
    )

    positive = ranks[d > 0].sum()
    negative = ranks[d < 0].sum()

    denominator = positive + negative

    if denominator == 0:
        return np.nan

    return float(
        (positive - negative)
        / denominator
    )


def paired_test(df, metric):
    paired = df[
        [
            "patient",
            f"{metric}_60",
            f"{metric}_120",
        ]
    ].dropna()

    a = paired[
        f"{metric}_60"
    ].to_numpy(dtype=float)

    b = paired[
        f"{metric}_120"
    ].to_numpy(dtype=float)

    differences = a - b

    if len(a) < 2:
        statistic = np.nan
        p_value = np.nan

    elif np.allclose(
        differences,
        0,
    ):
        statistic = 0.0
        p_value = 1.0

    else:
        statistic, p_value = wilcoxon(
            a,
            b,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )

        statistic = float(statistic)
        p_value = float(p_value)

    return {
        "metric": metric.upper(),
        "n_patients": len(a),
        "lookback_60_mean": np.mean(a),
        "lookback_60_sd": (
            np.std(a, ddof=1)
            if len(a) > 1
            else np.nan
        ),
        "lookback_120_mean": np.mean(b),
        "lookback_120_sd": (
            np.std(b, ddof=1)
            if len(b) > 1
            else np.nan
        ),
        "mean_improvement_60_minus_120": np.mean(
            differences
        ),
        "median_improvement_60_minus_120": np.median(
            differences
        ),
        "wilcoxon_stat": statistic,
        "p_value": p_value,
        "rank_biserial": rank_biserial(
            differences
        ),
        "n_120_better": int(
            np.sum(differences > 0)
        ),
        "n_60_better": int(
            np.sum(differences < 0)
        ),
        "n_equal": int(
            np.sum(differences == 0)
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-root",
        default=".",
    )

    parser.add_argument(
        "--lookback-60-dir",
        default=(
            "output/"
            "phase2_transformer_lookback_study/"
            "lookback_60min"
        ),
    )

    parser.add_argument(
        "--lookback-120-dir",
        default=(
            "output/"
            "phase2_transformer_lookback_study/"
            "lookback_120min"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "output/"
            "phase2_transformer_lookback_statistics"
        ),
    )

    args = parser.parse_args()

    root = (
        Path(args.repo_root)
        .expanduser()
        .resolve()
    )

    dir60 = (
        root / args.lookback_60_dir
    )

    dir120 = (
        root / args.lookback_120_dir
    )

    output_dir = (
        root / args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    file60 = find_result_file(dir60)
    file120 = find_result_file(dir120)

    result60 = load_results(
        file60,
        lookback=60,
    )

    result120 = load_results(
        file120,
        lookback=120,
    )

    # Merge only patients present in both runs.
    paired = result60.merge(
        result120,
        on="patient",
        how="inner",
        suffixes=("_60", "_120"),
    )

    paired = paired.sort_values(
        "patient"
    )

    if len(paired) < 2:
        raise ValueError(
            "Fewer than two paired patients were found."
        )

    # Keep the core paired patient table.
    patient_table = paired[
        [
            "patient",
            "mae_60",
            "mae_120",
            "rmse_60",
            "rmse_120",
        ]
    ].copy()

    patient_table[
        "mae_improvement_60_minus_120"
    ] = (
        patient_table["mae_60"]
        - patient_table["mae_120"]
    )

    patient_table[
        "rmse_improvement_60_minus_120"
    ] = (
        patient_table["rmse_60"]
        - patient_table["rmse_120"]
    )

    patient_path = (
        output_dir
        / "lookback_60_vs_120_patient_results.csv"
    )

    patient_table.to_csv(
        patient_path,
        index=False,
    )

    # Statistical tests.
    summary_rows = [
        paired_test(
            paired,
            "mae",
        ),
        paired_test(
            paired,
            "rmse",
        ),
    ]

    summary = pd.DataFrame(
        summary_rows
    )

    # Holm correction across the two primary tests.
    valid = summary["p_value"].notna()

    summary["p_holm"] = np.nan
    summary["significant_holm_0_05"] = False

    if valid.any():

        p_values = (
            summary.loc[
                valid,
                "p_value",
            ].to_numpy()
        )

        if multipletests is not None:

            rejected, corrected, _, _ = (
                multipletests(
                    p_values,
                    alpha=0.05,
                    method="holm",
                )
            )

            summary.loc[
                valid,
                "p_holm",
            ] = corrected

            summary.loc[
                valid,
                "significant_holm_0_05",
            ] = rejected

        else:
            # Manual Holm correction.
            order = np.argsort(
                p_values
            )

            corrected = np.empty_like(
                p_values
            )

            m = len(p_values)
            running = 0.0

            for rank, index in enumerate(order):
                value = min(
                    1.0,
                    (m - rank)
                    * p_values[index],
                )

                running = max(
                    running,
                    value,
                )

                corrected[index] = running

            summary.loc[
                valid,
                "p_holm",
            ] = corrected

            summary.loc[
                valid,
                "significant_holm_0_05",
            ] = (
                corrected < 0.05
            )

    summary_path = (
        output_dir
        / "lookback_60_vs_120_statistics.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # Print.
    print("\n" + "=" * 78)
    print(
        "60-MIN vs 120-MIN TRANSFORMER LOOKBACK "
        "STATISTICAL COMPARISON"
    )
    print("=" * 78)

    print(f"\n60-min results:\n{file60}")
    print(f"\n120-min results:\n{file120}")

    print(
        f"\nPaired patients "
        f"({len(paired)}): "
        f"{paired['patient'].tolist()}"
    )

    print(
        "\nPositive improvement = "
        "60-min error - 120-min error > 0."
    )

    print(
        "Therefore, positive values indicate "
        "lower error with the 120-min lookback."
    )

    print("\nPer-patient results:")
    print(
        patient_table.round(6).to_string(
            index=False
        )
    )

    print("\nStatistical summary:")
    print(
        summary.round(6).to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(patient_path)
    print(summary_path)


if __name__ == "__main__":
    main()
