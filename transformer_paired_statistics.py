#!/usr/bin/env python3
"""
Paired patient-level statistical comparison for Phase-2 Ohio 2018 forecasting.

Compares Transformer against GRU, LSTM and TCN at each horizon using the
same patient-level test results.

Primary analysis:
  - Wilcoxon signed-rank test on patient-level MAE
  - Wilcoxon signed-rank test on patient-level RMSE
  - paired mean difference: baseline - Transformer
  - paired median difference
  - rank-biserial correlation
  - Holm multiple-comparison correction

The script handles both result schemas used in this project:

Baseline:
  test_mae_mgdl
  test_rmse_mgdl

Transformer:
  mae
  rmse

The Transformer result file does not need a model column because it is
already Transformer-specific.
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
    """Normalize column names without changing their meaning."""
    mapping = {}
    for c in df.columns:
        normalized = re.sub(
            r"[^a-z0-9]+", "_", str(c).strip().lower()
        ).strip("_")
        mapping[c] = normalized

    return df.rename(columns=mapping)


def find_col(df, candidates):
    """Return the first matching column from a list of aliases."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def load_results(path, transformer_file=False):
    """
    Load a result CSV and convert it to a common schema:

        patient
        horizon_min
        model
        mae
        rmse

    If transformer_file=True, the model is automatically assigned
    as 'Transformer' when the CSV has no model column.
    """
    path = Path(path)
    df = pd.read_csv(path)
    df = normalize_columns(df)

    aliases = {
        "patient": [
            "patient",
            "patient_id",
            "id",
        ],
        "horizon": [
            "horizon_min",
            "horizon",
            "horizon_minutes",
        ],
        "model": [
            "model",
            "architecture",
            "model_name",
        ],
        "mae": [
            "mae",
            "mean_absolute_error",
            "test_mae_mgdl",
            "test_mae",
        ],
        "rmse": [
            "rmse",
            "root_mean_squared_error",
            "test_rmse_mgdl",
            "test_rmse",
        ],
    }

    cols = {
        key: find_col(df, candidates)
        for key, candidates in aliases.items()
    }

    # Transformer result CSV is already Transformer-specific and
    # therefore does not contain a model column.
    if (
        cols["model"] is None
        and "mae" in df.columns
        and "rmse" in df.columns
    ):
        df["model"] = "Transformer"
        cols["model"] = "model"

    # Explicit transformer mode also assigns Transformer if necessary.
    if transformer_file and cols["model"] is None:
        df["model"] = "Transformer"
        cols["model"] = "model"

    missing = [key for key, value in cols.items() if value is None]

    if missing:
        raise ValueError(
            f"{path} is missing required columns {missing}.\n"
            f"Available columns: {list(df.columns)}"
        )

    out = df[
        [
            cols["patient"],
            cols["horizon"],
            cols["model"],
            cols["mae"],
            cols["rmse"],
        ]
    ].copy()

    out.columns = [
        "patient",
        "horizon_min",
        "model",
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

    return out


def rank_biserial_from_differences(differences):
    """
    Rank-biserial correlation for paired observations.

    differences = baseline error - Transformer error

    Therefore:
      positive effect -> Transformer has lower error
      negative effect -> Transformer has higher error
    """
    d = np.asarray(differences, dtype=float)

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

    positive_rank_sum = ranks[d > 0].sum()
    negative_rank_sum = ranks[d < 0].sum()

    denominator = (
        positive_rank_sum
        + negative_rank_sum
    )

    if denominator == 0:
        return np.nan

    return float(
        (positive_rank_sum - negative_rank_sum)
        / denominator
    )


def safe_wilcoxon(baseline, transformer):
    """Run paired two-sided Wilcoxon safely."""
    a = np.asarray(baseline, dtype=float)
    b = np.asarray(transformer, dtype=float)

    valid = (
        np.isfinite(a)
        & np.isfinite(b)
    )

    a = a[valid]
    b = b[valid]

    if len(a) < 2:
        return np.nan, np.nan, len(a)

    differences = a - b

    if np.allclose(differences, 0):
        return 0.0, 1.0, len(a)

    try:
        statistic, p_value = wilcoxon(
            a,
            b,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )

        return (
            float(statistic),
            float(p_value),
            len(a),
        )

    except Exception as exc:
        print(
            f"Warning: Wilcoxon failed: {exc}"
        )

        return (
            np.nan,
            np.nan,
            len(a),
        )


def compare_pair(
    combined,
    baseline_model,
    metric,
    horizons,
):
    """Compare one baseline model against Transformer."""
    rows = []

    for horizon in horizons:

        baseline = combined[
            (
                combined["model"]
                .str.lower()
                == baseline_model.lower()
            )
            & (
                combined["horizon_min"]
                == horizon
            )
        ][
            [
                "patient",
                metric,
            ]
        ].rename(
            columns={
                metric: "baseline"
            }
        )

        transformer = combined[
            (
                combined["model"]
                .str.lower()
                == "transformer"
            )
            & (
                combined["horizon_min"]
                == horizon
            )
        ][
            [
                "patient",
                metric,
            ]
        ].rename(
            columns={
                metric: "transformer"
            }
        )

        paired = baseline.merge(
            transformer,
            on="patient",
            how="inner",
        )

        paired = paired.sort_values(
            "patient"
        )

        if paired.empty:
            rows.append(
                {
                    "baseline": baseline_model,
                    "metric": metric.upper(),
                    "horizon_min": horizon,
                    "n_patients": 0,
                    "baseline_mean": np.nan,
                    "transformer_mean": np.nan,
                    "mean_difference_baseline_minus_transformer": np.nan,
                    "median_difference": np.nan,
                    "wilcoxon_stat": np.nan,
                    "p_value": np.nan,
                    "rank_biserial": np.nan,
                    "direction": "insufficient paired data",
                    "patients": "",
                }
            )

            continue

        baseline_values = (
            paired["baseline"]
            .to_numpy(dtype=float)
        )

        transformer_values = (
            paired["transformer"]
            .to_numpy(dtype=float)
        )

        differences = (
            baseline_values
            - transformer_values
        )

        statistic, p_value, n = (
            safe_wilcoxon(
                baseline_values,
                transformer_values,
            )
        )

        effect_size = (
            rank_biserial_from_differences(
                differences
            )
        )

        mean_difference = np.mean(
            differences
        )

        median_difference = np.median(
            differences
        )

        if mean_difference > 0:
            direction = (
                "Transformer lower error"
            )
        elif mean_difference < 0:
            direction = (
                "Transformer higher error"
            )
        else:
            direction = (
                "equal mean error"
            )

        rows.append(
            {
                "baseline": baseline_model,
                "metric": metric.upper(),
                "horizon_min": horizon,
                "n_patients": n,
                "baseline_mean": np.mean(
                    baseline_values
                ),
                "transformer_mean": np.mean(
                    transformer_values
                ),
                "mean_difference_baseline_minus_transformer": mean_difference,
                "median_difference": median_difference,
                "wilcoxon_stat": statistic,
                "p_value": p_value,
                "rank_biserial": effect_size,
                "direction": direction,
                "patients": ",".join(
                    paired["patient"]
                    .astype(str)
                    .tolist()
                ),
            }
        )

    return rows


def apply_holm_correction(results):
    """Apply Holm correction across all valid hypothesis tests."""
    results = results.copy()

    valid = results["p_value"].notna()

    results["p_holm"] = np.nan
    results["significant_holm_0_05"] = False

    if not valid.any():
        return results

    p_values = (
        results.loc[
            valid,
            "p_value",
        ]
        .to_numpy(dtype=float)
    )

    if multipletests is not None:

        rejected, corrected, _, _ = (
            multipletests(
                p_values,
                alpha=0.05,
                method="holm",
            )
        )

        results.loc[
            valid,
            "p_holm",
        ] = corrected

        results.loc[
            valid,
            "significant_holm_0_05",
        ] = rejected

    else:
        # Manual Holm correction fallback.
        order = np.argsort(p_values)

        adjusted = np.empty_like(
            p_values
        )

        number_tests = len(
            p_values
        )

        running_max = 0.0

        for rank, index in enumerate(order):

            adjusted_value = min(
                1.0,
                (
                    number_tests
                    - rank
                )
                * p_values[index],
            )

            running_max = max(
                running_max,
                adjusted_value,
            )

            adjusted[index] = (
                running_max
            )

        results.loc[
            valid,
            "p_holm",
        ] = adjusted

        results.loc[
            valid,
            "significant_holm_0_05",
        ] = (
            adjusted < 0.05
        )

    return results


def find_transformer_file(
    root,
):
    """Find the Transformer result CSV."""
    transformer_dir = (
        root
        / "output"
        / "phase2_transformer_mgdl"
    )

    if not transformer_dir.exists():
        raise FileNotFoundError(
            f"Transformer directory not found:\n"
            f"{transformer_dir}"
        )

    csv_files = list(
        transformer_dir.glob("*.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            "No CSV files found in:\n"
            f"{transformer_dir}"
        )

    # Prefer transformer_results.csv.
    preferred = (
        transformer_dir
        / "transformer_results.csv"
    )

    if preferred.exists():
        return preferred

    # Otherwise prefer a filename containing "result".
    result_files = [
        path
        for path in csv_files
        if "result" in path.name.lower()
    ]

    if result_files:
        return result_files[0]

    return csv_files[0]


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Patient-level paired statistical "
            "comparison of Transformer against "
            "GRU/LSTM/TCN."
        )
    )

    parser.add_argument(
        "--repo-root",
        default=".",
        help=(
            "CGM Forecasting Research repository root."
        ),
    )

    parser.add_argument(
        "--baseline-file",
        default=None,
        help=(
            "Optional explicit baseline result CSV."
        ),
    )

    parser.add_argument(
        "--transformer-file",
        default=None,
        help=(
            "Optional explicit Transformer result CSV."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "output/phase2_statistics"
        ),
        help=(
            "Directory for statistical outputs."
        ),
    )

    args = parser.parse_args()

    root = (
        Path(args.repo_root)
        .expanduser()
        .resolve()
    )

    output_dir = (
        root
        / args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Locate baseline results.
    # ---------------------------------------------------------

    if args.baseline_file:

        baseline_path = (
            Path(args.baseline_file)
            .expanduser()
            .resolve()
        )

    else:

        baseline_path = (
            root
            / "output"
            / "phase2_baselines"
            / "ohio2018_all_patients_baseline_results.csv"
        )

    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Baseline result file not found:\n"
            f"{baseline_path}"
        )

    # ---------------------------------------------------------
    # Locate Transformer results.
    # ---------------------------------------------------------

    if args.transformer_file:

        transformer_path = (
            Path(args.transformer_file)
            .expanduser()
            .resolve()
        )

    else:

        transformer_path = (
            find_transformer_file(root)
        )

    if not transformer_path.exists():
        raise FileNotFoundError(
            f"Transformer result file not found:\n"
            f"{transformer_path}"
        )

    # ---------------------------------------------------------
    # Load.
    # ---------------------------------------------------------

    baseline = load_results(
        baseline_path,
        transformer_file=False,
    )

    transformer = load_results(
        transformer_path,
        transformer_file=True,
    )

    # Ensure Transformer label.
    transformer["model"] = (
        "Transformer"
    )

    baseline["model"] = (
        baseline["model"]
        .astype(str)
        .str.strip()
    )

    transformer["model"] = (
        transformer["model"]
        .astype(str)
        .str.strip()
    )

    # ---------------------------------------------------------
    # Combine.
    # ---------------------------------------------------------

    wanted_models = {
        "gru",
        "lstm",
        "tcn",
        "transformer",
    }

    combined = pd.concat(
        [
            baseline,
            transformer,
        ],
        ignore_index=True,
    )

    combined = combined[
        combined["model"]
        .str.lower()
        .isin(wanted_models)
    ].copy()

    # Remove duplicate patient/model/horizon rows.
    combined = combined.drop_duplicates(
        subset=[
            "patient",
            "horizon_min",
            "model",
        ],
        keep="last",
    )

    patients = sorted(
        combined["patient"]
        .astype(str)
        .unique()
        .tolist()
    )

    horizons = sorted(
        combined["horizon_min"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    print()
    print(
        "=================================================="
    )
    print(
        "Patient-level paired statistical comparison"
    )
    print(
        "=================================================="
    )

    print(
        f"Baseline file:    {baseline_path}"
    )

    print(
        f"Transformer file: {transformer_path}"
    )

    print(
        f"Patients: {patients}"
    )

    print(
        f"Horizons: {horizons}"
    )

    print()

    # ---------------------------------------------------------
    # Statistical comparisons.
    # ---------------------------------------------------------

    baseline_models = [
        "GRU",
        "LSTM",
        "TCN",
    ]

    rows = []

    for model in baseline_models:

        for metric in [
            "mae",
            "rmse",
        ]:

            rows.extend(
                compare_pair(
                    combined=combined,
                    baseline_model=model,
                    metric=metric,
                    horizons=horizons,
                )
            )

    results = pd.DataFrame(
        rows
    )

    results = apply_holm_correction(
        results
    )

    results = results.sort_values(
        [
            "metric",
            "horizon_min",
            "baseline",
        ]
    ).reset_index(
        drop=True
    )

    # ---------------------------------------------------------
    # Save complete results.
    # ---------------------------------------------------------

    complete_path = (
        output_dir
        / "transformer_paired_statistics.csv"
    )

    results.to_csv(
        complete_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Save compact results.
    # ---------------------------------------------------------

    compact_columns = [
        "baseline",
        "metric",
        "horizon_min",
        "n_patients",
        "baseline_mean",
        "transformer_mean",
        "mean_difference_baseline_minus_transformer",
        "median_difference",
        "wilcoxon_stat",
        "p_value",
        "p_holm",
        "rank_biserial",
        "significant_holm_0_05",
        "direction",
    ]

    compact = results[
        compact_columns
    ].copy()

    numeric_columns = [
        "baseline_mean",
        "transformer_mean",
        "mean_difference_baseline_minus_transformer",
        "median_difference",
        "wilcoxon_stat",
        "p_value",
        "p_holm",
        "rank_biserial",
    ]

    for column in numeric_columns:
        compact[column] = compact[
            column
        ].round(6)

    compact_path = (
        output_dir
        / "transformer_paired_statistics_compact.csv"
    )

    compact.to_csv(
        compact_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Print useful result table.
    # ---------------------------------------------------------

    print(
        "Interpretation of difference:"
    )

    print(
        "  baseline - Transformer > 0 "
        "means Transformer has lower error."
    )

    print()

    print(
        "Results:"
    )

    print(
        compact.to_string(
            index=False
        )
    )

    print()

    print(
        "=================================================="
    )

    print(
        "Saved:"
    )

    print(
        f"  {complete_path}"
    )

    print(
        f"  {compact_path}"
    )

    print(
        "=================================================="
    )


if __name__ == "__main__":
    main()
