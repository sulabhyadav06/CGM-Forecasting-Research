"""
significance_test.py -- UPDATED

Reproduces every statistical significance claim in the combined findings
report (CGM_Forecasting_Combined_Report.md), from the actual output CSVs,
rather than being computed ad hoc. Run this after both comparison scripts
have finished:
    python multimodel_compare_all.py
    python multimodel_architecture_compare.py

Covers two separate analyses:
  1. FEATURE-SET comparisons (output/full_results_by_patient.csv)
     - A vs C vs E across all 12 patients (cohort-agnostic feature sets)
     - A vs B vs D vs F within the 6 2018 patients only (wearable-dependent
       feature sets -- 2020 patients have no HR/Steps data)
  2. ARCHITECTURE comparisons (output/architecture_all12_results.csv)
     - GRU (best performer) vs LSTM, BiLSTM, TCN, Transformer,
       across all 12 patients, at each horizon

All tests are Wilcoxon signed-rank (paired by patient), matching the report.
"""

import os
import pandas as pd
from scipy.stats import wilcoxon


def paired_wilcoxon(df, group_col, group_a, group_b, patient_col, value_col,
                     filter_col=None, filter_val=None):
    """
    Paired Wilcoxon signed-rank test between two groups (e.g. two feature
    sets, or two architectures), matched by patient. Returns a dict with
    n, means, p-value, and how many patients favored group_b.
    Returns None if there are fewer than 2 paired observations (Wilcoxon
    needs at least 2 non-zero differences to run).
    """
    sub = df if filter_col is None else df[df[filter_col] == filter_val]
    a = sub[sub[group_col] == group_a].set_index(patient_col)[value_col]
    b = sub[sub[group_col] == group_b].set_index(patient_col)[value_col]
    common = a.index.intersection(b.index)
    if len(common) < 2:
        return None
    a, b = a.loc[common], b.loc[common]
    diffs = a - b
    if (diffs == 0).all():
        return {"n": len(common), "a_mean": a.mean(), "b_mean": b.mean(),
                "p": 1.0, "b_better": int((b < a).sum())}
    stat, p = wilcoxon(a, b)
    return {"n": len(common), "a_mean": a.mean(), "b_mean": b.mean(),
            "p": p, "b_better": int((b < a).sum())}


def sig_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def print_result(label, result):
    if result is None:
        print(f"  {label}: not enough paired patients to test")
        return
    r = result
    print(f"  {label}: A={r['a_mean']:.2f}  B={r['b_mean']:.2f}  "
          f"p={r['p']:.4f}{sig_stars(r['p'])}  (B better in {r['b_better']}/{r['n']})")


# --------------------------------------------------------------------------- #
# 1. Feature-set comparisons
# --------------------------------------------------------------------------- #

FEATURESET_PATH = "output/full_results_by_patient.csv"
HORIZONS = ["15min", "30min", "60min"]

MODEL_A = "A: Glucose only"
MODEL_B = "B: Glucose + HeartRate"
MODEL_C = "C: Glucose + Carbs(meals)"
MODEL_D = "D: Glucose + HR + Steps + Carbs + Sleep"
MODEL_E = "E: Glucose + Insulin (bolus+basal)"
MODEL_F = "F: Full multimodal (HR+Steps+Carbs+Sleep+Insulin)"

if os.path.exists(FEATURESET_PATH):
    feat = pd.read_csv(FEATURESET_PATH)

    print("=" * 78)
    print("FEATURE-SET COMPARISONS -- all 12 patients (A vs C vs E)")
    print("=" * 78)
    for horizon in HORIZONS:
        print(f"\n--- {horizon} ---")
        print_result("A vs C (carbs)", paired_wilcoxon(
            feat, "Model", MODEL_A, MODEL_C, "Patient", "RMSE", "Horizon", horizon))
        print_result("A vs E (insulin)", paired_wilcoxon(
            feat, "Model", MODEL_A, MODEL_E, "Patient", "RMSE", "Horizon", horizon))

    print("\n" + "=" * 78)
    print("FEATURE-SET COMPARISONS -- 2018 cohort only, n=6 (A vs B vs D vs F)")
    print("(B/D/F require HeartRate/Steps, unavailable for 2020 patients)")
    print("=" * 78)
    for horizon in HORIZONS:
        print(f"\n--- {horizon} ---")
        print_result("A vs B (heart rate)", paired_wilcoxon(
            feat, "Model", MODEL_A, MODEL_B, "Patient", "RMSE", "Horizon", horizon))
        print_result("A vs D (full wearable)", paired_wilcoxon(
            feat, "Model", MODEL_A, MODEL_D, "Patient", "RMSE", "Horizon", horizon))
        print_result("A vs F (+ insulin)", paired_wilcoxon(
            feat, "Model", MODEL_A, MODEL_F, "Patient", "RMSE", "Horizon", horizon))
else:
    print(f"[skip] {FEATURESET_PATH} not found -- run multimodel_compare_all.py first.")


# --------------------------------------------------------------------------- #
# 2. Architecture comparisons
# --------------------------------------------------------------------------- #

ARCHITECTURE_PATH = "output/architecture_all12_results.csv"

if os.path.exists(ARCHITECTURE_PATH):
    arch = pd.read_csv(ARCHITECTURE_PATH)

    print("\n\n" + "=" * 78)
    print("ARCHITECTURE COMPARISONS -- all 12 patients, GRU vs each other architecture")
    print("=" * 78)
    for horizon in HORIZONS:
        print(f"\n--- {horizon} ---")
        for other in ["LSTM", "BiLSTM", "TCN", "Transformer"]:
            print_result(f"GRU vs {other}", paired_wilcoxon(
                arch, "Architecture", "GRU", other, "Patient", "RMSE", "Horizon", horizon))
else:
    print(f"\n[skip] {ARCHITECTURE_PATH} not found -- run multimodel_architecture_compare.py first.")


print("\n\nNote: '*' p<0.05, '**' p<0.01, '***' p<0.001. All tests are Wilcoxon")
print("signed-rank, paired by patient. A significant result with a small n")
print("(e.g. the 6-patient 2018-only comparisons) should still be treated as")
print("preliminary -- see the report's Limitations section.")
