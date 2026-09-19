# Personalized Blood Glucose Forecasting

## Reference Paper

**Title:** Personalized Blood Glucose Forecasting From Limited CGM Data Using Incrementally Retrained LSTM (IEEE TBME, 2025). **PMC ID:** PMC11999170.

Note: the reference paper uses OpenAPS and Replace-BG datasets. This project reproduces the general approach (LSTM-based CGM forecasting, 15/30/60-min horizons) using the OhioT1DM dataset instead, since it was the accessible dataset with matching resolution (5-min CGM) and richer physiological signals (heart rate, steps, meals, sleep, insulin dosing) suited for the multimodal extension.

## Dataset

**OhioT1DM — all 12 patients**, obtained under a signed Data Use Agreement:
- **2018 release** (6 patients: 559, 563, 570, 575, 588, 591) — 5-minute CGM, plus Basis wearable signals (heart rate, steps, GSR, skin/air temperature), self-reported meal/sleep/exercise events, and insulin pump data (bolus, basal, temp basal).
- **2020 release** (6 patients: 540, 544, 552, 567, 584, 596) — 5-minute CGM, insulin pump data, and self-reported meal/sleep events. **No heart rate or step data** — this cohort used a different wristband (Empatica Embrace vs. the 2018 cohort's Basis Peak), which does not report those channels. Confirmed by direct inspection of the source XML (zero `basis_heart_rate`/`basis_steps` events across all six 2020 patients).

**Data is not included in this repository.** OhioT1DM is distributed under a Data Use Agreement that restricts redistribution; `data/` is git-ignored. To reproduce, request access via razvan.bunescu@charlotte.edu (subject: "OhioT1DM Request") and place the extracted XML files in `data/` before running `parse_xml.py`.

## Pipeline

1. **`parse_xml.py`** — parses each patient's XML into a per-patient multimodal CSV (glucose, heart rate, steps, carbs-last-60min, sleep flag, insulin bolus-last-60min, active basal rate). Handles both dataset releases' schema differences (e.g. 2020's `tbegin`/`tend` sleep attributes vs. 2018's `ts_begin`/`ts_end`).
2. **`multimodel_compare_all.py`** — trains a single LSTM per patient/horizon across 6 feature sets (A: glucose only, through F: full multimodal including insulin), for all 12 patients where the feature set's inputs are available.
3. **`multimodel_architecture_compare.py`** — fixes the feature set and instead compares 5 architectures (LSTM, GRU, BiLSTM, TCN, Transformer) across all 12 patients and all 3 horizons, with the full clinical metric suite (RMSE, MAE, MARD, time-lag, Clarke Error Grid zone distribution).
4. **`significance_test.py`** — Wilcoxon signed-rank significance testing (paired by patient) for both the feature-set and architecture comparisons, run directly from the output CSVs.
5. **`clinical_metrics.py`** — shared module: MARD, cross-correlation time-lag, and Clarke Error Grid zone classification (A–E).

All scripts train patients in parallel via `ProcessPoolExecutor` and use early stopping, since the full grid (12 patients × 3 horizons × up to 6 feature sets or 5 architectures) is a substantial training workload.

## Results: Baseline Reproduction (single patient, patient 570, 30-min horizon)

Original architectural sanity check before scaling to the full cohort.

| Model | RMSE (mg/dL) | MAE (mg/dL) |
|---|---|---|
| Linear Regression | 59.47 | 34.88 |
| Random Forest | 63.93 | 40.79 |
| Baseline LSTM (unscaled) | 68.26 | 45.51 |
| LSTM (scaled + engineered features) | 51.62 | 33.91 |

Note: our RMSE is higher than the paper's IS-LSTM benchmark (10.23–13.41 mg/dL on OpenAPS), which uses a more sophisticated incrementally retrained architecture and a different dataset. Our baseline LSTM serves as a fair architectural comparison point, not a like-for-like reproduction.

## Results: Feature-Set Comparison (12 patients, mean ± SD RMSE)

| Horizon | Feature Set | N | RMSE |
|---|---|---|---|
| 15min | A: Glucose only | 12 | 16.03 ± 2.56 |
| 15min | C: Glucose + Carbs | 12 | 15.83 ± 2.74 |
| 15min | E: Glucose + Insulin | 12 | 15.97 ± 2.63 |
| 30min | A: Glucose only | 12 | 24.35 ± 3.66 |
| 30min | C: Glucose + Carbs | 12 | 23.77 ± 3.61 |
| 30min | E: Glucose + Insulin | 12 | 23.95 ± 3.15 |
| 60min | A: Glucose only | 12 | 36.89 ± 4.95 |
| 60min | **C: Glucose + Carbs** | 12 | **36.14 ± 5.34** |
| 60min | E: Glucose + Insulin | 12 | 36.15 ± 4.78 |

Feature sets B/D/F (heart-rate-dependent) run on the 6 2018 patients only, since 2020 patients have no HR/Steps data — see the full report for those results.

**Finding:** across all 12 patients, only carbohydrate intake produced a statistically significant improvement over glucose-only forecasting, and only at the 60-minute horizon (Wilcoxon p=0.021). No auxiliary signal reached significance at 15 or 30 minutes.

> **Revision note:** an earlier n=6 analysis in this repo's history reported that heart rate *significantly worsened* forecasts (p=0.031). That result did not replicate under the full 12-patient re-analysis with a corrected pipeline (A vs. B on the 6-patient 2018 subset: p=0.44–1.0 across horizons, not significant) and is treated as not reproducible — likely an artifact of the small original sample size rather than a genuine effect.

## Results: Architecture Comparison (12 patients, mean ± SD RMSE)

| Horizon | GRU | LSTM | BiLSTM | TCN | Transformer |
|---|---|---|---|---|---|
| 15min | **17.32 ± 2.75** | 18.81 ± 3.40 | 18.71 ± 3.16 | 24.89 ± 5.85 | 31.04 ± 4.66 |
| 30min | **24.89 ± 3.35** | 25.96 ± 3.85 | 26.16 ± 3.90 | 31.48 ± 4.20 | 36.58 ± 5.17 |
| 60min | **36.78 ± 4.50** | 37.40 ± 5.04 | 38.01 ± 5.21 | 41.18 ± 4.32 | 44.38 ± 6.18 |

**Finding:** GRU is the best-performing and clinically safest architecture (lowest Clarke Error Grid dangerous-zone rate) at every horizon. Its advantage over TCN and Transformer is large and highly significant (p<0.001, winning 12/12 patients at every horizon). Its advantage over LSTM/BiLSTM is smaller and **disappears at 60 min** (GRU vs. LSTM: p=0.38, not significant) — at longer horizons, the two are statistically indistinguishable.

Two patients (540, 567 — both 2020 cohort) are consistently the hardest to forecast across every architecture and horizon; patient 540 also shows the worst clinical-safety numbers (11.8% of 60-min predictions in dangerous CEGA zones with Transformer). See the full report for the per-patient breakdown.

## Full Report

The complete findings — patient-wise tables for both experiments, full statistical testing, CEGA zone breakdowns, and limitations — are in [`reports/CGM_Forecasting_Combined_Report.md`](reports/CGM_Forecasting_Combined_Report.md).

## Next Steps

- Shanghai dataset (T1DM/T2DM, 15-min sampling) — exploratory analysis complete; training pipeline not yet run.
- N-BEATS (univariate variant) and Temporal Fusion Transformer — not yet implemented.
- Cross-dataset generalizability comparison (Ohio vs. Shanghai) once both pipelines are complete.
