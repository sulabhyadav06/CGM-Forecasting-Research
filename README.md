# Personalized Blood Glucose Forecasting

> **Forecast-alignment correction (2026-10-03):** The Phase-2 hybrid and ablation runners were updated on branch `fix/forecast-alignment-validation` to align target timestamps with the requested horizon, exclude training anchors whose targets cross the validation boundary, and reject windows spanning missing/non-5-minute intervals. Existing result tables and saved outputs were generated before these corrections and **must be treated as historical, not corrected results**. Re-run the affected experiments and regenerate reports before drawing scientific conclusions from the corrected pipeline.

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
6. **`CGM_Forecasting_Experiments.ipynb`** — documents the experimental configuration, preprocessing workflow, evaluation metrics, patient-level results, statistical analysis, Shanghai experiments, and cross-dataset comparison.

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


## Shanghai Dataset and Cross-Dataset Analysis

The Shanghai T1DM dataset was subsequently processed using a dedicated preprocessing and architecture-comparison pipeline.

The experiment includes:

- 12 patients
- 15-, 30-, and 60-minute forecasting horizons
- LSTM, GRU, BiLSTM, TCN, and Transformer architectures
- RMSE, MAE, MARD, Time Lag, and Clarke Error Grid Analysis (CEGA)

The resulting experiment contains 180 patient-level results:

**12 patients × 3 horizons × 5 architectures = 180 results**

Patient-level results and Mean ± SD summaries are retained separately.

A cross-dataset comparison between OhioT1DM and Shanghai was also performed to examine differences in architecture performance across the two datasets. These results are treated as dataset-level comparisons rather than assuming direct transferability between datasets.

## Full Report

The complete findings — patient-wise tables for both experiments, full statistical testing, CEGA zone breakdowns, and limitations — are in [`reports/CGM_Forecasting_Combined_Report.md`](reports/CGM_Forecasting_Combined_Report.md).

The experimental workflow and generated results are also documented in [`CGM_Forecasting_Experiments.ipynb`](CGM_Forecasting_Experiments.ipynb).



## Corrected full Phase-2 evaluation (2026-10-03)

**Use [the corrected full evaluation report](reports/phase2_full_evaluation_2026-10-03.md) for the latest run.** The older result tables elsewhere in this README describe earlier experiments and must not be treated as results from the corrected full-grid run.

The corrected run covers all 12 OhioT1DM patients, 10 architecture candidates, three horizons (15/30/60 minutes), and common-feature ablations. It records 198 model/feature fits and 594 patient-horizon metric rows, with no recorded training failures. Exact 5-minute contiguous windows, chronological train/validation boundaries, and horizon-aligned targets are enforced. Independent checks reported zero validation failures.

The corrected descriptive results do **not** support a blanket claim that one architecture is best or that multimodal inputs always help:
- The lowest mean-MAE architecture varied by cohort and forecast horizon.
- Tested models generally did not beat last-observation persistence at 15 minutes, rarely did at 30 minutes, and achieved only modest mean improvements at 60 minutes.
- Glucose-only forecasts had lower average MAE than the common multimodal feature sets at 15 and 30 minutes; auxiliary inputs showed small, patient-dependent differences at 60 minutes.
- These are research metrics, not clinical validation. Patient-level comparisons have small sample sizes, and clinical error-grid implementation details require expert review before publication.

The report describes the run protocol, feature availability, persistence comparison, limitations, and artifact inventory. The result bundle contains patient-level metrics, predictions, histories, audits, validation outputs, and logs in the local evaluation package; large generated artifacts are not all tracked in GitHub yet. OhioT1DM source data remain excluded under the dataset access agreement.


## Corrected Phase-2 status (3 October 2026)

- Corrected OhioT1DM bolus parsing: bolus events use `ts_begin`. Regenerated 2020 CSVs supersede earlier runs with zero bolus values.
- 2018 multimodal inputs: CGM, carbohydrates, insulin, heart rate and steps. 2020 inputs: CGM, carbohydrates, insulin and causal acceleration summaries.
- The model implementation supports five forecast horizons (15/30/60/90/120 min) and adaptive TCN–GRU–Transformer fusion.
- Regression tests and a reproducible runbook are now included. Raw OhioT1DM data must not be committed.
- See [corrected project status](reports/FINAL_PROJECT_STATUS_2026-10-03.md) and [reproducible runbook](reports/REPRODUCIBLE_RUNBOOK_2026-10-03.md).

**Research caution:** preliminary/short-epoch results do not establish superiority, novelty or clinical safety. Use only corrected-data results, report patient-level comparisons with multiplicity correction, and have the Clarke Error Grid implementation independently reviewed before publication.

- [Corrected core-run results (preliminary, six architectures)](reports/CORE_RUN_RESULTS_2026-10-03.md)

- [Corrected data-quality audit](reports/DATA_QUALITY_AUDIT_2026-10-03.md) — timestamp gaps, removed constant flags, missingness, and sleep-data limitation.

- [Corrected ten-architecture results and persistence comparison](reports/FULL_ARCHITECTURE_RESULTS_2026-10-03.md) — five horizons, 12 patients, and limitations.

- [Corrected feature ablation results](reports/FEATURE_ABLATION_RESULTS_2026-10-03.md) — common and cohort-specific modality sets.
- [Transformer lookback sensitivity](reports/LOOKBACK_SENSITIVITY_2026-10-03.md) — 60/120/180/240-minute windows at 15/120-minute horizons.
- [Paired statistical analysis](reports/STATISTICAL_ANALYSIS_2026-10-03.md) — Holm-corrected tests and effect-size summary.
