# Corrected Feature Ablation Results — 3 October 2026

**Exploratory one-seed, five-epoch results; not publication-ready.** The corrected ablation produced 390 patient × feature-set × horizon rows, using raw glucose for target scaling, train-only imputation/scaling, chronological validation, and a purge for maximum-horizon targets.

## Common-feature MAE (mg/dL), averaged over all 12 patients

| Horizon | Glucose | Glucose + carbs | Glucose + insulin | Glucose + carbs + insulin |
|---:|---:|---:|---:|---:|
| 15 min | 18.62 | 20.56 | 19.65 | 20.76 |
| 30 min | 23.95 | 23.62 | 24.14 | 23.97 |
| 60 min | 32.08 | 31.61 | 31.43 | 31.74 |
| 90 min | 37.31 | 37.17 | 36.85 | 36.93 |
| 120 min | 40.84 | 40.64 | 40.46 | 40.65 |

## Cohort-specific feature sets

- OhioT1DM 2018: heart rate, steps, carbs and insulin were tested in six patients. The full feature set mean MAE was 15.18, 19.79, 27.47, 33.20 and 37.69 mg/dL at 15/30/60/90/120 minutes.
- OhioT1DM 2020: acceleration, carbs and insulin were tested in six patients. The full feature set mean MAE was 24.82, 28.30, 35.16, 39.93 and 42.79 mg/dL at those horizons. Acceleration-only was worse in this short run; the stream has substantial missingness, so this does not establish that acceleration is unhelpful.
- `is_sleeping` was not available in the corrected CSVs, so sleep ablation was not possible.

## Paired tests

Paired Wilcoxon comparisons were made on matched patients with Holm correction and rank-biserial effect sizes. **No feature-set contrast was significant for MAE after Holm correction.** Some differences remained for RMSE, MARD and absolute time lag, but these are exploratory and metric-specific. Do not claim a modality improves forecasting based on these short runs alone.
