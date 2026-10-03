# Full Phase-2 CGM Forecasting Evaluation — 2026-10-03

## Status and scope
- Completed 198 model/feature-set fits across OhioT1DM 2018 and 2020; 594 patient × horizon metric rows; zero training failures.
- Forecast horizons: 15, 30, and 60 minutes; 120-minute lookback.
- Architecture grid: LSTM, GRU, BiLSTM, TCN, Transformer, TCN-GRU, GRU-Transformer, TCN-Transformer, TCN-GRU-Transformer, adaptive TCN-GRU-Transformer.
- Common-feature ablations: glucose; glucose+carbs; glucose+insulin; glucose+carbs+insulin. Cohort-specific feature sets also use 2018 heart rate/steps and 2020 acceleration.
- Maximum 15 epochs, early stopping patience 4, batch size 512, hidden dimension 32, seed 42.
- Chronological 85/15 split within each training CSV. Imputation/scaling fit only on the training partition; training targets cannot cross the split; validation targets begin at/after the split.
- Exact 5-minute contiguous windows only; no interpolation; target glucose is not imputed.
- Sleep was not ablated because is_sleeping is absent from the supplied Phase-2 CSVs.

## Validation
- Independent checks: 594 horizon checks and 72 segment-aware time-lag checks; zero validation failures.
- Synthetic 15-minute delay test returned +15 minutes.
- Clarke Error Grid implementation agreed with the existing helper on the validation sample. This is a software consistency check, not clinical certification; zone boundaries should be reviewed by a qualified clinical-metrics reviewer before publication.
- MAE/RMSE were independently recomputed from saved predictions. The full run supersedes earlier 3-epoch preliminary results.

## Architecture results
Lowest mean MAE architecture by cohort and horizon (mg/dL):

| Cohort | Horizon | Model | MAE | RMSE | MARD (%) |
|---|---:|---|---:|---:|---:|
| OhioT1DM 2018 | 15 min | adaptive TCN-GRU-Transformer | 13.61 | 19.33 | 8.73 |
| OhioT1DM 2018 | 30 min | GRU | 18.09 | 25.23 | 11.70 |
| OhioT1DM 2018 | 60 min | adaptive TCN-GRU-Transformer | 25.86 | 35.13 | 16.86 |
| OhioT1DM 2020 | 15 min | adaptive TCN-GRU-Transformer | 14.47 | 20.93 | 10.00 |
| OhioT1DM 2020 | 30 min | adaptive TCN-GRU-Transformer | 19.56 | 27.08 | 13.65 |
| OhioT1DM 2020 | 60 min | adaptive TCN-GRU-Transformer | 28.09 | 37.65 | 19.72 |

These are descriptive cohort means, not evidence of general superiority or clinical safety.

## Comparison with last-observation persistence
Positive improvement means lower MAE than persistence.

| Cohort | Horizon | Fits beating persistence | Mean MAE improvement (mg/dL) |
|---|---:|---:|---:|
| OhioT1DM 2018 | 15 min | 0/60 | -5.26 |
| OhioT1DM 2018 | 30 min | 1/60 | -2.59 |
| OhioT1DM 2018 | 60 min | 39/60 | +0.43 |
| OhioT1DM 2020 | 15 min | 0/60 | -6.59 |
| OhioT1DM 2020 | 30 min | 10/60 | -3.50 |
| OhioT1DM 2020 | 60 min | 37/60 | +0.64 |

The results indicate the tested models generally do not beat persistence at 15 minutes, rarely beat it at 30 minutes, and offer only modest average improvement at 60 minutes. The next methodological step should focus on residual/change forecasting and patient-level validation, not claiming model superiority from these results.

## Common feature ablation across all 12 patients
Mean MAE (mg/dL), ordered 15 / 30 / 60 minutes:
- Glucose only: 11.21 / 17.23 / 27.10
- Glucose + carbs: 12.04 / 17.28 / 26.24
- Glucose + insulin: 12.47 / 17.76 / 26.53
- Glucose + carbs + insulin: 12.82 / 17.87 / 26.29

Glucose-only was better on average at 15 and 30 minutes. Carbs/insulin features yielded small average differences at 60 minutes; effects vary by patient. Do not claim universal benefit from auxiliary inputs.

## Feature availability and limitations
- 2018 heart-rate statistics have about 6.6% missingness; steps statistics about 6.5%.
- 2020 acceleration statistics have about 34.5% missingness.
- Missing inputs are median-imputed using the training partition only.
- Patient-level paired comparisons have small sample sizes and should be considered exploratory.
- This run saved prediction CSVs and training histories, but not reusable model checkpoints.
- OhioT1DM source data are not included because of dataset access restrictions.

## Reproducibility files
The evaluation bundle includes the full patient-level metrics, per-model prediction CSVs and training histories, feature-availability audit, persistence comparisons, independent-validation outputs, test scripts, configuration, logs, and Python source under `reproducibility/`. See the repository's `reproducibility/README_FULL_EVALUATION.md` for run instructions.
