# GluNet-Inspired Huber Regression: Reproducibility Record

**Date:** 2026-10-21

**Status:** Experimental baseline audited; not an exact reproduction of the published GluNet model.

## 1. Executive summary

A personalized continuous-regression baseline was evaluated on the 12 OhioT1DM
patients at 30-minute and 60-minute forecasting horizons. Three training seeds
(7, 42, and 123) were run for every patient and horizon. Persistence forecasting
was evaluated on the same test timestamps.

The regression baseline had lower mean patient-level MAE, RMSE, and MARD at
both horizons. Its mean time-lag estimate was also closer to zero after the
test sequences were segmented at timestamp gaps. The corrected Clarke Error
Grid reference comparison shows higher mean Zone A proportions, but it does
not establish improved performance for every clinical error category.

These results are preliminary evidence for this experimental baseline. They
do not establish statistical or clinical generalization beyond this cohort,
and they should not be described as an exact reproduction of GluNet.

## 2. Experimental protocol

- Cohort: 12 OhioT1DM patients (540, 544, 552, 567, 584, 596, 559, 563,
  570, 575, 588, 591).
- Forecast horizons: 30 and 60 minutes.
- Training seeds: 7, 42, 123.
- Input cadence: 5-minute CGM data.
- Lookback: 16 input steps.
- Features: glucose, carbohydrates over the previous 60 minutes, bolus insulin
  over the previous 60 minutes, and cyclical time-of-day features.
- Target: continuous future glucose minus current glucose, in mg/dL.
- Model/loss: GluNet-inspired convolutional regression model with Huber loss,
  delta=1.0.
- Training: up to 30 epochs, early stopping with patience 5, batch size 128,
  Adam learning rate 0.001.
- Validation: final 20% of each patient's training partition, separated
  chronologically before sequence construction.
- Preprocessing: imputation and standardization fitted on the training
  segment and applied to validation/test data; raw glucose retained for
  target construction and evaluation.
- Sequence handling: sequences are constructed within contiguous segments;
  target timestamps are looked up exactly at the requested horizon.
- Comparator: persistence prediction, using current glucose as the forecast.

The predictions use the same patient/horizon test timestamps across all three
seeds. The saved run configurations, metric rows, and prediction files passed
the recorded consistency checks.

## 3. Final metrics

Values are the equal-weight mean and sample standard deviation across the
12 patients, after averaging each patient's three seeds first. MAE and RMSE
are in mg/dL. MARD and Clarke zones are percentages. Time lag is the
gap-aware estimate described in Section 4.

### 30-minute horizon

| Metric | Huber regression, mean +/- SD | Persistence, mean +/- SD |
|---|---:|---:|
| MAE (mg/dL) | 13.526 +/- 1.628 | 16.861 +/- 2.054 |
| RMSE (mg/dL) | 19.318 +/- 2.501 | 23.234 +/- 2.893 |
| Signed error (mg/dL) | -1.237 +/- 1.605 | 0.018 +/- 0.523 |
| MARD (%) | 9.114 +/- 1.745 | 11.339 +/- 2.100 |
| Gap-aware time lag (min) | 21.528 +/- 3.922 | 30.000 +/- 0.000 |
| Clarke Zone A (%) | 89.265 +/- 4.154 | 83.435 +/- 5.921 |
| Clarke Zone B (%) | 9.790 +/- 3.549 | 15.510 +/- 5.358 |
| Clarke Zone C (%) | 0.007 +/- 0.014 | 0.017 +/- 0.031 |
| Clarke Zone D (%) | 0.939 +/- 0.712 | 1.039 +/- 0.721 |
| Clarke Zone E (%) | 0.000 +/- 0.000 | 0.000 +/- 0.000 |

### 60-minute horizon

| Metric | Huber regression, mean +/- SD | Persistence, mean +/- SD |
|---|---:|---:|
| MAE (mg/dL) | 23.507 +/- 3.419 | 28.293 +/- 3.522 |
| RMSE (mg/dL) | 32.356 +/- 4.375 | 38.101 +/- 4.750 |
| Signed error (mg/dL) | -2.184 +/- 4.051 | 0.022 +/- 1.108 |
| MARD (%) | 15.907 +/- 3.342 | 19.243 +/- 3.598 |
| Gap-aware time lag (min) | 46.111 +/- 9.753 | 60.000 +/- 0.000 |
| Clarke Zone A (%) | 72.388 +/- 8.698 | 64.784 +/- 8.403 |
| Clarke Zone B (%) | 24.730 +/- 7.177 | 31.942 +/- 7.052 |
| Clarke Zone C (%) | 0.151 +/- 0.109 | 0.651 +/- 0.598 |
| Clarke Zone D (%) | 2.708 +/- 1.699 | 2.561 +/- 1.461 |
| Clarke Zone E (%) | 0.022 +/- 0.037 | 0.062 +/- 0.084 |

These are macro averages across patients, not pooled-window metrics. Zone
percentages were calculated using the existing detailed Clarke implementation
in `experiments/crnn/error_grids_reference.py`, with its numeric codes mapped
to zones A-E. That file is a reference implementation already present in the
repository, not an official implementation endorsed by the original GluNet
authors.

## 4. Time-lag methodology correction

The original `clinical_metrics.py` calculation concatenates the prediction
rows and searches cross-correlation lags. Prediction rows can be separated by
real timestamp gaps, so treating the full file as a continuous 5-minute signal
can pair observations from different segments.

A separate gap-aware analysis was therefore performed. It sorts by target
timestamp, splits at intervals differing from the expected 5-minute cadence
by more than 30 seconds, and uses only within-segment pairs. The original
source and run files were not modified.

The corrected mean absolute-lag improvements for regression relative to
persistence were:

- 30 minutes: 8.472 minutes; patient-bootstrap 95% CI 6.528 to 10.694.
- 60 minutes: 13.889 minutes; patient-bootstrap 95% CI 8.889 to 19.444.

The estimates were unchanged when the maximum lag search was increased from
60 to 120 and 180 minutes. Persistence's estimated lag equalled the forecast
horizon. Cross-correlation lag is a signal-timing diagnostic, not a standalone
measure of forecasting accuracy or clinical benefit.

## 5. Exploratory paired patient analysis

Three seeds were averaged within each patient. Paired comparisons then used
the 12 patient-level means. Bootstrap intervals used 20,000 resamples of
patients. The one-sided Wilcoxon tests were exploratory; p-values were not
adjusted for multiple comparisons.

Main paired improvements (positive favors regression):

| Horizon | Metric | Mean improvement | Bootstrap 95% CI | Exploratory p |
|---|---|---:|---:|---:|
| 30 min | MAE (mg/dL) | 3.335 | 2.593 to 4.167 | 0.0002 |
| 30 min | RMSE (mg/dL) | 3.916 | 2.854 to 5.059 | 0.0002 |
| 30 min | MARD (percentage points) | 2.226 | 1.627 to 2.875 | 0.0002 |
| 30 min | Absolute time lag (min) | 8.472 | 6.528 to 10.694 | 0.0002 |
| 30 min | Clarke Zone A (percentage points) | 5.830 | 3.970 to 7.917 | 0.0002 |
| 60 min | MAE (mg/dL) | 4.785 | 3.474 to 6.172 | 0.0002 |
| 60 min | RMSE (mg/dL) | 5.745 | 4.150 to 7.375 | 0.0002 |
| 60 min | MARD (percentage points) | 3.336 | 2.223 to 4.419 | 0.0005 |
| 60 min | Absolute time lag (min) | 13.889 | 8.889 to 19.444 | 0.0002 |
| 60 min | Clarke Zone A (percentage points) | 7.604 | 4.876 to 10.483 | 0.0002 |

The bootstrap unit was the patient, not the overlapping forecast window. The
small cohort size, fixed temporal test partitions, multiple outcomes, and
exploratory analysis mean that these statistics should not be presented as
definitive proof of generalization.

The corrected Clarke comparisons indicate no conclusive Zone D improvement:
30-minute mean difference (persistence minus regression) was +0.100 percentage
points, with CI -0.090 to +0.289. At 60 minutes it was -0.147 percentage
points, with CI -0.569 to +0.234. Zone E differences were also inconclusive.
Improved MAE/MARD and Zone A do not establish that every clinically important
error category is improved.

## 6. Saved forecast domain audit

Across the saved regression runs:

- All forecast values were finite.
- No negative or exactly-zero forecast values were found.
- At 30 minutes, 13 regression predictions exceeded 400 mg/dL across the
  three-seed run set.
- At 60 minutes, one regression prediction exceeded 400 mg/dL.
- No predictions exceeded 600 mg/dL.
- Persistence forecasts did not exceed 400 mg/dL.

Values above 400 mg/dL are flagged for review here, not automatically removed.
The domain review must use an explicitly stated clinical/evaluation policy.

## 7. Reproducibility checks

Completed checks:

- 72 expected patient/horizon/seed run keys were found: 12 patients, two
  horizons, and three seeds.
- All 72 run records passed local artifact, row-count, and arithmetic checks.
- Test-window counts and target signatures aligned across seeds for all
  patient/horizon combinations.
- Prediction reconstruction and error-column checks passed.
- The existing detailed Clarke reference implementation was compared with
  the root metric implementation. Boundary differences were small in
  aggregate, so the corrected tables use the detailed implementation.
- The full repository test suite passed: 71 tests and 20 subtests in 3.50 s.
- The negative, zero, and non-finite forecast audit was reconciled with the
  separate above-400 mg/dL check.

## 8. Important differences from the published GluNet method

This work is an experimental regression baseline inspired by the GluNet
workflow; it is not an exact reproduction of the paper's principal
probabilistic classifier.

Differences and unresolved items include:

1. The paper's 256-class probabilistic glucose-change output is replaced here
   by continuous delta prediction with Huber loss.
2. The 12-patient cohort is an extension beyond the six subjects used in the
   paper's OhioT1DM experiment.
3. The feature set includes glucose, recent carbohydrate and bolus features,
   and cyclical time-of-day features. Exact timestamp encoding in the paper
   is not specified sufficiently to claim equivalence.
4. The current conservative segmentation policy does not reproduce the
   paper's described interpolation/extrapolation of short CGM gaps.
5. Exact paper-specific outlier-correction thresholds and procedures remain
   unresolved.
6. The exact interpretation of the architecture's layer count and the
   placement/use of the paper's mu-law discussion remain unresolved.
7. The reference Clarke implementation is independent of the GluNet paper's
   model implementation and has not been endorsed by the original authors.

Do not describe the results as an exact GluNet reproduction until these
fidelity questions have been resolved or the adaptations have been explicitly
accepted as the study protocol.

## 9. Audit artifacts

Principal outputs retained under `reports/` include:

- `glunet_repro_audit_runs_20261021_155754.csv`
- `glunet_repro_audit_alignment_20261021_155754.csv`
- `glunet_regression_multiseed_detailed.csv`
- `glunet_regression_multiseed_corrected.csv`
- `glunet_regression_corrected_summary.csv`
- `glunet_gap_aware_time_lag_detail.csv`
- `glunet_gap_aware_time_lag_summary.csv`
- `glunet_gap_aware_time_lag_statistics.csv`
- `glunet_regression_multiseed_clarke_corrected.csv`
- `glunet_regression_clarke_corrected_summary.csv`
- `glunet_regression_clarke_corrected_stats.csv`
- `glunet_clarke_actual_reference_audit.csv`
- `glunet_clarke_actual_reference_mismatches.csv`
- `glunet_gap_aware_lag_sensitivity_detail.csv`
- `glunet_gap_aware_lag_sensitivity_summary.csv`
- `glunet_gap_aware_lag_boundary_counts.csv`
- `glunet_regression_h30_patient_metrics_20261021_141821.csv`
- `glunet_regression_h60_patient_metrics.csv`

The original timestamped run directories and predictions have been preserved.
The audit CSVs and this Markdown record are derived analysis artifacts; they
do not replace the original run outputs.

## 10. Recommended next work

1. Keep this baseline and its reports frozen.
2. Review flagged predictions above 400 mg/dL with timestamps and surrounding
   CGM/event context; do not auto-correct or drop them without a prespecified
   rule.
3. Resolve paper-fidelity questions from the publication and document every
   adaptation.
4. If pursuing a paper-faithful reproduction, implement the classifier and
   paper-specific preprocessing as a separately named experiment so the
   present continuous-regression results remain intact.
5. For further performance claims, use prespecified analyses and independent
   validation wherever available.
