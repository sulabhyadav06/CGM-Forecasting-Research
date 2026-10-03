# Corrected Phase-2 Research Status — 2026-10-03

## Data-processing correction

OhioT1DM bolus events use the `ts_begin` attribute. The parser was corrected to read this field and the 24 Phase-2 train/test CSVs were regenerated. Redundant event-observation flags that were always 1 were also removed. All results produced from earlier zero-bolus CSVs are superseded and must not be used as final findings.

The OhioT1DM 2020 wearable XML has scalar acceleration events with `ts` and `value` attributes. `ohio2020_acceleration_preprocessing.py` aggregates them causally to the CGM grid over the trailing interval `[t-5 min, t)`, producing mean, standard deviation, minimum, maximum, count, coverage and missingness features. No future acceleration observations are used.

## Corrected evaluation protocol

- Cohorts: all 12 OhioT1DM patients (6 from 2018, 6 from 2020).
- 2018 inputs: CGM, carbohydrates, bolus/basal insulin, heart rate and steps.
- 2020 inputs: CGM, carbohydrates, bolus/basal insulin and acceleration.
- Forecast horizons: 15, 30, 60, 90 and 120 minutes (3, 6, 12, 18 and 24 five-minute steps).
- Gap-safe sequence construction requires exact timestamps and rejects windows crossing gaps; no target interpolation.
- Train-only imputation/scaling, chronological validation and a purge around the validation boundary.
- Shared multi-horizon representation with horizon-specific output heads.
- Signed lag convention: positive means the prediction trails the reference. Lag is calculated within continuous target-time segments, not across timestamp gaps.
- Clinical metrics: MAE, RMSE, MARD, signed time lag and Clarke Error Grid zone percentages.
- Adaptive fusion weights are learned across TCN, GRU and Transformer representations.

## Corrected core run

The corrected core run completed for 12 patients, 6 architectures, and all 5 horizons: 360 patient × model × horizon metric rows. Training used a fixed seed and an 8-epoch cap with early stopping. These are single-seed results, so they are exploratory rather than proof of general superiority.

Mean MAE (mg/dL), across patients:

| Horizon | Adaptive TCN-GRU-Transformer | GRU | TCN | LSTM | BiLSTM | Transformer |
|---:|---:|---:|---:|---:|---:|---:|
| 15 min | 15.01 | 15.20 | 15.19 | 16.20 | 16.97 | 16.28 |
| 30 min | 19.84 | 19.77 | 19.63 | 20.59 | 21.14 | 21.06 |
| 60 min | 27.40 | 27.57 | 27.70 | 28.55 | 28.40 | 28.09 |
| 90 min | 33.58 | 33.26 | 33.61 | 34.42 | 34.13 | 33.89 |
| 120 min | 37.76 | 37.55 | 38.15 | 38.47 | 38.07 | 38.27 |

The adaptive model is not the lowest-MAE model at every horizon. The results do not justify a blanket claim of hybrid superiority.

## Tests and implementation checks

The local regression suite checks five-horizon timestamp alignment, rejection of sequences crossing gaps, causal acceleration aggregation, parsing bolus timestamps from `ts_begin`, known synthetic lag, model output dimensions and adaptive gate normalization. The corrected local regression suite passed 9 tests.

## Expanded corrected evaluation

The corrected ten-architecture grid (LSTM, GRU, BiLSTM, TCN, Transformer, TCN-GRU, GRU-Transformer, TCN-Transformer, TCN-GRU-Transformer, adaptive TCN-GRU-Transformer) plus persistence has now completed on all 12 patients at all five horizons: 660 patient × model/baseline × horizon rows. See [the full architecture summary](FULL_ARCHITECTURE_RESULTS_2026-10-03.md). Persistence is better at 15/30 minutes; the adaptive model is not universally best. Corrected feature ablations and paired statistics are being rerun after an inverse-target-scaling bug was found in an exploratory ablation script; those earlier ablation outputs must be discarded. The focused 15-/120-minute Transformer lookback sweeps are also being rerun after fixing patient-specific scaler handling.
- Statistical comparisons use paired patient-level values, Holm correction, and rank-biserial effect sizes; signed lag is reported descriptively and absolute lag magnitude is compared. Treat the full statistical output as provisional until the corrected ablation and lookback runs finish.
- Clarke Error Grid implementation and clinical interpretation need expert review before publication.
- A single seed and one small dataset cohort do not establish methodological novelty, external validity, or clinical safety.
- OhioT1DM raw data are not included in this repository because of the data-use agreement.
