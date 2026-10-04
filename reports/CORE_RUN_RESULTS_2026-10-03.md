# Corrected Core Run — Preliminary Results (3 October 2026)

**Status: exploratory single-seed run; not publication-ready.**

This corrected-data run evaluates six architectures on all 12 OhioT1DM patients, five horizons, a 120-minute lookback, and an eight-epoch cap with early stopping. It is distinct from the full ten-architecture evaluation.

## Mean patient-level test metrics

MAE and RMSE are in mg/dL. SD is across the 12 patient-level values.

| Horizon | Architecture | MAE mean ± SD | RMSE mean ± SD | MARD mean (%) | Mean signed lag (min) |
|---:|---|---:|---:|---:|---:|
| 15 | Adaptive TCN-GRU-Transformer | 15.01 ± 4.50 | 21.21 ± 6.88 | 10.20 | 13.28 |
| 15 | TCN | 15.19 ± 6.49 | 21.08 ± 8.59 | 10.59 | 13.81 |
| 15 | GRU | 15.20 ± 6.04 | 21.51 ± 8.78 | 10.54 | 12.00 |
| 30 | TCN | 19.63 ± 5.92 | 26.83 ± 7.65 | 13.57 | 24.68 |
| 30 | GRU | 19.77 ± 6.85 | 27.34 ± 9.34 | 13.65 | 22.81 |
| 30 | Adaptive TCN-GRU-Transformer | 19.84 ± 5.14 | 27.28 ± 7.33 | 13.67 | 24.03 |
| 60 | Adaptive TCN-GRU-Transformer | 27.40 ± 5.08 | 37.18 ± 7.30 | 18.74 | 37.22 |
| 60 | GRU | 27.57 ± 6.83 | 37.36 ± 8.90 | 19.13 | 37.33 |
| 90 | GRU | 33.26 ± 5.20 | 44.13 ± 6.78 | 23.05 | 42.62 |
| 90 | Adaptive TCN-GRU-Transformer | 33.58 ± 4.83 | 44.50 ± 6.69 | 22.95 | 42.70 |
| 120 | GRU | 37.55 ± 4.65 | 48.90 ± 5.52 | 26.26 | 41.68 |
| 120 | Adaptive TCN-GRU-Transformer | 37.76 ± 3.84 | 49.29 ± 4.98 | 26.11 | 39.60 |

The complete six-architecture, five-horizon summary has 360 patient × architecture × horizon rows in the local reproducibility package.

## Interpretation

- Adaptive fusion does **not** dominate every horizon: TCN has the lowest mean MAE at 30 minutes, while GRU has lower mean MAE at 90 and 120 minutes.
- These short single-seed runs do not establish statistical significance, methodological novelty, clinical safety, or external validity.
- Signed lag is positive when the prediction trails the reference. The implementation is segment-aware, but lag should be interpreted as a diagnostic rather than a standalone clinical metric.
- Clarke Error Grid implementation and interpretation require independent expert review.
- Only results generated from corrected bolus CSVs should be used. Previous 2020 results produced from zero-bolus data are superseded.
