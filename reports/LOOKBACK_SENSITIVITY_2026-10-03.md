# Transformer Lookback Sensitivity — 3 October 2026

Exploratory one-seed, five-epoch study on the six OhioT1DM 2018 patients. The scaler/imputer is fit per patient on chronological pre-validation training rows only.

| Forecast horizon | Lookback | MAE mean ± SD (mg/dL) | RMSE mean ± SD (mg/dL) |
|---:|---:|---:|---:|
| 15 min | 60 min | 12.29 ± 3.14 | 17.62 ± 4.06 |
| 15 min | 120 min | 12.32 ± 2.91 | 17.83 ± 4.31 |
| 15 min | 180 min | 12.73 ± 3.23 | 18.45 ± 4.67 |
| 15 min | 240 min | 14.27 ± 3.74 | 20.15 ± 5.04 |
| 120 min | 60 min | 37.99 ± 2.21 | 49.66 ± 4.57 |
| 120 min | 120 min | 37.99 ± 4.17 | 50.21 ± 6.12 |
| 120 min | 180 min | 40.33 ± 2.19 | 52.56 ± 5.38 |
| 120 min | 240 min | 40.10 ± 3.08 | 51.74 ± 5.54 |

A 60-minute context had the lowest mean MAE at both tested horizons, with 120 minutes nearly tied. Longer windows did not help in this short run. This is a focused Transformer experiment on the 2018 cohort, not proof that one context length is optimal across all architectures, patients or datasets.
