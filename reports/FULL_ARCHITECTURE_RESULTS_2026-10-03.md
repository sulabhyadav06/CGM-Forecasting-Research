# Corrected Ten-Architecture Evaluation — Preliminary Summary (3 October 2026)

**Status: the corrected-data computation completed, but these are exploratory single-seed, short-epoch results—not final publication findings.**

The run covers all 12 OhioT1DM patients, ten neural architectures, last-observation persistence, five horizons (15/30/60/90/120 minutes), and a 120-minute lookback. It produced 660 patient × architecture/baseline × horizon metric rows. Training was capped at five epochs with early stopping and one fixed seed.

## Best neural model by mean MAE vs persistence

MAE and RMSE are in mg/dL. MAE SD is across the 12 patient-level values.

| Horizon | Best neural model | Neural MAE mean ± SD | Neural RMSE mean | Persistence MAE mean | Persistence RMSE mean |
|---:|---|---:|---:|---:|---:|
| 15 min | Adaptive TCN-GRU-Transformer | 15.58 ± 4.62 | 21.94 | 9.64 | 13.61 |
| 30 min | TCN | 20.14 ± 6.00 | 27.54 | 17.09 | 23.53 |
| 60 min | Adaptive TCN-GRU-Transformer | 27.74 ± 5.04 | 37.52 | 28.61 | 38.46 |
| 90 min | TCN-GRU | 33.34 ± 4.84 | 44.04 | 36.95 | 48.87 |
| 120 min | TCN-GRU | 37.61 ± 4.15 | 48.78 | 43.35 | 56.47 |

## Interpretation

- Persistence is substantially better on average at 15 and 30 minutes; neural models do not beat it at these horizons.
- The adaptive hybrid has a modest mean-MAE advantage over persistence at 60 minutes (27.74 vs 28.61 mg/dL). Paired patient-level tests are needed to determine whether that difference is reliable.
- TCN-GRU is the lowest-MAE neural model at 90 and 120 minutes in this run. The adaptive hybrid is not the universal winner.
- This run is a reproducibility pass with a five-epoch cap, not a converged or fully tuned experiment. Rankings can change with repeated seeds and tuning.
- Statistical significance, Holm correction, effect sizes, patient-level difficulty analysis, corrected feature ablations and lookback studies should be reviewed before a paper claim.
- Data audit: no duplicate timestamps or missing glucose values in the 24 stored files; timestamp gaps over 30 minutes occur in every file and must remain discontinuities. The sequence builder rejects windows crossing gaps.
- The sleep feature is not available in the corrected Phase-2 CSVs and was not evaluated.
- Clarke Error Grid implementation and clinical interpretation require independent expert review. This is not a clinical decision-support system.
