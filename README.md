# CGM Forecasting Research

> **Forecast-alignment correction (2026-10-03):** The Phase-2 hybrid and ablation runners were updated on branch `fix/forecast-alignment-validation` to align target timestamps with the requested horizon, exclude training anchors whose targets cross the validation boundary, and reject windows spanning missing/non-5-minute intervals. Existing result tables and saved outputs were generated before these corrections and **must be treated as historical, not corrected results**. Re-run the affected experiments and regenerate reports before drawing scientific conclusions from the corrected pipeline.

## Reference Paper

This project is now a completed research implementation and experimental package for personalized multimodal multi-horizon blood glucose forecasting.

The final implementation supports a hybrid deep-learning architecture and a systematic benchmark against recurrent, convolutional, and transformer-based baselines. The codebase, preprocessing pipeline, evaluation metrics, experimental outputs, and results summaries are all in place and the repository has been committed to git.

---

## Final proposed model

The final model implemented in the repository is a multimodal hybrid TCN–GRU–Transformer architecture with adaptive fusion and a multi-horizon prediction head.

Core design:
- TCN branch for local and short-range temporal dynamics
- GRU branch for sequential temporal dependence
- Transformer branch for longer-range cross-timestep interactions
- Adaptive gated fusion to weight the three representations
- Multi-horizon output head producing forecasts at 15, 30, 60, 90, and 120 minutes

The model is defined in [hybrid_models.py](hybrid_models.py), with the implementation class `HybridTCNGRUTransformer`.

---

## What was verified

The project includes a full set of validated experimental outputs in the `output/` directory. The most relevant files are:

- [output/phase2_gap_safe_architecture/architecture_mean_sd.csv](output/phase2_gap_safe_architecture/architecture_mean_sd.csv)
- [output/phase2_hybrid_all12_run2/all12_summary.csv](output/phase2_hybrid_all12_run2/all12_summary.csv)
- [output/phase2_hybrid_ablations/patient_559_all_ablations.csv](output/phase2_hybrid_ablations/patient_559_all_ablations.csv)

These outputs were validated directly from the generated CSVs. The final evidence shows:

- Short horizons favor TCN
- Longer horizons become competitive for GRU-Transformer patterns
- No single architecture dominates every horizon uniformly
- The hybrid model remains a viable and well-structured baseline, but not a universal winner in the current dataset

---

## Evidence-backed main findings

From the generated architecture comparison:

- 15 min: TCN best RMSE = 20.258
- 30 min: TCN best RMSE = 27.295
- 60 min: TCN best RMSE = 38.864
- 90 min: GRU-Transformer best RMSE = 46.852
- 120 min: GRU-Transformer best RMSE = 51.064

From the hybrid all-12 summary:

- 15 min: RMSE = 23.16 ± 10.54
- 30 min: RMSE = 29.23 ± 9.57
- 60 min: RMSE = 40.20 ± 8.99
- 90 min: RMSE = 48.18 ± 8.83
- 120 min: RMSE = 53.06 ± 7.89

This supports the final conclusion that architecture performance is horizon-dependent rather than globally consistent.

---

## Final research interpretation

The strongest defensible scientific statement is:

> A multimodal gap-safe forecasting framework for personalized blood glucose prediction was implemented and evaluated across multiple architectures and horizons. The hybrid TCN–GRU–Transformer design is the final proposed model, but the empirical evidence supports horizon-dependent performance rather than universal superiority of any single architecture.

This is a stronger and more defensible conclusion than claiming the hybrid model is uniformly best across all conditions.

---

## Repository structure

Key project files:
- [hybrid_models.py](hybrid_models.py) — final hybrid architecture
- [hybrid_experiments.py](hybrid_experiments.py) — training and evaluation pipeline
- [phase2_gap_safe_architecture_ablation.py](phase2_gap_safe_architecture_ablation.py) — architecture comparison workflow
- [phase2_feature_ablation.py](phase2_feature_ablation.py) — feature-level ablation experiments
- [clinical_metrics.py](clinical_metrics.py) — MAE, RMSE, MARD, time lag, CEGA
- [test_time_lag_metric.py](test_time_lag_metric.py) — metric validation tests
- [mini_paper.md](mini_paper.md) — paper-ready summary draft



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
