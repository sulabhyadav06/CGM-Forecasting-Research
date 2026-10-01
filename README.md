# CGM Forecasting Research

## Project status

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

---

## Final status

This project is complete as a research package, with:
- implemented multimodal preprocessing
- final hybrid model architecture
- validated benchmark outputs
- result summaries and paper-ready narrative
- repository commit recorded in git

The remaining work, if desired, is purely editorial: turning the existing results into a polished journal article, conference abstract, or final thesis chapter.
