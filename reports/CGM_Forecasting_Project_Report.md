# CGM Forecasting Project Report

## 1. Project status

This repository contains the completed implementation and validation workflow for personalized multimodal multi-horizon blood glucose forecasting in Type 1 diabetes. The project combines data preprocessing, patient-level evaluation, architecture comparison, hybrid modeling, clinical metrics, and result synthesis.

The original historical notebook, [../CGM_Forecasting_Experiments.ipynb](../CGM_Forecasting_Experiments.ipynb), is retained as the preserved project record. The current experimental workflow continues through the validated scripts, outputs, and final report materials, rather than through a separate replacement notebook.

The project is in a verified evidence-based state for its core implementation and architecture benchmark workflow: the preprocessing pipeline, causal acceleration handling, feature-availability gating, and configuration-controlled architecture comparisons have all been validated against real repository outputs. The remaining open issue is not the implementation itself but the strength of the final novelty and component-contribution claims, which still require additional paired ablation evidence.

---

## 2. Final proposed model

The final modeling direction implemented in the codebase is a hybrid TCN–GRU–Transformer architecture with adaptive fusion and a multi-horizon forecast head.

Core design elements:
- TCN branch for short-range temporal dynamics
- GRU branch for sequential dependence
- Transformer branch for longer-range interactions
- Adaptive fusion layer for combining branch features
- Multi-horizon predictions for 15, 30, 60, 90, and 120 minutes

Key implementation files:
- [../hybrid_models.py](../hybrid_models.py)
- [../hybrid_experiments.py](../hybrid_experiments.py)
- [../hybrid_ablation_experiments.py](../hybrid_ablation_experiments.py)
- [../phase2_hybrid_model.py](../phase2_hybrid_model.py)

---

## 3. Notebook history

The primary historical notebook is:
- [../CGM_Forecasting_Experiments.ipynb](../CGM_Forecasting_Experiments.ipynb)

This is the preserved notebook corresponding to the previous project commit. It captures the original project workflow and the research progression leading to the final validated setup.

---

## 4. Core documentation files

Top-level project documentation:
- [../README.md](../README.md)
- [../mini_paper.md](../mini_paper.md)
- [../README_BASELINES.md](../README_BASELINES.md)
- [../README_PHASE2_GAP_SAFE.md](../README_PHASE2_GAP_SAFE.md)
- [../README_PHASE2_TRANSFORMER.md](../README_PHASE2_TRANSFORMER.md)

These files summarize the project scope, benchmark setup, gap-safe architecture evaluation, transformer study, and the final research narrative.

---

## 5. Research and modeling scripts

The repository includes the core experimental workflow and model scripts:

### Main modeling and experiments
- [../hybrid_models.py](../hybrid_models.py)
- [../hybrid_models_v2.py](../hybrid_models_v2.py)
- [../hybrid_experiments.py](../hybrid_experiments.py)
- [../hybrid_ablation_experiments.py](../hybrid_ablation_experiments.py)
- [../phase2_hybrid_model.py](../phase2_hybrid_model.py)

### Baselines and comparisons
- [../baseline_experiments.py](../baseline_experiments.py)
- [../transformer_experiments.py](../transformer_experiments.py)
- [../capacity_controlled_architecture_sweep.py](../capacity_controlled_architecture_sweep.py)
- [../matched_gru_joint_validation.py](../matched_gru_joint_validation.py)
- [../multimodel_architecture_compare.py](../multimodel_architecture_compare.py)
- [../multimodel_compare.py](../multimodel_compare.py)
- [../multimodel_compare_all.py](../multimodel_compare_all.py)
- [../multimodel_lstm.py](../multimodel_lstm.py)
- [../lstm_model.py](../lstm_model.py)
- [../improved_transformer.py](../improved_transformer.py)

### Data preprocessing and utilities
- [../phase2_multimodal_preprocessing.py](../phase2_multimodal_preprocessing.py)
- [../phase2_sequence_utils.py](../phase2_sequence_utils.py)
- [../phase2_data_quality_audit.py](../phase2_data_quality_audit.py)
- [../phase2_feature_ablation.py](../phase2_feature_ablation.py)
- [../phase2_gap_safe_architecture_ablation.py](../phase2_gap_safe_architecture_ablation.py)
- [../parse_xml.py](../parse_xml.py)
- [../add_sleep_to_phase2.py](../add_sleep_to_phase2.py)
- [../shanghai_preprocessing.py](../shanghai_preprocessing.py)
- [../ohio2020_acceleration_preprocessing.py](../ohio2020_acceleration_preprocessing.py)

### Metrics and validation
- [../clinical_metrics.py](../clinical_metrics.py)
- [../test_time_lag_metric.py](../test_time_lag_metric.py)
- [../significance_test.py](../significance_test.py)
- [../transformer_paired_statistics.py](../transformer_paired_statistics.py)
- [../lookback_60_vs_120_statistics.py](../lookback_60_vs_120_statistics.py)
- [../transformer_lookback_study.py](../transformer_lookback_study.py)

### Additional project files
- [../main.py](../main.py)
- [../dashboard.py](../dashboard.py)
- [../multimodal_dataset.py](../multimodal_dataset.py)
- [../parameter_matched_ablation.py](../parameter_matched_ablation.py)
- [../parameter_matched_ablation_v3.py](../parameter_matched_ablation_v3.py)
- [../parameter_matched_ablation_v4.py](../parameter_matched_ablation_v4.py)
- [../residual_ablation_experiment.py](../residual_ablation_experiment.py)

---

## 6. Data assets

### Raw and processed data
- [../data](../data)
- [../data_shanghai_converted](../data_shanghai_converted)

Data categories include:
- OhioT1DM training and testing datasets
- multimodal CSVs with CGM, insulin, meal, sleep, and physiological inputs
- XML sources for patient signal reconstruction
- Shanghai conversion artifacts and summary data

### Patient-level audit files
- [../acceleration_audit_2020.csv](../acceleration_audit_2020.csv)
- [../phase2_2018_multimodal_audit.csv](../phase2_2018_multimodal_audit.csv)
- [../phase2_2018_sequence_audit.csv](../phase2_2018_sequence_audit.csv)
- [../phase2_2020_audit.csv](../phase2_2020_audit.csv)
- [../phase2_sequence_counts.csv](../phase2_sequence_counts.csv)

These files support dataset-quality checks and signal audit tracking.

---

## 7. Output folders and benchmark artifacts

The output directory contains the main experimental results, trained model checkpoints, and patient-wise summaries.

### Major output folders
- [../output/phase2_baselines](../output/phase2_baselines)
- [../output/phase2_capacity_sweep](../output/phase2_capacity_sweep)
- [../output/phase2_data_audit](../output/phase2_data_audit)
- [../output/phase2_feature_ablation](../output/phase2_feature_ablation)
- [../output/phase2_frozen_v4_vs_gru](../output/phase2_frozen_v4_vs_gru)
- [../output/phase2_gap_safe_architecture](../output/phase2_gap_safe_architecture)
- [../output/phase2_hybrid](../output/phase2_hybrid)
- [../output/phase2_hybrid_ablations](../output/phase2_hybrid_ablations)
- [../output/phase2_hybrid_all12](../output/phase2_hybrid_all12)
- [../output/phase2_hybrid_all12_run2](../output/phase2_hybrid_all12_run2)
- [../output/phase2_matched_gru_gap_safe_validation](../output/phase2_matched_gru_gap_safe_validation)
- [../output/phase2_parameter_matched](../output/phase2_parameter_matched)
- [../output/phase2_residual_ablation](../output/phase2_residual_ablation)
- [../output/phase2_single_horizon_ablation](../output/phase2_single_horizon_ablation)
- [../output/phase2_statistics](../output/phase2_statistics)
- [../output/phase2_transformer](../output/phase2_transformer)
- [../output/phase2_transformer_lookback_statistics](../output/phase2_transformer_lookback_statistics)
- [../output/phase2_transformer_lookback_study](../output/phase2_transformer_lookback_study)
- [../output/phase2_transformer_mgdl](../output/phase2_transformer_mgdl)

### Key result files
- [../output/phase2_gap_safe_architecture/architecture_mean_sd.csv](../output/phase2_gap_safe_architecture/architecture_mean_sd.csv)
- [../output/phase2_gap_safe_architecture/all_results.csv](../output/phase2_gap_safe_architecture/all_results.csv)
- [../output/phase2_hybrid_all12/all12_summary.csv](../output/phase2_hybrid_all12/all12_summary.csv)
- [../output/phase2_hybrid_all12_run2/all12_summary.csv](../output/phase2_hybrid_all12_run2/all12_summary.csv)
- [../output/phase2_hybrid_ablations/patient_559_all_ablations.csv](../output/phase2_hybrid_ablations/patient_559_all_ablations.csv)
- [../output/phase2_data_audit/phase2_data_quality_summary.csv](../output/phase2_data_audit/phase2_data_quality_summary.csv)
- [../output/phase2_frozen_v4_vs_gru/patient_mean_sd.csv](../output/phase2_frozen_v4_vs_gru/patient_mean_sd.csv)

### Model checkpoint artifacts
These directories include trained model weights and training histories for patient-specific and cohort-level experiments.

Examples:
- [../output/phase2_hybrid_all12_run2/ohio2018/559/best_model.pt](../output/phase2_hybrid_all12_run2/ohio2018/559/best_model.pt)
- [../output/phase2_hybrid_all12_run2/ohio2020/540/best_model.pt](../output/phase2_hybrid_all12_run2/ohio2020/540/best_model.pt)
- [../output/phase2_hybrid/patient_559_model.pt](../output/phase2_hybrid/patient_559_model.pt)

---

## 8. Reports and visual outputs

The project includes final summary and visual report materials:

- [../reports/Architecture comparison chart.png](../reports/Architecture comparison chart.png)
- [../reports/Per patient rmse .png](../reports/Per%20patient%20rmse%20.png)
- [../reports/architecture_patient_wise.csv](../reports/architecture_patient_wise.csv)
- [../reports/featureset_patient_wise.csv](../reports/featureset_patient_wise.csv)
- [../reports/gru_patient_wise.csv](../reports/gru_patient_wise.csv)
- [../reports/CGM_Forecasting_Combined_Report.md](../reports/CGM_Forecasting_Combined_Report.md)
- [../reports/CGM_Forecasting_Combined_Report.pdf](../reports/CGM_Forecasting_Combined_Report.pdf)
- [../reports/CGM_Forecasting_Combined_Report.docx](../reports/CGM_Forecasting_Combined_Report.docx)

The repository also has a summary document in the root project summary and presentation-style materials in the reports directory.

---

## 9. Established results

The following conclusions are supported by the current code and generated outputs.

### 9.1 Data integrity and leakage controls
- The cohort-specific multimodal inputs are intentionally gated by data availability: 2018 uses heart-rate/steps features, while 2020 uses acceleration-derived features; the pipeline explicitly avoids using unavailable signals in a cohort where they do not exist. See [../phase2_multimodal_preprocessing.py](../phase2_multimodal_preprocessing.py) and [../ohio2020_acceleration_preprocessing.py](../ohio2020_acceleration_preprocessing.py).
- The acceleration preprocessing is strictly causal: it aggregates only acceleration observations before each CGM timestamp and does not use a future-looking window.
- Sequence construction uses a gap-safe historical lookback with future target offsets and rejects sequences that cross missing gaps or invalid temporal boundaries; this prevents future leakage from entering the input window. See [../phase2_sequence_utils.py](../phase2_sequence_utils.py).

### 9.2 Controlled architecture comparisons
- The architecture benchmark script uses the same patient sets, same lookback, same horizons, same feature set, and the same evaluation pipeline across the compared architectures. See [../phase2_gap_safe_architecture_ablation.py](../phase2_gap_safe_architecture_ablation.py).
- The generated output file [../output/phase2_gap_safe_architecture/all_results.csv](../output/phase2_gap_safe_architecture/all_results.csv) contains 600 patient-level architecture comparisons across 10 architectures, 12 patients, and 5 horizons.
- Paired patient-wise Wilcoxon tests on RMSE from the generated output show some short-horizon significance:
  - GRU vs TCN at 15 min: p = 0.0425
  - GRU vs Transformer at 15 min: p = 0.0024
  - GRU vs Transformer at 30 min: p = 0.0068
  - GRU vs gru_transformer at 30 min: p = 0.0210
  - GRU vs tcn_gru_transformer at 15 min: p = 0.0010
  - GRU vs tcn_gru_transformer at 30 min: p = 0.0015
- These results support a horizon-dependent interpretation rather than a universal dominance claim.

### 9.3 Verified all-12 hybrid summary
- The all-12 hybrid summary file [../output/phase2_hybrid_all12/all12_summary.csv](../output/phase2_hybrid_all12/all12_summary.csv) contains patient-wise mean RMSE summaries across the full 12-patient cohort.
- The current generated summary indicates the following robust trend:
  - 15 min: RMSE mean = 21.87
  - 30 min: RMSE mean = 28.61
  - 60 min: RMSE mean = 39.72
  - 90 min: RMSE mean = 48.03
  - 120 min: RMSE mean = 53.41
- This is a strong evidence-backed summary of the final hybrid pipeline, but it is not a proof that every component of the model contributes independently in a statistically significant way.

---

## 10. New experimental findings

The following are supported as exploratory or emerging findings, but remain more limited than a final proof claim.

- The hybrid architecture appears competitive in the all-12 assessment and is strongest at short-to-mid horizons, while longer-horizon differences are less pronounced.
- The single-patient hybrid ablation snapshot [../output/phase2_hybrid_ablations/patient_559_all_ablations.csv](../output/phase2_hybrid_ablations/patient_559_all_ablations.csv) shows that the architecture search is operational and that branch combinations are being tested under a consistent data pipeline.
- The hybrid outputs suggest a plausible performance advantage over some baselines in selected horizons, but the current repository evidence does not yet support a definitive all-patient claim that the complete branch set is strictly superior to all alternatives.
- The stronger statistical statement remains open pending an all-patient paired ablation analysis.

---

## 11. Hypotheses about novelty

The following statements should be treated as hypotheses rather than established conclusions.

- The most defensible novelty claim is the integration of gap-safe multimodal preprocessing with a hybrid TCN–GRU–Transformer forecasting architecture under a patient-wise evaluation workflow.
- A stronger novelty claim — that each branch contributes a statistically measurable improvement beyond the others — is not yet established by the available repository outputs.
- The data support a plausible research narrative that the hybrid model is promising and especially relevant for short-horizon forecasting, but the component-wise contribution, branch necessity, and universal superiority remain to be demonstrated with full paired ablation statistics across all patients.

---

## 12. Final interpretation

The project is best described as a complete and reproducible research implementation with validated architecture benchmarking and a strong hybrid-model hypothesis, rather than a final definitively proven winner-takes-all architecture result.

The strongest defensible conclusion is that the repository contains a mature forecasting workflow, a coherent benchmark pipeline, and evidence-backed short-horizon architecture differences, while the final claim that the hybrid model definitively outperforms every alternative or that each component contributes independently remains an open but promising research question.
