# CGM Forecasting Project Report

## 1. Project status

This repository contains the completed implementation and validation workflow for personalized multimodal multi-horizon blood glucose forecasting in Type 1 diabetes. The project combines data preprocessing, patient-level evaluation, architecture comparison, hybrid modeling, clinical metrics, and result synthesis.

The notebook state has been restored to the last committed project version. The current-stage continuation notebook was removed so the project remains aligned with the previous commit snapshot.

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

## 9. Key findings

The final evidence supports a horizon-dependent conclusion rather than a universal superiority claim.

Validated summary:
- 15 min: TCN best RMSE = 20.258
- 30 min: TCN best RMSE = 27.295
- 60 min: TCN best RMSE = 38.864
- 90 min: GRU-Transformer best RMSE = 46.852
- 120 min: GRU-Transformer best RMSE = 51.064

The hybrid all-12 summary remains strong across the full forecast window, but the data do not support claiming that one architecture dominates every horizon uniformly.

---

## 10. Final interpretation

The project is best summarized as a completed research implementation in personalized multimodal blood glucose forecasting. The final proposed model is the hybrid TCN–GRU–Transformer architecture, and the final evidence supports a nuanced scientific interpretation: architecture quality depends on the forecast horizon and should be evaluated accordingly.

This is the most defensible project position supported by the outputs in the repository.
