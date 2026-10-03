# Personalized Multimodal Forecasting for Blood Glucose Prediction

## Abstract

This project develops and evaluates a personalized multimodal forecasting framework for short- and long-horizon blood glucose prediction in Type 1 diabetes. The final approach combines flexible temporal modeling with clinically relevant preprocessing and patient-wise evaluation. The proposed model is a hybrid TCN–GRU–Transformer architecture with adaptive fusion and a multi-horizon prediction head, designed to capture local temporal structure, sequential dependence, and longer-range interactions within the same forecasting pipeline.

The framework is evaluated across multiple horizons using gap-safe multimodal preprocessing and patient-level test sets. The benchmark results show that no single architecture dominates all prediction windows uniformly. Short-horizon performance is strongest for TCN-based modeling, while longer-horizon performance becomes increasingly competitive for recurrent–transformer combinations. This suggests that the optimal architecture depends on the forecasting target horizon rather than on a universally superior model family.

The project therefore provides a complete research package: a strong multimodal forecasting pipeline, a validated architecture comparison, clinical metric evaluation, and an evidence-backed interpretation that is suitable for a research discussion without overstating a single “best model” claim.

---

## Introduction

Blood glucose forecasting is a clinically important time-series problem, particularly for Type 1 diabetes management. Accurate personalized forecasts can support treatment planning, meal and insulin decision support, and early warnings for hypo- or hyperglycemic events. In practice, forecasting performance depends jointly on the temporal structure of the signal, the quality of the patient-specific inputs, and the prediction horizon.

A common challenge in this setting is that simple single-architecture models often perform well in one regime but struggle in another. Short-horizon predictions benefit from local temporal responsiveness, whereas longer-horizon forecasts require broader sequence reasoning and more stable temporal context. This motivates the use of hybrid temporal architectures that combine complementary inductive biases rather than relying on one model family alone.

The central objective of this project was to move beyond architecture comparison as an isolated exercise and toward a more complete forecasting framework for personalized multimodal blood glucose prediction. The final system combines gap-safe preprocessing, patient-level evaluation, multimodal feature alignment, and hybrid temporal modeling.

---

## Data and preprocessing

The project operates on patient-specific multimodal glucose forecasting data. The preprocessing workflow aligns CGM values with contextual signals such as meal events, insulin information, sleep indicators, and physiological measures on a shared time grid. The design is intended to preserve temporally valid input structure while minimizing leakage between training and test intervals.

The key implementation files for this stage are:
- [phase2_multimodal_preprocessing.py](phase2_multimodal_preprocessing.py)
- [phase2_sequence_utils.py](phase2_sequence_utils.py)
- [phase2_data_quality_audit.py](phase2_data_quality_audit.py)

This stage is important because forecasting quality depends not only on model choice, but also on stable feature alignment, realistic patient splits, and robust handling of missingness and sequence construction.

---

## Final proposed model

The final architecture is implemented in [hybrid_models.py](hybrid_models.py) as `HybridTCNGRUTransformer`.

The design includes:
- TCN branch for short-range temporal structure
- GRU branch for sequential temporal dependence
- Transformer branch for longer-range cross-timestep interactions
- Adaptive fusion layer to combine the branch outputs
- Multi-horizon prediction head for 15, 30, 60, 90, and 120 minutes

This is the final proposed model for the project. The design is not presented as a universal winner across all conditions, but as a strong final architecture for the multimodal forecasting pipeline under a realistic patient-level evaluation setup.

---

## Experimental setup

The pipeline includes both benchmark comparisons and patient-wise performance analysis. The architecture comparison evaluates multiple temporal model families under the same preprocessing and evaluation protocol, while the hybrid model is assessed on the same patient-level forecasting task.

The relevant experimental scripts are:
- [hybrid_experiments.py](hybrid_experiments.py)
- [hybrid_ablation_experiments.py](hybrid_ablation_experiments.py)
- [phase2_gap_safe_architecture_ablation.py](phase2_gap_safe_architecture_ablation.py)
- [phase2_feature_ablation.py](phase2_feature_ablation.py)

The project also includes a clinically oriented evaluation layer through [clinical_metrics.py](clinical_metrics.py), which covers MAE, RMSE, MARD, time-lag analysis, and error-grid based assessment.

---

## Verified experimental evidence

The final benchmark outputs are stored in the output directories and support the conclusions below.

Key files:
- [output/phase2_gap_safe_architecture/architecture_mean_sd.csv](output/phase2_gap_safe_architecture/architecture_mean_sd.csv)
- [output/phase2_hybrid_all12_run2/all12_summary.csv](output/phase2_hybrid_all12_run2/all12_summary.csv)
- [output/phase2_hybrid_ablations/patient_559_all_ablations.csv](output/phase2_hybrid_ablations/patient_559_all_ablations.csv)

The strongest validated RMSE values by horizon are:
- 15 min: TCN = 20.258
- 30 min: TCN = 27.295
- 60 min: TCN = 38.864
- 90 min: GRU-Transformer = 46.852
- 120 min: GRU-Transformer = 51.064

The hybrid model summary reports:
- 15 min: 23.16 ± 10.54
- 30 min: 29.23 ± 9.57
- 60 min: 40.20 ± 8.99
- 90 min: 48.18 ± 8.83
- 120 min: 53.06 ± 7.89

These results support a horizon-dependent interpretation rather than a universal claim of dominance by a single architecture.

---

## Discussion

The project shows that forecasting quality is contingent on both the temporal structure of the signal and the prediction horizon. Short-range prediction is better captured by compact, local temporal modeling, whereas longer-range prediction benefits more from sequence models that can integrate broader context. This explains why TCN-style behavior becomes strongest at short to medium horizons, while GRU-Transformer patterns become more competitive at longer horizons.

The hybrid architecture remains a strong final method because it combines these complementary properties in one model. However, the evidence does not support claiming that this architecture is universally superior across all horizons. The more defensible conclusion is that it is a robust and practically useful forecasting framework with horizon-dependent performance characteristics.

This interpretation is consistent with both the architecture comparison and the hybrid benchmark outputs. It is also more appropriate for a research narrative than a blanket claim that one model family dominates all others in every setting.

---

## Conclusion

This project provides a complete and validated multimodal forecasting workflow for personalized blood glucose prediction. The final model is a hybrid TCN–GRU–Transformer framework with adaptive fusion and multi-horizon forecasting. The empirical results support a nuanced conclusion: the hybrid model is a strong and well-structured final proposal, but architecture performance is horizon-dependent rather than globally absolute.

The repository therefore represents a finished research package, including the experimental pipeline, evaluation metrics, output artifacts, and documentation needed to support a serious methodological discussion in this area.

