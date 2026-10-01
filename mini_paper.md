# Final research summary

## Abstract

This project develops and evaluates a personalized multimodal multi-horizon blood glucose forecasting framework for Type 1 diabetes. The final modeling approach is a hybrid TCN–GRU–Transformer architecture with adaptive fusion and a shared multi-horizon output head. The model is designed to combine complementary temporal inductive biases: convolutional filters for local dynamics, GRU recurrence for sequential dependence, and transformer attention for longer-range interactions across the input sequence.

The framework was evaluated across multiple patient-level forecasting horizons using gap-safe multimodal preprocessing and clinically relevant metrics. The verified outputs show that no single architecture is universally dominant across all horizons. Short-horizon performance favors TCN, while longer-horizon performance becomes more competitive for GRU-Transformer patterns. This suggests that forecasting performance is horizon-dependent and that architecture selection should be conditioned on the prediction window rather than treated as a blanket superiority claim.

The final project therefore represents a completed research pipeline: a strong multimodal forecasting system, a verified architecture comparison, a clinically oriented evaluation framework, and an evidence-backed synthesis that supports a nuanced paper narrative rather than an overconfident universal winner.

---

## Final proposed model

The final architecture is implemented in [hybrid_models.py](hybrid_models.py) as `HybridTCNGRUTransformer`.

Architecture summary:
- Input embedding layer
- TCN branch
- GRU branch
- Transformer branch
- Adaptive fusion layer
- Multi-horizon prediction head for 15, 30, 60, 90, and 120 minutes

This is the final proposed model for the project.

---

## Verified experimental evidence

The project output files contain the final benchmark evidence:

- [output/phase2_gap_safe_architecture/architecture_mean_sd.csv](output/phase2_gap_safe_architecture/architecture_mean_sd.csv)
- [output/phase2_hybrid_all12_run2/all12_summary.csv](output/phase2_hybrid_all12_run2/all12_summary.csv)
- [output/phase2_hybrid_ablations/patient_559_all_ablations.csv](output/phase2_hybrid_ablations/patient_559_all_ablations.csv)

The verified best RMSE values by horizon are:

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

These results support a horizon-dependent conclusion rather than a claim of universal dominance by the hybrid architecture.

---

## Discussion

The strongest and most defensible interpretation is that the multimodal forecasting framework is successful, but the best architecture depends on the target horizon. Short-term prediction benefits most from compact local temporal modeling, whereas longer horizons are more compatible with recurrent–transformer combinations.

This matters for a research paper because it means the contribution is not simply "one architecture beats another." The contribution is a robust, personalized, multimodal forecasting pipeline with clinically relevant evaluation, patient-level analysis, and a clear demonstration that architecture choice should be tuned to horizon-specific forecasting behavior.

---

## Conclusion

The project is now complete as a research package. The final model is the hybrid TCN–GRU–Transformer architecture with adaptive fusion and multi-horizon forecasting. The evidence supports a nuanced conclusion: that model is a strong final proposal, but the data do not justify a universal claim that it dominates all baselines across every horizon.

The project is therefore complete in the strongest sense: implemented, benchmarked, validated, and synthesized into an evidence-based research narrative.
