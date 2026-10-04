# Reproducible Phase-2 Runbook (2026-10-03)

This runbook accompanies `reports/FINAL_PROJECT_STATUS_2026-10-03.md`. Raw OhioT1DM data must stay outside GitHub under the applicable data-use agreement.

## Protocol

- 12 patients: 2018 IDs 559, 563, 570, 575, 588, 591; 2020 IDs 540, 544, 552, 567, 584, 596.
- Five horizons: 15/30/60/90/120 minutes.
- Cohort features: 2018 CGM + carbs + bolus/basal insulin + heart rate + steps; 2020 CGM + carbs + bolus/basal insulin + acceleration.
- Compare LSTM, GRU, BiLSTM, TCN, Transformer, TCN-GRU, GRU-Transformer, TCN-Transformer, TCN-GRU-Transformer, adaptive TCN-GRU-Transformer.
- Use train-only preprocessing, chronological validation, a horizon purge, gap-safe windows, persistence baseline, patient-level clinical metrics and paired patient-level statistical tests with Holm correction.
- Treat corrected 2020 bolus results as authoritative. Earlier zero-bolus runs are superseded.

## Local commands

The downloadable companion research package includes `final_clinical_architecture_evaluation.py`, `final_common_feature_analysis.py`, `final_statistical_analysis.py`, and the associated tests/configuration. The GitHub repository currently tracks the core model/preprocessing modules, runbook, tests and summary reports; not every generated experiment runner or patient-level output is committed to GitHub. Keep restricted raw data outside version control.

```bash
python -m pip install -r requirements-research.txt
python -m py_compile *.py
python -m pytest -q tests/test_research_protocol.py test_time_lag_metric.py
python phase2_data_quality_audit.py --data-root data/phase2 --output-root output/final_corrected_data_audit
python final_clinical_architecture_evaluation.py --data-root data/phase2 --out output/final_corrected_architecture_lb120 --epochs 15 --patience 3 --lookback 120
python final_common_feature_analysis.py
python final_statistical_analysis.py --input output/final_corrected_architecture_lb120/patient_horizon_clinical.csv --features-input output/final_common_feature/common_feature_patient_horizon.csv --out output/final_statistics
python patient_difficulty_analysis.py
```

Run lookback studies separately at 60/120/180/240 minutes; report the horizon and cohort clearly. The current focused lookback runner is a Transformer experiment on OhioT1DM 2018, not a substitute for the full cross-cohort architecture experiment.

## Reporting rules

Publish per-patient metrics and mean ± SD. Compare model pairs on matched patient values, adjust multiple comparisons, and include effect sizes. Report signed lag (positive means prediction trails truth) and absolute lag magnitude separately. Check the lag metric on synthetic delayed signals. The Clarke Error Grid implementation needs independent expert review. Short/single-seed runs are exploratory and cannot establish model superiority, novelty, clinical safety or external validity.
