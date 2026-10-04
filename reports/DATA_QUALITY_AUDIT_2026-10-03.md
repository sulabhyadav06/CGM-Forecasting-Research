# Corrected Data Quality Audit — 3 October 2026

The corrected local Phase-2 audit checked all 24 OhioT1DM train/test CSVs (12 participants × two splits).

- Missing files: 0.
- Duplicate timestamps: 0.
- Missing glucose targets in stored rows: 0.
- Timestamp gaps over 30 minutes: present in all 24 files, so they must be treated as real discontinuities. Gap-safe sequence construction rejects windows crossing these gaps; the preprocessing does not interpolate glucose targets.
- Constant `carbs_observed_60min` and `bolus_observed_60min` columns were removed because they were always 1 and provided no information. They are not model inputs.
- Wearable coverage varies by participant; the 2020 acceleration stream is materially incomplete for some participants. Retain missingness/coverage features and report patient-wise results.
- `is_sleeping` is not available in the corrected Phase-2 CSVs, so a sleep ablation cannot be reported from these files.

The audit is a data-integrity check, not proof against all model-level leakage. Train-only imputation/scaling, chronological validation, purged training targets, exact target alignment, and patient-level review remain necessary. The Clarke Error Grid implementation needs independent expert review before publication.
