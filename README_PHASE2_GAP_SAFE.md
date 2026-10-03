# Phase-2 gap-safe sequence construction

## Files

- `phase2_sequence_utils.py` — reusable sequence builder.
- `matched_gru_joint_validation.py` — corrected matched-GRU validation runner.

## What changed

The sequence builder preserves the original timestamp grid before checking values.
A sample is rejected if its lookback, prediction point, or any future target
crosses a timestamp discontinuity.

This is especially important because the OhioT1DM Phase-2 CSVs contain
multi-hour and multi-day gaps.

The corrected matched-GRU runner uses:

- Ohio 2018 patients 559, 563, 570, 575, 588, 591
- 120-minute lookback
- 15/30/60/90/120-minute horizons
- chronological 85/15 train/validation split
- patient-specific train-only scaling
- seed 42
- AdamW, lr=3e-4, weight decay=1e-4
- batch size 128
- SmoothL1 loss
- early stopping
- hidden size 152
- joint six-patient training
- validation-only model selection
- no held-out test evaluation

The output is written to:

`output/phase2_matched_gru_gap_safe_validation/`

This run is intended to establish the corrected baseline protocol before
implementing the final PMTFN.


## v2 correction: missing feature values

The Phase-2 wearable features can contain missing values. The first gap-safe
runner intentionally rejected them, which stopped before training.

The updated runner performs **train-only median imputation**:

1. compute each feature median from that patient's chronological training
   partition only;
2. replace missing feature values in training with those medians;
3. apply the same training medians to validation;
4. scale after imputation;
5. retain the original timestamps for gap/sequence validation.

Glucose targets are not imputed. If glucose is missing, the run stops.

The update also casts source feature columns to floating point before scaled
assignment, removing the pandas incompatible-dtype FutureWarnings.
