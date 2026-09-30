# Phase-2 Improved Transformer

## Purpose

This module is the Transformer stage of the Phase-2 CGM forecasting experiments.

The intended workflow is:

1. Freeze/audit the multimodal 5-minute preprocessing.
2. Establish an improved Transformer.
3. Compare lookback and Transformer configuration.
4. Use the selected Transformer as one branch of the later hybrid.
5. Only then implement TCN-GRU-Transformer adaptive fusion.

## Model

```text
Multimodal 5-min inputs
        |
        v
Feature Projection
        |
        v
Sinusoidal Positional Encoding
        |
        v
Transformer Encoder
  - multi-head self-attention
  - GELU feed-forward
  - pre-normalization
        |
        v
Pooling
  - last token
  - mean pooling
  - attention pooling
        |
        v
Prediction Head
        |
        v
Future glucose
```

## Initial pilot

Do NOT immediately run every possible combination.

Start with one patient and a small configuration sweep:

- lookback: 60, 120, 180, 240 minutes
- horizon: 30 or 60 minutes
- d_model: 64, 128
- heads: 2, 4
- encoder layers: 2, 3
- pooling: last, attention

After identifying a reasonable configuration, run the complete 12-patient experiment across:

- 15
- 30
- 60
- 90
- 120 minutes

## Important protocol

- 5-minute sampling.
- Patient-wise evaluation.
- Training scaler only; never fit scaler on test data.
- No sequences across timestamp gaps.
- Test set is not used for configuration selection.
- Fixed random seed.
- Report patient-level results, then mean ± SD.
- Keep model parameter counts for parameter-matched comparisons.

## Later hybrid

The Transformer produced here is not the final proposed model.

Later:

```text
TCN branch  -> local patterns
GRU branch  -> sequential dynamics
Transformer -> longer-range dependencies
                    |
                    v
          Adaptive/Gated Fusion
                    |
                    v
       Multi-Horizon Prediction
       15/30/60/90/120 minutes
```

The branches should have distinct roles rather than simply concatenating identical representations.
