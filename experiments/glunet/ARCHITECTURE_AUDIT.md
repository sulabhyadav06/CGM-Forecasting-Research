# GluNet Architecture Audit

## Reference
Li et al., "GluNet: A Deep Learning Framework for Accurate
Glucose Forecasting," IEEE Journal of Biomedical and Health
Informatics, 2020.

## Paper-specified requirements

- Input window: 16 time steps.
- Prediction horizons: 30 and 60 minutes.
- Inputs: glucose, meal, insulin, and timestamps.
- Five convolutional layers.
- First three layers: 32 hidden units.
- Final two layers: 64 hidden units.
- Dilation rates stated in the paper: 1, 2, 4, 8.
- Gated activations using tanh and sigmoid.
- Residual and parameterized skip connections.
- Output: 256 glucose-change classes.

## Existing baseline

File: glunet_model_baseline.py

- Four gated residual blocks.
- 32 channels throughout.
- Dilation rates: 1, 2, 4, 8.
- Three input features by default.
- 256 output classes.

This is a GluNet-inspired baseline, not an exact reproduction.

## Details still requiring verification

1. How the five-layer description maps to the four listed dilation rates.
2. Exact channel transitions and residual projections.
3. Exact skip-connection aggregation.
4. Timestamp representation.
5. Dataset-specific preprocessing and missing-data treatment.
6. Optimizer, learning rate, and other training settings.
7. Conversion of predicted class probabilities to point forecasts.

## Experimental policy

- Preserve the baseline.
- Keep the CRNN pipeline unchanged.
- Do not describe an adapted architecture as an exact reproduction.
- Validate architecture and data before full training.
- Report all deviations from the paper.
