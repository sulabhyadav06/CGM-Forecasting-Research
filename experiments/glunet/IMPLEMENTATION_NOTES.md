# GluNet Reproduction: Implementation Notes

## Reference
K. Li et al., "GluNet: A Deep Learning Framework for Accurate
Glucose Forecasting," IEEE Journal of Biomedical and Health
Informatics, 2020.

## Paper-specified design

- Input factors: glucose, meal, insulin, and timestamps.
- Sliding window: 16 time steps.
- Forecast horizons: 30 and 60 minutes.
- Five convolutional layers.
- First three hidden layers: 32 units each.
- Final two hidden layers: 64 units each.
- Dilations: 1, 2, 4, 8.
- Causal dilated convolutions.
- Gated activations.
- Residual and parameterized skip connections.
- Output: softmax over 256 glucose-change categories.
- Glucose change: future glucose minus current glucose.
- Quantization interval: 0.1 mg/dL.
- Reconstruct absolute glucose from the predicted glucose change.

## Dataset adaptation

This reproduction uses the existing OhioT1DM predefined training
and testing partitions and targets all 12 available subjects.

The paper's OhioT1DM evaluation used six subjects. Results on all
12 subjects are an extension, not an exact reproduction of the
original experimental cohort.

The current CRNN data loader is not modified by this implementation.

## Details requiring verification

- Exact convolution padding and channel dimensions.
- Exact residual and skip-connection implementation.
- Exact preprocessing thresholds for OhioT1DM.
- Exact timestamp encoding.
- Optimizer and all numerical training hyperparameters.
- Conversion from the 256-class probability distribution to a
  continuous glucose forecast.
- Treatment of missing or irregular timestamps.

Do not describe unspecified details as exact matches to the paper.
