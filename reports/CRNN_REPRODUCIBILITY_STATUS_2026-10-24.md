# CRNN Reproducibility Status
Date: 2026-10-24

## 1. Scope
CRNN glucose forecasting on the 12 OhioT1DM patients.
Forecast horizons: 30 and 60 minutes.

## 2. Verified prediction metrics
All 24 patient-horizon prediction files were checked against
output/crnn_initial/plots/crnn_metrics_recomputed_from_predictions.csv.
The audit reported 24/24 files checked and zero mismatches for sample
counts, MAE, RMSE, and MARD within an absolute tolerance of 1e-5.

| Metric | 30 minutes | 60 minutes |
|---|---:|---:|
| Mean patient MAE (mg/dL) | 13.4944 | 23.6365 |
| SD of patient MAE (mg/dL) | 1.4801 | 2.7142 |
| Mean patient RMSE (mg/dL) | 19.3091 | 32.5765 |
| SD of patient RMSE (mg/dL) | 2.2028 | 3.7182 |
| Mean patient MARD (%) | 9.0727 | 15.7825 |
| SD of patient MARD (%) | 1.5911 | 2.7551 |

These are results of the current implementation, not proof of exact
reproduction of the published paper.

## 3. Current implementation
- Input lookback: 24 samples at nominal 5-minute intervals.
- Features: glucose, carbohydrates over the preceding 60 minutes,
  and insulin bolus over the preceding 60 minutes.
- Target: future glucose minus current glucose.
- Forecast horizons: 30 and 60 minutes.
- CNN channels: 8, 16, and 32.
- Convolution kernel size: 4; padding="same".
- Three max-pooling layers reduce the sequence length from 24 to 3.
- Modified LSTM: input size 32, hidden size 64.
- Dense dimensions: 64 -> 256 -> 32 -> 1.
- Loss: MAE.
- Optimizer: RMSprop, learning rate 0.001.
- Default maximum epochs: 50; early-stopping patience: 7.
- Validation fraction: latest 15% of the training timeline.
- Batch size: 256; dropout: 0.2; seed: 42.
- Existing test partition is evaluated after model selection.

## 4. Paper-fidelity issues still unresolved
1. The paper's Table V reports 24,832 LSTM parameters. The current
   modified LSTM has 25,088 standard gate parameters plus a 2,048-
   parameter residual projection, totaling 27,136.
2. The paper's parameter count is consistent with one 256-element
   gate-bias vector; the current implementation has two. The precise
   original implementation convention must be verified before changes.
3. The paper describes y_t = h_t + x_t. The current model projects the
   32-dimensional input to 64 dimensions before residual addition.
   This projection is an adaptation, not a verified paper detail.
4. Exact activation, convolution-padding, preprocessing, filtering,
   and optimization details must be checked against the full paper.
5. Current gap-safe sequence construction and exact target-timestamp
   matching are implementation choices; equivalence to the paper's
   original data pipeline has not been established.
6. The saved checkpoints record patient, horizon, feature count,
   dropout, best epoch, validation MAE, and seed. No separate config
   JSON was found at the searched output depth.

## 5. Reproduction status
Status: numerically audited baseline; exact paper reproduction not yet
established.

Preserve output/crnn_initial and all current checkpoints. Any paper-
aligned implementation should be developed and evaluated separately
so the existing baseline remains reproducible.

## 6. Reference
Li K, Daniels J, Liu C, Herrero P, Georgiou P.
Convolutional Recurrent Neural Networks for Glucose Prediction.
IEEE Journal of Biomedical and Health Informatics, 2020.
DOI: 10.1109/JBHI.2019.2908488
Open manuscript: https://arxiv.org/abs/1807.03043
