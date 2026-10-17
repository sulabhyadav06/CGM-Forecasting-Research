# CRNN Reproduction: Implementation Notes

## Reference
Li et al., "Convolutional Recurrent Neural Networks for Glucose
Prediction," IEEE Journal of Biomedical and Health Informatics, 2020.

## Modified LSTM
The reference describes:
    h_t = o_t * tanh(c_t)
    y_t = h_t + x_t

The current implementation keeps h_t as the internal recurrent
state and uses y_t as the output passed to the next layer.

## Residual dimension adaptation
The CNN produces 32 features per time step, while the LSTM hidden
state has 64 features. The current implementation uses a learned,
bias-free linear projection from 32 to 64 dimensions before the
residual addition.

This projection is an implementation adaptation. The paper does
not explicitly specify this projection or explain how the residual
dimensions are reconciled.

## Parameter-count discrepancy
The current modified LSTM has 27,136 trainable parameters:
- Standard LSTM parameters: 25,088
- Residual projection: 2,048

The paper's Table V reports 24,832 LSTM parameters. The reason for
the 256-parameter discrepancy has not been established.

## Reproduction status
The architecture has passed basic tensor-shape and forward-pass
checks. Exact architectural fidelity has not yet been established.
