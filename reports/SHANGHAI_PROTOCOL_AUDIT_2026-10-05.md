# Shanghai T1DM External-Dataset Protocol Audit

- 12 T1DM patients: 1001–1012
- Horizons: 15, 30, 60 minutes
- Architectures: LSTM, GRU, BiLSTM, TCN, Transformer
- Sampling: 15 minutes
- Lookback: 60 minutes
- Input: glucose-only
- Split: chronological 80/20
- Training normalization: training-set mean/std
- Windows: constructed separately within train/test
- Internal validation: Keras validation_split=0.1

Important limitation:
This Shanghai experiment does not use the full corrected OhioT1DM Phase-2 protocol and should therefore be presented as an external-dataset architecture comparison, not as a directly equivalent replication of the OhioT1DM multimodal hybrid experiment.
