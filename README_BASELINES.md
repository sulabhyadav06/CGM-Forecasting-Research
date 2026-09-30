# Phase-2 Baseline Comparison

This script provides matched GRU, LSTM and TCN baselines for the Phase-2 multimodal
OhioT1DM data.

The preprocessing is intentionally aligned with the corrected Transformer pilot:
- train-only median imputation
- train-only standardization
- train-only glucose target standardization
- continuous 5-minute sequence construction
- chronological validation split
- purge = lookback_steps + horizon_steps
- identical optimizer/early-stopping defaults

For the first controlled Patient 559 experiment:

```bash
python baseline_experiments.py \
  --data-root data/phase2/ohio2018 \
  --patient 559 \
  --lookbacks 60 \
  --horizons 15 30 60 90 120 \
  --epochs 15
```

The script writes CSV results under:
`output/phase2_baselines/`

Note: the initial baseline comparison uses hidden=64 and 2 layers for GRU/LSTM,
and two 64-channel TCN blocks. Parameter counts are reported. This is a controlled
baseline, not yet a parameter-matched comparison. Parameter matching can be performed
after the first baseline table is established.
