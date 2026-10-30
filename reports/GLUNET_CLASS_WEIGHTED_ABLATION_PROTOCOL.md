# GluNet Class-Weighted Loss Ablation

## Purpose

Test whether conservative class reweighting improves intermediate glucose-change classification for the explicitly labelled GluNet candidate. This is an ablation, not a verified reproduction of the published GluNet method.

## Experimental comparison

- Baseline: `--loss-mode standard` (ordinary cross-entropy).
- Ablation: `--loss-mode sqrt_inverse_frequency` (square-root inverse-frequency weighted cross-entropy).

The class weights are calculated from the training labels for each patient/horizon only: `w_c = n_c^-0.5`, normalized so the sample-weighted mean training weight is 1. Classes absent from the training segment receive weight 0. Validation and test labels are never used to calculate weights.

The training objective is weighted only in the ablation. Validation loss remains ordinary, unweighted cross-entropy for both modes, so early stopping/checkpoint selection uses the same validation objective. The test set is used only for final evaluation.

## Controlled run

Start with patient 540 and 30 minutes, keeping the same seed, batch size, learning rate, validation fraction, epochs, patience, architecture, features, preprocessing, and split as the baseline run. Use a new output directory so no prior artifact is overwritten.

```bash
python -m experiments.glunet.train_glunet_candidate_all_patients \
  --data-root "$HOME/Desktop/OhioT1DM_data_backup" \
  --output-dir "$HOME/Desktop/CGM-reproduction-test/output/glunet_weighted_ablation" \
  --patients 540 \
  --horizons 30 \
  --epochs 20 \
  --patience 5 \
  --batch-size 128 \
  --seed 42 \
  --loss-mode sqrt_inverse_frequency
```

The baseline run used ordinary cross-entropy. Record the exact baseline run directory and compare both runs on the same test windows.

## Required comparisons

Report glucose forecast MAE, RMSE, MARD, and clinical error-grid distribution for expected-class and argmax decoders; top-1, top-5, and top-10 accuracy overall and separately for endpoint versus intermediate targets; number of predicted classes, endpoint prediction share, average endpoint probability mass, mean entropy, and mean maximum probability; best epoch and ordinary validation cross-entropy.

## Interpretation

The ablation is promising only if intermediate-class performance improves without unacceptable deterioration in overall glucose forecast metrics or endpoint performance. A higher number of predicted classes alone is not evidence of improvement. Do not scale to all patients based on this single-patient experiment. Do not alter quantization, use test-informed weights, or tune on test metrics.
