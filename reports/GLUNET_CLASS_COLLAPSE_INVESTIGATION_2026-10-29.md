# GluNet Class-Collapse Investigation — 2026-10-29

## Status

The 256-class model is an explicitly labelled candidate, not a verified exact reproduction. A two-epoch smoke run for patient 540 at a 30-minute horizon predicted only classes 0 and 255. This is a warning, not a conclusive training diagnosis: two epochs are too few to establish whether the collapse persists.

## Evidence collected locally

The user ran read-only checks on `540_training_multimodal.csv` and `540_testing_multimodal.csv`.

| Diagnostic | Training | Testing |
|---|---:|---:|
| Exact 30-minute pairs | 10,918 | 2,630 |
| Median 30-minute change (mg/dL) | -1 | -3 |
| Min / max change (mg/dL) | -115 / +133 | -116 / +97 |
| Changes outside the approximately [-12.7, +12.8] mg/dL class-centre range | 53.16% | 57.83% |
| Out-of-range changes among complete 7-reading sequences | 53.09% | 57.81% |

Most timestamp intervals in the training CSV were 300 seconds (11,769); only 30 intervals differed from 300 seconds by more than 30 seconds. The examples shown for large glucose changes often changed smoothly across consecutive readings. This does not prove that every extreme change is physiologically valid, but it argues against blaming missing 5-minute readings alone.

## Interpretation

1. Endpoint clipping is part of the published 256-class label transform. Do not widen the label range merely to reduce endpoint frequency; that would change the method.
2. The high endpoint share makes preprocessing fidelity especially important.
3. The paper describes preprocessing that includes outlier handling for CGM/insulin/meal data, interpolation for shorter CGM gaps, treatment of longer gaps as separate segments, and alignment of input channels. The current candidate drops rows with missing timestamps/glucose and builds gap-safe windows, but does not implement the paper's complete outlier and interpolation/extrapolation pipeline. This is a material reproduction gap.
4. The paper describes a probabilistic forecast distribution and recovery from quantized change classes. The candidate currently reports both expected-class-value and argmax decoders; neither should be described as the exact paper decoder until confirmed from the source.
5. The candidate's timestamp sine/cosine features, feature scaling, median imputation, validation split, optimizer settings, and channel/skip interpretation must remain documented as adaptations until verified.

## Change made for the next controlled run

The candidate trainer now records validation diagnostics after each epoch in the existing `history.csv`:
- unique true and predicted classes;
- percentage of true labels at endpoint classes 0/255;
- percentage of predictions at endpoint classes 0/255;
- share of the most common predicted class;
- mean predictive entropy and mean maximum class probability.

The objective, optimizer, data, and checkpoint-selection rule are unchanged by this instrumentation. These diagnostics should show whether class collapse is persistent across epochs, rather than relying on a two-epoch final prediction histogram.

## Next local run

Run the diagnostic candidate only for patient 540 and the 30-minute horizon first, using a fresh output directory. Do not overwrite prior smoke-test artifacts. Use 20 epochs and early stopping:

```bash
cd ~/Desktop/CGM-reproduction-test
git fetch origin
git switch --detach origin/research/glunet-collapse-diagnostics
export CGM_OHIO_DATA_ROOT="$HOME/Desktop/OhioT1DM_data_backup"
python -m experiments.glunet.train_glunet_candidate_all_patients \
  --data-root "$HOME/Desktop/OhioT1DM_data_backup" \
  --output-dir "$HOME/Desktop/CGM-reproduction-test/output/glunet_collapse_diagnostic" \
  --patients 540 \
  --horizons 30 \
  --epochs 20 \
  --patience 5 \
  --batch-size 128
```

The script creates a timestamped run directory and does not overwrite earlier run directories. Review its `patient540_30min/history.csv`, `class_collapse_diagnostics.csv`, and `test_predictions_and_diagnostics.csv` before running all patients.

## Interpretation rules for the next run

- If predicted unique classes remain near 2 while validation CE plateaus, investigate preprocessing and class-target handling before architecture tuning.
- If class diversity increases but test predictions still collapse, inspect validation-vs-test class distribution and generalization.
- If validation predictions are diverse but test predictions collapse, inspect split distribution, temporal drift, and test preprocessing.
- Do not use SMOTE, endpoint reweighting, a wider quantizer, or test-set tuning as a first response. Any such experiment must be a separately named ablation with a defensible rationale.
- Never claim this candidate is a faithful reproduction until paper-specific preprocessing, architecture, and decoder choices are verified.

## Primary reference

Li K, Liu C, Zhu T, Herrero P, Georgiou P. *GluNet: A Deep Learning Framework for Accurate Glucose Forecasting*. IEEE Journal of Biomedical and Health Informatics. 2020;24(2):414–423. doi:10.1109/JBHI.2019.2931842.
