# Reproducing the CRNN / GluNet OhioT1DM benchmark

Scope: 30- and 60-minute forecasts for all 12 OhioT1DM patients (2018 and 2020 releases), predefined
training/testing partitions, chronological validation inside the training partition.
Not yet done: 15-, 90- and 120-minute horizons, multiple seeds, GluNet P_i / R_a inputs.

## 1. Data (not included)
OhioT1DM is distributed under a Data Use Agreement and must not be committed to this repository.
Place the preprocessed files `<patient>_training_multimodal.csv` and `<patient>_testing_multimodal.csv`
(patients 540, 544, 552, 559, 563, 567, 570, 575, 584, 588, 591, 596) in one folder and set:

    export CGM_OHIO_DATA_ROOT="/path/to/that/folder"

## 2. Environment
Python 3.10, PyTorch 2.x (runs used Apple MPS; CPU also works but is slower), pandas, numpy, scipy, matplotlib.
All CRNN-pipeline runs use seed 42 (set per patient in `experiments/crnn/train_crnn.py`).

## 3. Runs (from the repository root)

Primary CRNN (batch 128, inputs standardised with training-window statistics):

    for h in 30 60; do python -u -m experiments.crnn.train_crnn --horizon $h --epochs 50 \
      --batch-size 128 --normalize --output-root output/crnn_v2_bs128_norm; done

CRNN ablation (batch 256, no standardisation; the original run, kept as `output/crnn_initial_bs256_backup`):

    for h in 30 60; do python -m experiments.crnn.train_crnn --horizon $h --epochs 50 \
      --batch-size 256 --output-root output/crnn_initial_bs256_backup; done

Baselines through the identical pipeline (GRU, LSTM, BiLSTM, TCN, Transformer; same recipe as the primary CRNN):

    for m in gru lstm tcn transformer bilstm; do for h in 30 60; do \
      python -u -m experiments.crnn.train_crnn --model $m --horizon $h --epochs 50 --batch-size 128 \
      --normalize --output-root output/bench_$m; done; done

GluNet, paper-literal label transform (256 classes, clipped at about +/-12.8 mg/dL):

    python -m experiments.glunet.train_glunet_candidate_all_patients --data-root "$CGM_OHIO_DATA_ROOT" \
      --output-dir output/glunet_initial --horizons 30 60 --epochs 30 --batch-size 128

GluNet regression adaptation (unclipped glucose change): `experiments/glunet/train_glunet_regression_all_patients.py`
(plain), `..._huber_all_patients.py`, `..._mae_all_patients.py`; run with `--help` for arguments. Their outputs
are under `experiments/glunet/runs/all_patients_regression*`. Clinical/lag evaluation of these runs:
`python -m experiments.glunet.evaluate_regression_clinical` (writes `experiments/glunet/reports/clinical_evaluation/`).

## 4. Evaluation and report assets

    python -m experiments.benchmark_all          # common-timestamp metrics, lag, Clarke, paired Wilcoxon + Holm
    python experiments/paper_style_lag.py        # lag computed per ~3-day period, as the GluNet paper describes
    python experiments/make_report_assets.py     # patient-wise LaTeX tables and figure into report/

`benchmark_all` reads the folders above and writes `output/benchmark/{patientwise,summary_12,summary_6,paired_tests}.csv`.
All models are scored on the intersection of their eligible test timestamps. Last-value persistence is included as a baseline.
The manuscript is `report/manuscript.tex`; every table cites the CSV it comes from.

## 5. Documented deviations from the papers
- CRNN residual: the paper adds a 32-dim input to a 64-unit output; we add a learned 32->64 projection (2,048 parameters).
  PyTorch's LSTM has a second bias vector (+256 parameters), so the LSTM has 27,136 parameters vs 24,832 in the paper's Table V.
- No Gaussian smoothing (the paper gives no filter width; a symmetric filter would use samples after the forecast origin).
- No interpolation: windows with missing timestamps are dropped (stricter than the papers).
- Input standardisation (primary CRNN only) is our addition. Dropout 0.2 and batch size are our choices where unspecified.
- GluNet: padding, optimiser and decoding are unspecified in the paper; the candidate is an interpretation, and the
  regression variant is an adaptation. Results use the probability-weighted decoder.
- Baseline models are small, untuned, and share one training recipe.

## 6. Known limitations
Single seed; validation slice (last 15% of training) behaves differently for some patients; prediction lag is long for all
models (about 23 and 49 min) and the papers' short lags are not reproduced.

## 7. Before committing
Do not commit `data/`, raw OhioT1DM files, or model checkpoints. Suggested `.gitignore` additions:

    data/
    *.xml
    output/**/best_model.pt
    logs/
