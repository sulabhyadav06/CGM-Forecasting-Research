# Personalized Multimodal Blood Glucose Forecasting

**Research stage:** exploratory research prototype with corrected, leakage-safe Phase-2 experiments. This repository is not clinically validated and does not claim a universally superior model.

## Overview

This project investigates personalized blood-glucose forecasting from continuous glucose monitoring (CGM) and available physiological/contextual signals.

The study covers:

- OhioT1DM 2018 and 2020 cohorts
- 12 patients
- multimodal feature construction
- 15, 30, 60, 90 and 120-minute forecasting
- LSTM, GRU, BiLSTM, TCN and Transformer baselines
- pairwise and three-way hybrid architectures
- adaptive TCN-GRU-Transformer fusion
- feature ablation
- capacity-controlled architecture comparison
- persistence baseline comparison
- patient-wise clinical/error metrics
- statistical analysis
- Shanghai T1DM exploratory evaluation
- reproducibility and protocol-integrity checks

## Main research notebook

The primary research notebook is:

`CGM_Forecasting_Experiments.ipynb`

It documents the complete experimental pipeline, saved results, architecture investigation, ablations, final 12-patient hybrid experiment, Shanghai protocol, limitations and reproducibility checks.

## Research question

Can recent CGM history and available physiological/context signals improve personalized glucose forecasts at 15, 30, 60, 90 and 120 minutes compared with simpler neural models and a last-observation persistence baseline?

A second methodological question is whether complementary temporal operators—TCN, GRU and Transformer—have distinct useful roles in multi-horizon forecasting, and whether their combination should be adaptively fused rather than simply concatenated.

## Dataset and permitted use

The project uses OhioT1DM 2018 and 2020 releases under the applicable Data Use Agreement.

**Raw XML and patient data are intentionally not included in this repository.** Keep authorized source files locally and outside version control.

### OhioT1DM 2018

Patients: `559, 563, 570, 575, 588, 591`

Available channels include CGM, meal/carbohydrate events, insulin, heart rate and steps, subject to per-file availability.

### OhioT1DM 2020

Patients: `540, 544, 552, 567, 584, 596`

Available channels include CGM, meal/carbohydrate events, insulin and acceleration. Feature availability differs from the 2018 cohort.

### Shanghai T1DM

Shanghai experiments are maintained separately because their preprocessing and evaluation protocol is not identical to the corrected OhioT1DM protocol.

## Corrected Phase-2 protocol

- **Patients:** all 12 OhioT1DM participants
- **Horizons:** 15, 30, 60, 90 and 120 minutes
- **Baseline:** last-observation persistence
- **Architectures:** LSTM, GRU, BiLSTM, TCN, Transformer, TCN-GRU, GRU-Transformer, TCN-Transformer, TCN-GRU-Transformer and adaptive TCN-GRU-Transformer
- **Evaluation:** patient-level MAE, RMSE, MARD, signed time lag and Clarke Error Grid zone distribution
- **Splitting:** chronological
- **Preprocessing:** train-only imputation and scaling
- **Target handling:** target glucose is not imputed
- **Sequence construction:** exact contiguous 5-minute windows
- **Leakage protection:** training targets cannot cross split boundaries
- **Sensor aggregation:** causal aggregation only
- **Gap handling:** windows crossing timestamp gaps are rejected

## Architecture investigation

The hybrid architecture was not selected simply by combining three popular networks.

| Component | Intended role |
|---|---|
| **GRU** | Sequential state evolution and recurrent temporal dynamics |
| **TCN** | Local and multi-scale temporal patterns through dilated convolutions |
| **Transformer** | Longer-range temporal dependencies and attention-based context |

The multimodal feature vector first passes through a shared embedding. The three branches then act as different temporal operators.

Their representations are **adaptively fused with learned gates**, rather than simply concatenated.

The prediction module uses **horizon-specific heads** for the five forecast horizons.

### Important finding

The experiments do **not** establish that the three-way hybrid is universally optimal.

Development ablations and capacity-controlled experiments show that the best architecture can vary by forecast horizon. GRU performs strongly at short horizons, while more complex combinations can become useful at longer horizons.

Therefore the defensible conclusion is that forecast horizon affects the relative value of different temporal mechanisms—not that one complex architecture always wins.

## Final 12-patient hybrid experiment

The final hybrid run contains:

**12 patients × 5 horizons = 60 patient-horizon results.**

Saved results:

`output/phase2_hybrid_all12_run2/all_12_test_results.csv`

Mean results:

| Horizon | Mean MAE (mg/dL) | Mean RMSE (mg/dL) |
|---:|---:|---:|
| 15 min | 16.77 | 23.16 |
| 30 min | 21.41 | 29.23 |
| 60 min | 29.92 | 40.20 |
| 90 min | 36.43 | 48.18 |
| 120 min | 40.46 | 53.06 |

These values are descriptive and should not be interpreted as universal superiority over all baselines.

## Persistence baseline

Persistence is especially difficult to beat at short forecasting horizons.

The corrected experiments show that model advantages become more visible at longer horizons. Persistence should therefore remain in serious comparisons.

## Feature ablation

Feature experiments evaluate:

- glucose
- carbohydrates
- insulin
- heart rate
- steps
- acceleration for the 2020 cohort
- sleep where available

The common-feature experiment uses glucose + carbohydrates + insulin across all 12 patients.

Sleep was not treated as a fully audited independent ablation because it was not consistently available in the corrected Phase-2 CSV inputs.

## Capacity-controlled architecture comparison

The capacity-controlled sweep compares:

- GRU
- GRU-TCN
- GRU-Transformer
- Full GRU-TCN-Transformer

at approximately 100K, 160K and 256K parameters.

The purpose is to distinguish architectural effects from raw parameter-count effects.

The results do not support a universal claim that the largest or most complex architecture is always best.

## Clinical and statistical evaluation

The project evaluates:

- MAE
- RMSE
- MARD
- signed time lag
- Clarke Error Grid zone distribution

Paired statistical comparisons are included where available. Because the number of patients is limited and many architecture/horizon comparisons are made, statistical results should be interpreted as exploratory.

## Shanghai evaluation

The Shanghai line contains 12 patients, 15/30/60-minute horizons, and BiLSTM, GRU, LSTM, TCN and Transformer comparisons.

Shanghai results should be reported separately rather than pooled directly with Ohio results.

## Repository layout

```text
.
├── CGM_Forecasting_Experiments.ipynb
├── hybrid_models.py
├── hybrid_models_v2.py
├── hybrid_experiments.py
├── hybrid_ablation_experiments.py
├── phase2_hybrid_model.py
├── run_phase2_hybrid_all_patients.py
├── phase2_feature_ablation.py
├── ohio2020_acceleration_preprocessing.py
├── phase2_multimodal_preprocessing.py
├── clinical_metrics.py
├── tests/
├── data/
├── output/
└── reports/
```

## Environment setup

```bash
git clone https://github.com/sulabhyadav06/CGM-Forecasting-Research.git
cd CGM-Forecasting-Research

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements-research.txt
python -m pip install jupyter ipykernel
```

## Data placement

Authorized data can use layouts supported by the experiment loaders, including:

```text
data/train/540.csv
data/test/540.csv
```

or:

```text
data/540_training_multimodal.csv
data/540_testing_multimodal.csv
```

Check the relevant preprocessing and experiment scripts for exact accepted layouts.

**Never commit restricted patient data or source XML files to GitHub.**

## Run protocol tests

```bash
python -m pytest -q tests/test_forecast_alignment.py tests/test_research_protocol.py
```

Passing these tests does not establish clinical safety or scientific superiority.

## Run the research notebook

From the repository root:

```bash
jupyter notebook CGM_Forecasting_Experiments.ipynb
```

The notebook primarily reads saved result artifacts. Expensive training experiments should be launched explicitly through the corresponding scripts.

## Optional smoke test

```bash
python hybrid_experiments.py   --data-root data   --patient 540   --horizons 15 30 60 90 120   --lookback 120   --epochs 2   --output-dir output/smoke_test_540
```

For the full multi-patient hybrid run:

```bash
python run_phase2_hybrid_all_patients.py   --data-root data   --output-root output/phase2_hybrid_corrected   --horizons 15 30 60 90 120   --lookback 120   --epochs 15   --seed 42
```

## Important reports

Key reports include:

```text
reports/FINAL_PROJECT_STATUS_2026-10-03.md
reports/FULL_ARCHITECTURE_RESULTS_2026-10-03.md
reports/FEATURE_ABLATION_RESULTS_2026-10-03.md
reports/DATA_QUALITY_AUDIT_2026-10-03.md
reports/REPRODUCIBLE_RUNBOOK_2026-10-03.md
reports/STATISTICAL_ANALYSIS_2026-10-03.md
reports/LOOKBACK_SENSITIVITY_2026-10-03.md
reports/CORE_RUN_RESULTS_2026-10-03.md
reports/SHANGHAI_PROTOCOL_AUDIT_2026-10-05.md
reports/HYBRID_VS_BASELINE_STATISTICS_2026-10-05.md
```

Corrected Phase-2 reports supersede affected earlier exploratory results.

## Limitations

1. Several architecture experiments use single seeds and relatively short training schedules.
2. Capacity-controlled architecture experiments are primarily development-stage evidence.
3. No architecture is a universal winner across horizons.
4. Persistence remains a strong short-horizon baseline.
5. Feature availability and missingness differ between OhioT1DM cohorts.
6. The 2020 acceleration stream has substantial missingness.
7. Patient-wise sample sizes are limited.
8. Shanghai and Ohio experiments use different protocols.
9. Raw data require separate authorized access.
10. This repository is a research prototype, not a medical device or treatment system.

## Reproducibility principle

Every reported result should be traceable to:

1. source dataset and cohort
2. preprocessing configuration
3. temporal split
4. feature set
5. model architecture
6. training configuration
7. random seed
8. saved result artifact
9. corresponding report or notebook section

Do not mix corrected five-horizon results with older exploratory tables unless their protocols have been explicitly reconciled.

## Data access

Dataset access and redistribution are governed by the applicable dataset agreements. Do not redistribute restricted patient data with this repository.
