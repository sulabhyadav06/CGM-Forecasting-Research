# CRNN/GluNet OhioT1DM Reproduction — Completion Status
**Updated:** 2026-10-29  
**Repository:** sulabhyadav06/CGM-Forecasting-Research  
**Scope:** CRNN and GluNet reproductions on predefined OhioT1DM partitions.

## Executive status
The repository already contains substantial code and preliminary runs. It is **not yet a completed faithful reproduction or publication-ready study**. Existing numerical outputs have been audited in places, but paper-specific preprocessing and architectural details remain unresolved. Raw data are governed by a data-use agreement and are not committed here.

Estimated progress based on code and audit records, not a fresh full run:
- **Infrastructure and preliminary implementations: about 70% complete.**
- **Faithful reproduction, all requested experiments, and final manuscript: about 40% complete.**
These are planning estimates, not claims that the project has passed final verification.

## What is already present

### Data and shared infrastructure
- Loaders exist for predefined train/test partitions for all 12 patients.
- Gap-safe sequence construction uses exact target timestamps and prevents windows/targets crossing large timestamp gaps.
- Chronological validation splits are implemented in the training runners.
- The user's 2026-10-28 one-pass audit reported 24/24 phase-2 CSVs readable, no basic row-integrity errors, and 7/7 preprocessing tests passing.
- The audit found sleep labels all-zero for Ohio 2020 patient 540 train/test and patient 567 test. Constant event-observed flags are uninformative and must not be treated as evidence that events were observed.

### CRNN
- Code exists in experiments/crnn/: model, modified LSTM, data loader, trainer, clinical metric utilities, and implementation notes.
- Existing audited preliminary 30/60-minute results cover 12 patients and 24 patient-horizon files. Recomputed MAE/RMSE/MARD matched saved values within 1e-5 in the reported audit.
- Existing preliminary patient-level mean ± sample SD:

| Metric | 30 min | 60 min |
|---|---:|---:|
| MAE (mg/dL) | 13.4944 ± 1.4801 | 23.6365 ± 2.7142 |
| RMSE (mg/dL) | 19.3091 ± 2.2028 | 32.5765 ± 3.7182 |
| MARD (%) | 9.0727 ± 1.5911 | 15.7825 ± 2.7551 |

These are results of the current implementation, **not proof of exact reproduction**.
- Unresolved fidelity issues: Gaussian filtering is not yet verified/implemented as specified by the paper; convolution activation/padding and training details require source-level confirmation; the modified LSTM has 27,136 parameters versus 24,832 reported in the paper, and uses a learned 32-to-64 residual projection that is an adaptation.

### GluNet
- Code exists for a 256-class gated causal-convolution candidate, quantisation/decoding, training, data loading, and diagnostics.
- Existing all-patient 30/60-minute continuous Huber-regression results are **GluNet-inspired, not the original 256-class classifier**.

| Metric | 30 min | 60 min |
|---|---:|---:|
| MAE (mg/dL) | 13.526 ± 1.628 | 23.507 ± 3.419 |
| RMSE (mg/dL) | 19.318 ± 2.501 | 32.356 ± 4.375 |
| MARD (%) | 9.114 ± 1.745 | 15.907 ± 3.342 |
| Gap-aware lag (min) | 21.528 ± 3.922 | 46.111 ± 9.753 |

These results use a continuous regression objective and must not be presented as exact GluNet results.
- The 256-class candidate exists, but decoder conventions, architecture details, original preprocessing, and optional physiological features still require verification.

## What remains — priority order

| Priority | Work remaining | Completion condition |
|---|---|---|
| P0 | Run CRNN at 30/60 with isolated outputs | 12 patients per horizon or explicit per-patient failure report |
| P0 | Run GluNet 256-class candidate at 30/60 | Same, including class-collapse and class-coverage diagnostics |
| P0 | Verify paper-specific preprocessing and settings against complete papers | Parameter-by-parameter fidelity table with page/figure citations |
| P1 | Implement/verify CRNN Gaussian filtering and eligibility masks | Unit tests plus documented before/after eligibility counts |
| P1 | Resolve modified-LSTM parameter-count/residual discrepancy | Source-aligned architecture or explicitly frozen, named adaptation |
| P1 | Confirm GluNet layer/channel schedule, quantisation boundaries, probability decoder and timestamp features | Boundary tests and documented mapping |
| P1 | Add time-lag and Clarke Error Grid to CRNN patient results | Synthetic tests and patient-level tables |
| P1 | Run both models at 15/90/120 minutes | Complete patient/horizon rows and summaries |
| P2 | Compare against existing baselines on common target timestamps | Alignment audit and patient-wise paired table |
| P2 | Optional physiological Pi and Ra features | Separate experiment after equations/assumptions are defensible |
| P2 | IEEE-style working manuscript | Source compiles and results are populated only from traceable run artifacts |
| P3 | Multi-seed robustness, statistics and expert clinical review | Pre-specified patient-level analysis; no overlapping windows treated as independent subjects |

## One-command local experiment runner
scripts/run_reproduction_suite.sh runs:
1. CRNN at 30 and 60 minutes;
2. GluNet 256-class candidate at 30 and 60 minutes;
3. CRNN at 15, 90 and 120 minutes;
4. GluNet 256-class candidate at 15, 90 and 120 minutes.

It checks the 24 required local partition files before training. All newly generated outputs/logs go under output/reproduction_suite/, not historical result directories. Raw data are not uploaded.

Run from the repository root:
bash scripts/run_reproduction_suite.sh

Expected local files:
~/Desktop/OhioT1DM_data_backup/<patient>_training_multimodal.csv and
~/Desktop/OhioT1DM_data_backup/<patient>_testing_multimodal.csv for all 12 patients.
If the directory differs, set CGM_OHIO_DATA_ROOT=/absolute/path/to/data before running.

Optional runtime controls:
CRNN_EPOCHS=30 GLUNET_EPOCHS=20 bash scripts/run_reproduction_suite.sh

Use the default epochs for the primary run. Reduced epochs are for smoke/debug runs and must be labelled accordingly.

## Important limitation
The runner executes the current repository candidates; it does not magically make them paper-faithful. Until the outstanding fidelity checks are resolved, outputs must be labelled as candidate/reproduction-adapted. The local data and Mac runtime are not accessible to GitHub-hosted tooling from this conversation, so actual training cannot be claimed complete here.
