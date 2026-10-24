# Project Validation Record

Date: 2026-10-24

## Automated tests
Command:
python -m pytest -q --import-mode=importlib

Recorded result:
78 passed, 20 subtests passed in 3.45s.

## CRNN prediction audit
- 24/24 patient-horizon prediction files checked.
- Zero mismatches against recomputed sample counts, MAE, RMSE, and MARD.
- Horizons: 30 and 60 minutes.
- Patients: 12 OhioT1DM patients.

## CRNN aggregate results

| Metric | 30 minutes | 60 minutes |
|---|---:|---:|
| Mean patient MAE (mg/dL) | 13.4944 | 23.6365 |
| SD patient MAE (mg/dL) | 1.4801 | 2.7142 |
| Mean patient RMSE (mg/dL) | 19.3091 | 32.5765 |
| SD patient RMSE (mg/dL) | 2.2028 | 3.7182 |
| Mean patient MARD (%) | 9.0727 | 15.7825 |
| SD patient MARD (%) | 1.5911 | 2.7551 |

## Limitations
The CRNN results are numerically audited for the current implementation.
Exact reproduction of the published paper has not been established.
GluNet experiments are separate and should not be conflated with CRNN.
Existing local changes, checkpoints, and research outputs must be preserved.
