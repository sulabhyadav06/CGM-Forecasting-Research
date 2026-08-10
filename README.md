## Personalized Blood Glucose Forecasting

## Reference Paper
**Title:** Personalized Blood Glucose Forecasting From Limited CGM Data
Using Incrementally Retrained LSTM (IEEE TBME, 2025)
**PMC ID:** PMC11999170.

Note: the reference paper uses OpenAPS and Replace-BG datasets. This
project reproduces the general approach (LSTM-based CGM forecasting,
30/60-min horizons) using the OhioT1DM dataset instead, since it was
the accessible dataset with matching resolution (5-min CGM) and richer
physiological signals (heart rate, steps, sleep, meals) suited for the
multimodal extension in Step 10.

## Dataset
OhioT1DM — 6 patients (559, 563, 570, 575, 588, 591), 5-minute CGM
intervals, plus Basis wearable signals (heart rate, steps, GSR, skin/
air temperature) and self-reported meal/sleep/exercise events.

## Results: Baseline Reproduction (single patient, patient 570, 30-min horizon)

| Model | RMSE (mg/dL) | MAE (mg/dL) |
|---|---|---|
| Linear Regression | 59.47 | 34.88 |
| Random Forest | 63.93 | 40.79 |
| Baseline LSTM (unscaled) | 68.26 | 45.51 |
| LSTM (scaled + engineered features) | 51.62 | 33.91 |

Note: our RMSE is higher than the paper's IS-LSTM benchmark (10.23-13.41
mg/dL on OpenAPS), which uses a more sophisticated incrementally
retrained architecture and a different dataset. Our baseline LSTM
serves as a fair architectural comparison point, not a like-for-like
reproduction.

## Results: Multimodal Extension (6 patients, mean ± std RMSE)

| Horizon | Model | RMSE mean | RMSE std |
|---|---|---|---|
| 30min | A: Glucose only | 23.32 | 2.82 |
| 30min | B: Glucose + HeartRate | 24.08 | 2.31 |
| 30min | C: Glucose + Carbs | 23.07 | 2.94 |
| 30min | D: Full multimodal | 23.10 | 3.28 |
| 60min | A: Glucose only | 35.28 | 3.49 |
| 60min | B: Glucose + HeartRate | 35.99 | 3.45 |
| 60min | C: Glucose + Carbs | 35.55 | 3.45 |
| 60min | D: Full multimodal | 35.39 | 3.71 |

Statistical significance (paired Wilcoxon signed-rank test, n=6 patients):
- A vs B (Heart Rate): **p=0.031** (significant — HR consistently *worsens* RMSE)
- A vs C (Carbs): p=0.44-0.84 (not significant)
- A vs D (Full multimodal): p=0.44-1.0 (not significant)

## Key finding
Across 6 OhioT1DM patients, adding raw heart rate to glucose-only LSTM
forecasting significantly worsened RMSE at both 30-min and 60-min
horizons (Wilcoxon p=0.031). Carbohydrate intake and full multimodal
feature sets showed no statistically significant improvement over
glucose-only baselines. This suggests short-horizon CGM forecasting is
dominated by glucose autocorrelation, and naive physiological feature
addition does not reliably help without further preprocessing (e.g.
HR normalization relative to baseline, activity-aware features, or
testing at longer horizons where autocorrelation weakens further).