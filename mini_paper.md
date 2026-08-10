# Multimodal Feature Evaluation for Short-Horizon Blood Glucose Forecasting: A Reproduction and Extension Study on OhioT1DM

**Author:** Sulabh Yadav
**Date:** August 2026

This work reproduces the general approach of Shen and Kleinberg (2025) on a different dataset (OhioT1DM) and extends it with a multimodal ablation study. It is not affiliated with or endorsed by the original authors.

## Abstract

Accurate short-horizon blood glucose (BG) forecasting matters for artificial pancreas systems and for helping patients with Type 1 diabetes act before a glycemic excursion happens. This project builds on the LSTM-based approach used in Personalized Blood Glucose Forecasting From Limited CGM Data Using Incrementally Retrained LSTM (IEEE TBME, 2025), and asks a narrower question: do physiological signals beyond glucose itself actually help short-horizon forecasting? Using six patients from the OhioT1DM dataset, I trained the same LSTM architecture on four different feature sets — glucose alone, glucose plus heart rate, glucose plus meal carbohydrates, and a full combination including step count and sleep state — at both 30- and 60-minute horizons, then tested the differences with paired Wilcoxon signed-rank tests. The results were not what I expected going in. Raw heart rate made predictions significantly worse at both horizons (p=0.031), and neither carbohydrate intake nor the full multimodal set gave a statistically reliable improvement over glucose alone (p=0.44-1.0). Interestingly, this lines up with a similar degradation effect the reference paper reports for meal and insulin data, which suggests the problem may be less about which signals you include and more about how you represent them.

---

## 1. Introduction

People with Type 1 diabetes (T1D) don't produce insulin on their own, so managing blood glucose (BG) means constantly making decisions about insulin dosing, food, and activity. Getting this wrong over time is linked to serious complications like kidney disease and stroke, and even day-to-day, the sheer number of decisions required is exhausting for patients. Continuous glucose monitors (CGMs) have made this easier by giving readings every five minutes without fingersticks, and they're also the backbone of artificial pancreas (AP) systems, which use that stream of data to automatically calculate insulin needs. For either use case — a human making decisions or an automated system doing it — being able to forecast where glucose is heading in the next 30 to 60 minutes is what actually makes preventative action possible.

LSTM and CNN-LSTM models have become the standard approach for this kind of forecasting, and they perform well on average. But "well on average" hides a lot: individual patients can have wildly different forecast accuracy, especially those with high glucose variability, since a model trained across a population doesn't necessarily capture any one person's specific patterns. This is part of why personalized approaches have gained traction, including the Incrementally Retrained Stacked LSTM (IS-LSTM) method that this project takes as its starting point — it adapts to each patient's data over time rather than relying on a fixed population model.

A less settled question is whether adding more than just glucose history — heart rate, activity, meal carbohydrates — actually improves forecasts. It's an intuitive idea: glucose excursions are physiologically driven by exactly these things. Several papers have built multimodal architectures around this assumption and reported gains. But the gains aren't consistent across studies or datasets, and there's a more specific question that doesn't get asked as often: does feeding in *raw* wearable signals help, or does it need to be processed into something more meaningful first? That's the gap this project tries to fill.

The approach here reproduces the general LSTM forecasting setup from the IS-LSTM paper on OhioT1DM, then runs a controlled comparison: same architecture, same training procedure, same horizons, with only the input features changing across four conditions, tested across six patients with paired statistics. Holding everything else constant isolates what each added signal is actually contributing, instead of comparing across papers that differ in architecture and preprocessing in ways that make it hard to tell what's really driving any reported improvement.

## 2. Related Work

Work on CGM-based forecasting spans classical statistics, standard machine learning, and now mostly deep learning. Hameed and Kleinberg, and later CNN-LSTM work on the Replace-BG dataset, got strong average RMSE at 30-minute horizons using glucose alone, but both also found substantial variation between patients — something this project ran into as well, since per-patient RMSE standard deviations ended up comparable in size to the differences between feature-set conditions. That variability is a big part of what motivated the personalized, incrementally retrained IS-LSTM approach used as this project's reference point, though I didn't implement its incremental retraining — models here were trained from scratch per patient instead.

A separate thread of work has gone the multimodal route, adding insulin, carbohydrates, and wearable data to CGM history. Rabby et al. built a stacked LSTM using carbohydrate, insulin, heart rate, and step count as engineered features on OhioT1DM, and found step count in particular helped. Other work has combined CGM with meal, activity, and insulin data in hybrid LSTM-GRU architectures for 15/30/60-minute forecasts. More recently, MetaboNet-Bench has pointed out that this whole area lacks a standardized, reproducible evaluation pipeline — their own results suggest adding modalities generally helps, especially around meals, but that the benefit is inconsistent across models and can get hidden if you only look at aggregate RMSE instead of breaking results out by glycemic range.

There's a useful counterpoint buried in the reference paper's own related work section, though: Hameed and Kleinberg found that adding meal and insulin data actually *hurt* accuracy. The reference paper's own results back this up — their multivariate IS-LSTM (glucose, insulin, carbs) had significantly higher RMSE than the univariate version at both horizons, on both datasets they tested. They attribute this to how sparse meal and insulin events are compared to the dense stream of CGM readings, which makes their influence harder for the model to learn well without a lot of data. This sits in direct tension with Rabby et al.'s finding that step count helped. Neither paper resolves the disagreement, and that's roughly the gap this project's ablation is aimed at — figuring out, at least for heart rate, steps, sleep, and meal-derived features, which direction the effect actually goes.

This project differs from the prior work in two ways. First, instead of proposing a new architecture, it holds the architecture fixed and only varies the feature set, which isolates each modality's contribution more cleanly than comparing across papers with different models and preprocessing. Second, it reports paired significance testing across patients rather than just aggregate RMSE, following MetaboNet-Bench's point that aggregate numbers can hide whether an apparent gain is real or just noise. The finding that heart rate significantly *hurts* accuracy — rather than the more commonly assumed direction that more data helps — suggests that how a signal is represented matters at least as much as whether it's included at all.

## 3. Dataset

This project uses the OhioT1DM dataset, which provides CGM readings every 5 minutes for patients with Type 1 diabetes, along with data from a Basis wearable band (heart rate, galvanic skin response, skin and air temperature, step count) and self-reported logs for meals, sleep, exercise, illness, and stressors.

Six patients were used (IDs 559, 563, 570, 575, 588, 591), each with the chronological training/testing split that comes built into the dataset.

**A note on dataset choice:** the reference paper evaluates on OpenAPS and Replace-BG, not OhioT1DM. I used OhioT1DM instead because it has matching 5-minute CGM resolution plus the richer physiological data this project's multimodal extension needed. That means the glucose-only baseline numbers here aren't directly comparable to the paper's reported RMSE — I'm treating their numbers as a rough literature benchmark, not something to reproduce exactly.

### 3.1 Feature Construction

| Feature | Source | Processing |
|---|---|---|
| Glucose | `glucose_level` | Used directly, 5-min resolution |
| Heart Rate | `basis_heart_rate` | Aligned to glucose timeline via nearest-timestamp match (5-min tolerance) |
| Steps | `basis_steps` | Same alignment as heart rate |
| Carbohydrate intake | `meal` (sparse events) | Converted to rolling "carbs consumed in last 60 minutes" |
| Sleep state | `sleep` + `basis_sleep` | Converted to binary "is sleeping" flag per timestamp |

## 4. Methodology

### 4.1 Problem Formulation

Given a 1-hour window of past readings (12 timesteps at 5-min resolution), predict the glucose value at a fixed prediction horizon (PH) of either 30 minutes (6 steps ahead) or 60 minutes (12 steps ahead).

### 4.2 Model Architecture

Every model uses the same architecture, so the only thing that changes between conditions is the input feature set:

```
Input (window=12, features=N)
  -> LSTM(64)
  -> Dropout(0.2)
  -> Dense(32, ReLU)
  -> Dense(1)
```

Trained with the Adam optimizer, MSE loss, 15 epochs, batch size 32. All inputs and targets were scaled with `MinMaxScaler` fit only on training data.

### 4.3 Feature-Set Conditions

| Model | Features |
|---|---|
| A | Glucose only |
| B | Glucose + Heart Rate |
| C | Glucose + Carbohydrate intake (60-min rolling) |
| D | Glucose + Heart Rate + Steps + Carbohydrate + Sleep |

### 4.4 Baselines

Alongside the LSTM comparison, Linear Regression and Random Forest were run on a single patient (570) as a sanity check for the LSTM's relative performance (see Section 5.1).

### 4.5 Evaluation

RMSE, MAE, and R² were computed on each patient's held-out test split. Each of Models A-D was trained and evaluated separately per patient per horizon, giving 6 paired RMSE values per model per horizon. A paired Wilcoxon signed-rank test (n=6) was used to check whether each multimodal variant (B, C, D) differed significantly from the glucose-only baseline (A).

## 5. Results

### 5.1 Baseline Reproduction (Patient 570, 30-min horizon)

| Model | RMSE (mg/dL) | MAE (mg/dL) |
|---|---|---|
| Linear Regression | 59.47 | 34.88 |
| Random Forest | 63.93 | 40.79 |
| LSTM (unscaled inputs) | 68.26 | 45.51 |
| LSTM (scaled + engineered features) | 51.62 | 33.91 |

Scaling the inputs made a real difference — about a 24% drop in RMSE compared to the unscaled version. Worth flagging: some of what looked like a "model improvement" early in this project was actually just fixing preprocessing, not a better architecture.

### 5.2 Multimodal Feature Comparison (6 patients, mean ± std RMSE)

| Horizon | Model | RMSE mean | RMSE std |
|---|---|---|---|
| 30-min | A: Glucose only | 23.32 | 2.82 |
| 30-min | B: + Heart Rate | 24.08 | 2.31 |
| 30-min | C: + Carbs | 23.07 | 2.94 |
| 30-min | D: Full multimodal | 23.10 | 3.28 |
| 60-min | A: Glucose only | 35.28 | 3.49 |
| 60-min | B: + Heart Rate | 35.99 | 3.45 |
| 60-min | C: + Carbs | 35.55 | 3.45 |
| 60-min | D: Full multimodal | 35.39 | 3.71 |

### 5.3 Statistical Significance

Paired Wilcoxon signed-rank test against Model A (glucose-only), n=6 patients:

| Comparison | 30-min p-value | 60-min p-value | Significant? |
|---|---|---|---|
| A vs B (Heart Rate) | 0.031 | 0.031 | **Yes** — B is worse |
| A vs C (Carbs) | 0.688 | 0.844 | No |
| A vs D (Full multimodal) | 0.438 | 1.000 | No |

## 6. Discussion

The clearest result here is that adding raw heart rate made forecasts significantly worse across all 6 patients, at both horizons. The p-value (0.031) is actually the smallest possible outcome for a 6-pair Wilcoxon test, which means every single patient showed the effect in the same direction — this isn't one or two outliers dragging the average down.

My read on why: heart rate moves on timescales driven by activity and stress that mostly don't line up with glucose dynamics over a 30-60 minute window. Feeding raw, unnormalized heart rate into the model probably just adds noise it has to learn to ignore, and with a relatively small amount of training data per patient, it may end up partially fitting to that noise instead of filtering it out.

Carbohydrate intake and the full multimodal set moved in the direction you'd expect — small improvements — but not enough to be statistically reliable at n=6. This tracks with a broader pattern in short-horizon CGM forecasting: recent glucose history is so autocorrelated that it dominates the prediction, and physiological signals might matter more at longer horizons where that autocorrelation starts to break down.

**Limitations worth being upfront about:**
- Six patients isn't a lot of statistical power. A non-significant result here means "not detected," not "definitely doesn't exist."
- Heart rate and steps were used as raw values, not normalized against each patient's resting baseline or made activity-context-aware. That's probably part of why they didn't help.
- The carbohydrate feature (a rolling 60-minute sum) is a rough stand-in for actual carb absorption dynamics, which are more gradual and complex than a step function.
- Models were trained from scratch per patient rather than using the incremental/transfer learning the reference paper uses, which likely explains some of the RMSE gap against their reported numbers.
- The input window here (1 hour, 12 steps) is shorter than the reference paper's (2 hours, 24 steps), and the architecture (single LSTM layer, 64 units) is simpler than their two-layer stacked LSTM with tuned hyperparameters. Both were chosen to keep the multimodal ablation clean rather than to match their setup exactly, but it does limit direct numerical comparison.
- The null result for carbs and full multimodal sits awkwardly between two conflicting findings already in the literature: Hameed and Kleinberg found meal/insulin data hurt accuracy (which lines up with the heart rate result here), while Rabby et al. found step count helped (which doesn't line up with the null result for Model D here). A separate wide-deep LSTM-GRU model with attention, trained on CGM, activity, carbs, and insulin, also reported that activity data helped (RMSE 17.19±3.22 mg/dL at 30-min PH). Taken together, this inconsistency — including the result in this project — points toward feature engineering and model capacity mattering as much as which modalities get included in the first place.

## 7. Future Work

- Test longer horizons (90-120 min), where glucose autocorrelation should weaken and physiological signals might start to matter more.
- Normalize heart rate against each patient's resting baseline instead of using raw BPM.
- Replace the rolling-window carb feature with something closer to a real absorption curve.
- Try incremental/transfer-learning retraining, as in the reference paper, to see if it closes the RMSE gap to their reported benchmark.
- Extend to the full OhioT1DM cohort and/or a second dataset like OpenAPS to get more statistical power.

## References

1. Shen, Y. and Kleinberg, S. "Personalized Blood Glucose Forecasting from Limited CGM Data Using Incrementally Retrained LSTM." IEEE Transactions on Biomedical Engineering, 72(4), 1266-1277, 2025. doi:10.1109/TBME.2024.3494732
2. Marling, C. and Bunescu, R. "The OhioT1DM Dataset for Blood Glucose Level Prediction: Update 2020." CEUR Workshop Proceedings, 2675, 71-74, 2020.
3. Hameed, H. and Kleinberg, S. "Comparing machine learning techniques for blood glucose forecasting using free-living and patient generated data." Proceedings of Machine Learning Research, 126, 871-894, 2020. (cited via reference 1's related work)
4. Rabby, M.F., Tu, Y., Hossen, M.I., Lee, I., Maida, A., and Hei, X.S. "Stacked LSTM based deep recurrent neural network with Kalman smoothing for blood glucose prediction." BMC Medical Informatics and Decision Making, 21, 2021. (cited via reference 1's related work; used heart rate and step count as engineered features)
5. Kalita, D., Sharma, H., Panda, J.K., and Mirza, K.B. "Platform for precise, personalised glucose forecasting through continuous glucose and physical activity monitoring and deep learning." Medical Engineering & Physics, October 2024. https://www.sciencedirect.com/science/article/abs/pii/S1350453324001425
6. MetaboNet-Bench: A Multi-modal Benchmark for Glucose Forecasting in Type 1 Diabetes, 2025. https://arxiv.org/pdf/2606.18640
