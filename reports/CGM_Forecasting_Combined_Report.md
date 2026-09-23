# CGM Glucose Forecasting on OhioT1DM and Shanghai_T1DM: Combined Findings Report

**Datasets:** OhioT1DM, all 12 patients (6 from the 2018 release: 559, 563, 570, 575, 588, 591; 6 from the 2020 release: 540, 544, 552, 567, 584, 596) — and Shanghai_T1DM, 12 patients (1001-1012)
**Horizons:** 15 / 30 / 60 minutes ahead
**Lookback window:** 60 minutes (12 steps at 5-minute sampling for Ohio; 4 steps at 15-minute sampling for Shanghai — same wall-clock window, different step count)
**Repository:** CGM-Forecasting-Research

This report combines three experiments: (1) a **feature-set ablation** on OhioT1DM (does adding heart rate, meals, sleep, or insulin data to a fixed LSTM improve forecasts?), (2) an **architecture comparison** on OhioT1DM (does a different model — GRU, BiLSTM, TCN, Transformer — outperform a plain LSTM on the best available feature set?), and (3) the same **architecture comparison replicated on the Shanghai_T1DM cohort**, testing whether the best-performing architecture generalizes across datasets, populations, and sampling resolutions. Patient-wise results and mean ± SD summaries are included throughout, so an average never hides a patient-level outlier without being flagged.

---

## 1. Executive Summary

- **Auxiliary signals (HR, carbs, sleep, insulin) give at most a small, mostly non-significant boost** over glucose-only forecasting at 15–30 min horizons on OhioT1DM. The one clear exception: **meal/carb data significantly improves 60-min forecasts** (p=0.021).
- **Architecture matters more than extra features did.** Swapping the model architecture (holding a fixed feature set) produced larger, statistically robust differences than swapping feature sets did.
- **GRU is the best-performing and safest architecture at every horizon tested on OhioT1DM**, significantly beating LSTM, BiLSTM, TCN, and Transformer at 15/30 min, and beating BiLSTM/TCN/Transformer (but tying LSTM) at 60 min.
- **GRU's advantage replicates on Shanghai_T1DM — and holds even more consistently.** GRU significantly beats LSTM at all three horizons on Shanghai (p=0.016, 0.001, 0.043), where on Ohio the GRU-LSTM gap disappeared at 60 min. This is a genuine cross-dataset generalizability result, not an Ohio-specific artifact.
- **TCN and Transformer underperform simple recurrent models on both datasets** — a clean, consistent, and clinically meaningful finding given the limited lookback window and cohort sizes.
- **One Shanghai patient (1012) shows an architecture-ranking reversal across horizons**: GRU is dramatically the best architecture for this patient at 15/30 min, but becomes the *worst* architecture at 60 min. This is flagged as a genuine anomaly worth investigating, not averaged away.
- **Two OhioT1DM patients (540, 567) are consistently the hardest to forecast** across every architecture and horizon, and patient 540 also has the worst clinical-safety numbers. This should not be averaged away either.
- **Shanghai shows lower RMSE but higher MARD than Ohio at every horizon** — likely driven by Shanghai's cleaner data (near-zero missingness) combined with shorter per-patient recordings and a single 80/20 split rather than Ohio's dedicated train/test files, producing more relative-error noise despite better absolute accuracy.

---

## 2. Experiment 1 — Feature-Set Ablation on OhioT1DM (fixed LSTM architecture)

**Protocol:** a single LSTM (64 units) trained separately per patient, per horizon, per feature set. Feature sets B, D, F depend on HeartRate/Steps, which are 100% missing for all 6 2020 patients (different wristband hardware — confirmed by directly inspecting the XML, zero `basis_heart_rate`/`basis_steps` events) — so those three feature sets ran on the 6 2018 patients only. Feature sets A, C, E ran on all 12.

| Feature Set | Channels |
|---|---|
| A | Glucose only |
| B | Glucose + HeartRate *(2018 only, n=6)* |
| C | Glucose + Carbs (meals) |
| D | Glucose + HR + Steps + Carbs + Sleep *(2018 only, n=6)* |
| E | Glucose + Insulin (bolus + basal) |
| F | Full multimodal: HR + Steps + Carbs + Sleep + Insulin *(2018 only, n=6)* |

### 2.1 Statistical significance (Wilcoxon signed-rank, paired by patient)

**Across all 12 patients (A vs. C vs. E):**

| Horizon | Comparison | p-value | Result |
|---|---|---|---|
| 15min | A vs C | 0.24 | not significant |
| 15min | A vs E | 0.63 | not significant |
| 30min | A vs C | 0.11 | not significant |
| 30min | A vs E | 0.85 | not significant |
| **60min** | **A vs C** | **0.021** | **C significantly better (carbs help at 60min)** |
| 60min | A vs E | 0.70 | not significant |

**Within the 6 2018 patients only (does HR/Steps/full multimodal beat glucose-only?):**

| Horizon | A vs B (HeartRate) | A vs D (full wearable) | A vs F (+ insulin) |
|---|---|---|---|
| 15min | p=0.44 (B better 4/6) | p=0.16 (D better 5/6) | p=0.84 (F better 3/6) |
| 30min | p=0.69 (B better 4/6) | p=0.56 (D better 4/6) | p=0.44 (F better 4/6) |
| 60min | p=0.69 (B better 4/6) | p=0.16 (D better 5/6) | p=0.44 (F better 2/6) |

**Interpretation:** none of these 6-patient comparisons reach significance — likely underpowered at n=6 rather than a true null result, given D wins in 5/6 patients at two horizons. "D: full wearable + meals + sleep" is the most promising direction worth revisiting with more patients, but should not yet be claimed as proven.

---

## 3. Experiment 2 — Architecture Comparison on OhioT1DM (fixed feature set)

**Protocol:** LSTM, GRU, BiLSTM, TCN, and Transformer, each trained separately per patient, per horizon, on Glucose + carbs + sleep + insulin (the feature subset available for all 12 patients). All 12 patients included in every architecture/horizon cell.

### 3.1 Summary (mean ± SD across all 12 patients)

| Horizon | Architecture | RMSE | MAE | MARD |
|---|---|---|---|---|
| 15min | GRU | 17.32 ± 2.75 | 12.00 ± 1.53 | 6.99 ± 1.26 |
| 15min | LSTM | 18.81 ± 3.40 | 13.21 ± 2.51 | 8.00 ± 1.64 |
| 15min | BiLSTM | 18.71 ± 3.16 | 13.12 ± 2.09 | 7.35 ± 1.51 |
| 15min | TCN | 24.89 ± 5.85 | 18.15 ± 4.52 | 8.87 ± 1.29 |
| 15min | Transformer | 31.04 ± 4.66 | 22.85 ± 3.20 | 16.16 ± 3.55 |
| 30min | GRU | 24.89 ± 3.35 | 17.75 ± 2.18 | 11.61 ± 2.28 |
| 30min | LSTM | 25.96 ± 3.85 | 18.80 ± 2.73 | 11.92 ± 2.38 |
| 30min | BiLSTM | 26.16 ± 3.90 | 19.03 ± 2.71 | 11.71 ± 2.49 |
| 30min | TCN | 31.48 ± 4.20 | 23.02 ± 3.27 | 13.39 ± 2.79 |
| 30min | Transformer | 36.58 ± 5.17 | 27.33 ± 3.80 | 18.69 ± 4.24 |
| 60min | GRU | 36.78 ± 4.50 | 27.63 ± 3.51 | 17.81 ± 3.58 |
| 60min | LSTM | 37.40 ± 5.04 | 28.01 ± 3.83 | 18.28 ± 3.85 |
| 60min | BiLSTM | 38.01 ± 5.21 | 28.46 ± 3.95 | 19.11 ± 4.14 |
| 60min | TCN | 41.18 ± 4.32 | 31.24 ± 3.58 | 20.04 ± 4.15 |
| 60min | Transformer | 44.38 ± 6.18 | 34.03 ± 5.07 | 24.31 ± 4.88 |

### 3.2 Statistical significance (Wilcoxon signed-rank, GRU vs. each other architecture, n=12)

| Horizon | GRU vs LSTM | GRU vs BiLSTM | GRU vs TCN | GRU vs Transformer |
|---|---|---|---|---|
| 15min | p=0.0015 ** (11/12) | p=0.0015 ** (11/12) | p=0.0005 *** (12/12) | p=0.0005 *** (12/12) |
| 30min | p=0.0210 * (10/12) | p=0.0015 ** (11/12) | p=0.0005 *** (12/12) | p=0.0005 *** (12/12) |
| 60min | p=0.38 (n.s., 7/12) | p=0.0161 * (10/12) | p=0.0005 *** (12/12) | p=0.0005 *** (12/12) |

*(\* p<0.05, \*\* p<0.01, \*\*\* p<0.001)*

**Interpretation:** GRU's advantage over TCN and Transformer is large, consistent, and highly significant at every horizon. GRU's edge over LSTM and BiLSTM is real but smaller, and **disappears entirely between GRU and LSTM at 60 min** (p=0.38) — at longer horizons on Ohio, the two are statistically indistinguishable.

### 3.3 Clinical safety — Clarke Error Grid, dangerous zones (D + E)

| Horizon | GRU | LSTM | BiLSTM | TCN | Transformer |
|---|---|---|---|---|---|
| 15min | **1.11%** | 1.46% | 1.41% | 2.44% | 2.84% |
| 30min | **2.26%** | 2.46% | 2.60% | 2.96% | 3.75% |
| 60min | 4.04% | 4.29% | 4.05% | 4.84% | **5.44% (worst)** |

GRU has the lowest dangerous-zone rate at every horizon (tied with BiLSTM at 60min). Transformer is worst at every horizon, reaching **11.8% dangerous-zone predictions for patient 540 at 60min.**

### 3.4 Per-patient consistency (OhioT1DM, GRU)

**Range at 60 min:** 31.1 (patient 596) to 44.4 mg/dL (patient 540) — a 13.3 mg/dL spread. **Consistently hardest patients across every architecture and horizon: 540 and 567** (both 2020 cohort). No systematic 2018-vs-2020 cohort gap was found (Mann-Whitney p=0.39–0.82 across horizons) — the difficulty is patient-specific, not cohort-specific.

---

## 4. Experiment 3 — Cross-Dataset Generalizability: Architecture Comparison on Shanghai_T1DM

**Protocol:** the same five architectures (LSTM, GRU, BiLSTM, TCN, Transformer), same horizons (15/30/60 min) and same 60-minute lookback window, replicated on the 12 Shanghai_T1DM patients (1001-1012, each patient's earliest recorded visit). Feature set is Glucose-only, since Shanghai's insulin/diet columns are sparse event markers rather than the dense per-timestep channels available for Ohio — this also makes it directly comparable to Ohio's "A: Glucose only" baseline. Each patient's single longest contiguous CGM segment was split chronologically 80/20 into train/test (Shanghai does not ship separate train/test files the way OhioT1DM does).

### 4.1 Summary (mean ± SD across 12 patients, GRU)

| Horizon | RMSE | MARD | CEGA_A (accurate zone) |
|---|---|---|---|
| 15min | 14.59 ± 4.56 | 9.44 ± 3.95% | 88.58% |
| 30min | 20.51 ± 4.58 | 13.98 ± 5.69% | 79.39% |
| 60min | 33.08 ± 12.51 | 22.51 ± 9.61% | 62.60% |

### 4.2 Statistical significance (Wilcoxon signed-rank, GRU vs. each other architecture, n=12)

| Horizon | GRU vs LSTM | GRU vs BiLSTM | GRU vs TCN | GRU vs Transformer |
|---|---|---|---|---|
| 15min | p=0.0161 * (10/12) | p=0.0771 (n.s., 10/12) | p=0.0049 ** (10/12) | p=0.0024 ** (11/12) |
| 30min | p=0.0010 *** (11/12) | p=0.0015 ** (11/12) | p=0.0269 * (9/12) | p=0.0005 *** (12/12) |
| 60min | p=0.0425 * (10/12) | p=0.0771 (n.s., 9/12) | p=0.0034 ** (11/12) | p=0.0005 *** (12/12) |

**GRU's advantage over LSTM is significant at all three horizons on Shanghai** — notably stronger replication than Ohio, where the GRU-LSTM gap vanished at 60 min. GRU vs. TCN/Transformer remains large and significant throughout, matching the Ohio pattern closely.

### 4.3 Ohio vs. Shanghai, direct comparison (GRU)

| Horizon | Ohio RMSE | Shanghai RMSE | Ohio MARD | Shanghai MARD |
|---|---|---|---|---|
| 15min | 17.32 ± 2.75 | 14.59 ± 4.56 | 6.99% | 9.44% |
| 30min | 24.89 ± 3.35 | 20.51 ± 4.58 | 11.61% | 13.98% |
| 60min | 36.78 ± 4.50 | 33.08 ± 12.51 | 17.81% | 22.51% |

Shanghai's RMSE is lower than Ohio's at every horizon — plausibly reflecting its much cleaner CGM data (near-zero missingness vs. Ohio's real sensor gaps). But Shanghai's MARD is higher throughout, and its 60-min SD is nearly 3x Ohio's (12.51 vs. 4.50). Likely drivers: Shanghai recordings are much shorter per patient (7–14 days vs. Ohio's multi-week), giving each model far less training data, and the 80/20 chronological split (vs. Ohio's dedicated, separately-collected test files) is a less rigorous held-out evaluation.

### 4.4 A genuine anomaly: patient 1012's architecture-ranking reversal

Patient 1012 does not follow the overall pattern. At 15/30 min, GRU is *dramatically* the best architecture for this patient (16.21 vs. LSTM's 26.15 at 15min; 25.48 vs. LSTM's 51.00 at 30min). But at 60 min, **GRU becomes the worst-but-one architecture for this same patient** (RMSE 63.56 — only TCN's 67.91 is worse, and it's now behind Transformer's 64.01 and every recurrent alternative). This is a genuine within-patient architecture-ranking reversal across horizons, not a data artifact as far as we can tell from the aggregate pipeline — it warrants direct inspection of this patient's raw CGM trace and predictions before drawing further conclusions from the Shanghai results in isolation. Patient 1005 is also a consistently poor performer across all architectures and horizons (RMSE 48-57 at 60min), though without the same ranking reversal.

---

## 5. Combined Interpretation

Putting all three experiments together: **architecture choice mattered more than feature-set choice on Ohio, and the specific winning architecture (GRU) generalizes to a second, independently-collected dataset with a different population, different sampling rate, and different data quality profile.** This is a stronger result than either experiment alone would suggest — it's not just "GRU happened to fit Ohio's 6 2018 patients well," it's a genuine architecture-level finding that replicates.

That said, the Shanghai replication comes with real caveats the report should not gloss over: higher relative error (MARD) despite lower absolute error (RMSE), much larger patient-to-patient variance at 60 min, and one clear anomalous patient (1012) whose architecture ranking flips entirely at the longest horizon. Before treating "GRU generalizes" as a settled conclusion, it's worth (a) investigating patient 1012 directly, and (b) re-running the Shanghai comparison with a more rigorous train/test split (e.g. a dedicated held-out period per patient, rather than a chronological 80/20 split) to rule out that Shanghai's higher variance is a data-splitting artifact rather than a genuine population difference.

---

## 6. Known Limitations

- **Feature-set comparisons for B/D/F run on only 6 patients** (2018 cohort) — likely underpowered; the promising "D: full wearable" direction needs a larger sample before being treated as established.
- **CEGA zone classification and MARD/time-lag are only computed in Experiments 2 and 3** (architecture comparisons) — Experiment 1 (feature-set) only reports RMSE/MAE/R². A useful follow-up would be re-running the feature-set ablation with the full clinical metric suite.
- **N-BEATS/N-HiTS and TFT were not included** in either architecture comparison — N-BEATS needs a univariate-only variant, and TFT needs a different stack (`pytorch-forecasting`) than the Keras models used here.
- **Shanghai uses a single 80/20 chronological split per patient**, not a dedicated separately-collected test set like Ohio — a genuine methodological gap between the two experiments that likely contributes to Shanghai's higher variance (see §4.3-4.4).
- **Shanghai feature set is glucose-only** — an insulin-inclusive comparison (mirroring Ohio's bolus/basal extraction) has not yet been built for Shanghai.
- **Single train/val/test split per patient in all experiments** — no cross-validation; results reflect one held-out test period per patient, not a K-fold estimate.
- **Patient 1012's anomalous 60-min result has not been root-caused** — flagged in §4.4, not yet investigated further.

---

## Appendix: Data Files

- `featureset_patient_wise.csv` — full per-patient results, Experiment 1 (OhioT1DM feature-set ablation)
- `architecture_patient_wise.csv` — full per-patient results, Experiment 2 (OhioT1DM architecture comparison)
- `shanghai_t1dm_architecture_results.csv` — full per-patient results, Experiment 3 (Shanghai_T1DM architecture comparison)
- `feat_summary.csv` / `arch_summary.csv` / `shanghai_t1dm_architecture_summary.csv` — mean ± SD summary tables in machine-readable form
- `gru_patient_wise.csv` — the per-patient OhioT1DM GRU breakdown used in §3.4
- `architecture_comparison_chart.png` — mean RMSE by architecture across horizons (OhioT1DM)
- `per_patient_rmse.png` — per-patient RMSE spread, glucose-only baseline (OhioT1DM)
