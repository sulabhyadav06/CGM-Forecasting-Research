"""Common-timestamp benchmark of every model, same eligible test windows.

Models (those whose outputs exist are used):
  CRNN (primary v2), CRNN_bs256 (ablation), GRU, LSTM, BiLSTM, TCN, Transformer,
  GluNet_literal (paper Eq.4 labels), GluNet_regression, Persistence.
Writes output/benchmark/{patientwise,summary_12,summary_6,paired_tests}.csv
"""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from experiments.crnn.clinical_metrics import clarke_error_grid_zones
from experiments.crnn.time_lag_metrics import evaluate as lag_evaluate

PATIENTS = ["540","544","552","559","563","567","570","575","584","588","591","596"]
SIX = {"559","563","570","575","588","591"}
HORIZONS = (15, 30, 60, 90, 120)
PRIMARY = Path("output/crnn_v2_bs128_norm")
ABLATION = Path("output/crnn_initial_bs256_backup")
BENCH = {"GRU": "gru", "LSTM": "lstm", "TCN": "tcn", "Transformer": "transformer", "BiLSTM": "bilstm"}
LIT_RUNS = sorted(Path("output/glunet_initial").glob("candidate_*"))
REG = Path("experiments/glunet/runs/all_patients_regression")
OUT = Path("output/benchmark")
OUT.mkdir(parents=True, exist_ok=True)

def crnn_style(root, p, h):
    f = root / f"patient_{p}" / f"horizon_{h}min" / "test_predictions.csv"
    return pd.read_csv(f) if f.exists() else None

reg_all = []
if REG.exists():
    for f in REG.rglob("*prediction*.csv"):
        d = pd.read_csv(f)
        if {"target_timestamp","predicted_glucose_mg_dL","patient","horizon_minutes"} <= set(d.columns):
            reg_all.append(d[["patient","horizon_minutes","target_timestamp","predicted_glucose_mg_dL"]])
reg_all = (pd.concat(reg_all).drop_duplicates(["patient","horizon_minutes","target_timestamp"])
           if reg_all else None)
if reg_all is not None:
    reg_all["patient"] = reg_all["patient"].astype(str)

rows = []
for h in HORIZONS:
    for p in PATIENTS:
        base = crnn_style(PRIMARY, p, h)
        if base is None:
            print(f"SKIP {p} {h}: primary CRNN predictions missing"); continue
        series = {"CRNN": base.set_index("target_timestamp")["predicted_glucose"]}
        for name, root in (("CRNN_bs256", ABLATION),):
            d = crnn_style(root, p, h)
            if d is not None: series[name] = d.set_index("target_timestamp")["predicted_glucose"]
        for name, key in BENCH.items():
            d = crnn_style(Path(f"output/bench_{key}"), p, h)
            if d is not None: series[name] = d.set_index("target_timestamp")["predicted_glucose"]
        for run in reversed(LIT_RUNS):
            f = run / f"patient{p}_{h}min" / "test_predictions_and_diagnostics.csv"
            if f.exists():
                d = pd.read_csv(f)
                series["GluNet_literal"] = d.set_index("target_timestamp")["prediction_expected_class_value"]
                break
        if reg_all is not None:
            r = reg_all[(reg_all.patient == p) & (reg_all.horizon_minutes == h)]
            if len(r): series["GluNet_regression"] = r.set_index("target_timestamp")["predicted_glucose_mg_dL"]
        common = set(base["target_timestamp"])
        for s in series.values():
            common &= set(s.index)
        b = base[base["target_timestamp"].isin(common)].reset_index(drop=True)
        if len(b) < 200:
            print(f"SKIP {p} {h}: only {len(b)} common windows"); continue
        y = b["target_glucose"].to_numpy(float)
        preds = {m: s.reindex(b["target_timestamp"]).to_numpy(float) for m, s in series.items()}
        preds["Persistence"] = b["current_glucose"].to_numpy(float)
        for m, pr in preds.items():
            e = pr - y
            z = clarke_error_grid_zones(y, pr)
            with tempfile.TemporaryDirectory() as t:
                tmp = Path(t) / "pred.csv"
                d = b.copy(); d["predicted_glucose"] = pr; d.to_csv(tmp, index=False)
                lag = lag_evaluate(tmp, p, h)["time_lag_min"]
            row = dict(model=m, patient=p, horizon=h, n_common=len(b),
                       RMSE=np.sqrt(np.mean(e**2)), MAE=np.mean(np.abs(e)),
                       MARD=100*np.mean(np.abs(e)/y), lag_min=lag)
            for k in "ABCDE": row[f"CEG_{k}"] = 100*np.mean(z == k)
            rows.append(row)

res = pd.DataFrame(rows)
res.to_csv(OUT / "patientwise.csv", index=False)
cols = ["RMSE","MAE","MARD","lag_min","CEG_A","CEG_B","CEG_D"]
for name, sub in (("12", res), ("6", res[res.patient.isin(SIX)])):
    g = sub.groupby(["horizon","model"])[cols].agg(["mean","std"]).round(2)
    g.to_csv(OUT / f"summary_{name}.csv")
    print(f"\n===== mean/SD over {name} patients (common timestamps) =====")
    for h in HORIZONS:
        t = g.loc[h]
        out = pd.DataFrame({c: t[c]["mean"].astype(str) + " ± " + t[c]["std"].astype(str) for c in cols})
        print(f"\n-- {h} min --"); print(out.sort_values("RMSE").to_string())

tests = []
for h in HORIZONS:
    w = res[res.horizon == h].pivot(index="patient", columns="model", values="RMSE")
    if "CRNN" not in w: continue
    cand = []
    for m in w.columns:
        if m == "CRNN": continue
        diff = (w[m] - w["CRNN"]).dropna()
        if len(diff) < 6 or (diff == 0).all(): continue
        p = wilcoxon(w.loc[diff.index, m], w.loc[diff.index, "CRNN"]).pvalue
        cand.append([h, m, len(diff), diff.mean(), int((diff > 0).sum()), p])
    cand.sort(key=lambda r: r[5])
    k = len(cand); run = 0.0
    for i, r in enumerate(cand):
        run = max(run, min(1.0, (k - i) * r[5])); r.append(run)
    tests += cand
tt = pd.DataFrame(tests, columns=["horizon","model_vs_CRNN","n_patients","mean_RMSE_diff","CRNN_better_in","p_raw","p_holm"])
tt.to_csv(OUT / "paired_tests.csv", index=False)
print("\n===== paired Wilcoxon on RMSE: model minus CRNN (positive = CRNN better) =====")
print(tt.round(4).to_string(index=False))
