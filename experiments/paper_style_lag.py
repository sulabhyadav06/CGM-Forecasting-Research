from pathlib import Path
import numpy as np, pandas as pd

def chunk_lag(t, y, p, max_min=90, min_min=-30, step=5, chunk_days=3):
    t = pd.to_datetime(pd.Series(t)).reset_index(drop=True)
    y = pd.Series(np.asarray(y, float), index=t.values)
    p = pd.Series(np.asarray(p, float), index=t.values)
    ck = ((t - t.min()) // pd.Timedelta(days=chunk_days)).values
    lags = []
    for c in np.unique(ck):
        m = ck == c
        yc, pc = y[m], p[m]
        if len(yc) < 100: continue
        best, bl = -2, np.nan
        for k in range(min_min, max_min + 1, step):
            ys = yc.copy(); ys.index = ys.index + pd.Timedelta(minutes=k)
            j = pc.index.intersection(ys.index)
            if len(j) < 50: continue
            r = np.corrcoef(pc.loc[j], ys.loc[j])[0, 1]
            if np.isfinite(r) and r > best + 1e-12: best, bl = r, k
        if np.isfinite(bl): lags.append(bl)
    return (np.mean(lags) if lags else np.nan), len(lags)

CRNN = Path("output/crnn_initial")
REG = Path("experiments/glunet/runs/all_patients_regression")
PATIENTS = ["540","544","552","559","563","567","570","575","584","588","591","596"]

reg = []
for f in REG.rglob("*prediction*.csv"):
    d = pd.read_csv(f)
    if {"target_timestamp","predicted_glucose_mg_dL","patient","horizon_minutes","target_glucose"} <= set(d.columns):
        reg.append(d[["patient","horizon_minutes","target_timestamp","target_glucose","predicted_glucose_mg_dL","current_glucose"]])
reg = pd.concat(reg).drop_duplicates(["patient","horizon_minutes","target_timestamp"])
reg["patient"] = reg["patient"].astype(str)

rows = []
for h in (30, 60):
    for p in PATIENTS:
        c = pd.read_csv(CRNN/f"patient_{p}"/f"horizon_{h}min"/"test_predictions.csv")
        r = reg[(reg.patient == p) & (reg.horizon_minutes == h)]
        for name, d, col in (("CRNN", c, "predicted_glucose"),
                             ("Persistence", c, "current_glucose"),
                             ("GluNet_regression", r, "predicted_glucose_mg_dL")):
            lag, n = chunk_lag(d["target_timestamp"], d["target_glucose"], d[col])
            rows.append(dict(model=name, horizon=h, patient=p, lag_min=lag, n_chunks=n))
res = pd.DataFrame(rows)
res.to_csv("output/paper_style_lag.csv", index=False)
print(res.groupby(["horizon","model"]).agg(lag_mean=("lag_min","mean"), lag_sd=("lag_min","std"),
                                           chunks=("n_chunks","mean")).round(2).to_string())
