from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

CRNN = Path("output/crnn_initial")
LIT = Path("output/glunet_initial/candidate_20261031_095507")
REG = Path("experiments/glunet/runs/all_patients_regression")
PATIENTS = ["540","544","552","559","563","567","570","575","584","588","591","596"]
SIX = {"559","563","570","575","588","591"}
HORIZONS = (30, 60)

def crnn(p, h):
    d = pd.read_csv(CRNN/f"patient_{p}"/f"horizon_{h}min"/"test_predictions.csv")
    return d[["target_timestamp","target_glucose","current_glucose","predicted_glucose"]].rename(columns={"predicted_glucose":"CRNN"})

def lit(p, h):
    d = pd.read_csv(LIT/f"patient{p}_{h}min"/"test_predictions_and_diagnostics.csv")
    return d[["target_timestamp","prediction_expected_class_value"]].rename(columns={"prediction_expected_class_value":"GluNet_literal"})

reg_all = []
for f in REG.rglob("*prediction*.csv"):
    d = pd.read_csv(f)
    need = {"target_timestamp","predicted_glucose_mg_dL","patient","horizon_minutes"}
    if need <= set(d.columns):
        reg_all.append(d[["patient","horizon_minutes","target_timestamp","predicted_glucose_mg_dL"]])
if not reg_all:
    raise SystemExit("No regression prediction file with the expected columns found")
reg_all = pd.concat(reg_all).drop_duplicates(["patient","horizon_minutes","target_timestamp"])
reg_all["patient"] = reg_all["patient"].astype(str)

rows = []
for h in HORIZONS:
    for p in PATIENTS:
        d = crnn(p, h).merge(lit(p, h), on="target_timestamp")
        r = reg_all[(reg_all.patient == p) & (reg_all.horizon_minutes == h)][["target_timestamp","predicted_glucose_mg_dL"]]
        d = d.merge(r.rename(columns={"predicted_glucose_mg_dL":"GluNet_regression"}), on="target_timestamp")
        d["Persistence"] = d["current_glucose"]
        y = d["target_glucose"].to_numpy(float)
        for m in ["CRNN","GluNet_literal","GluNet_regression","Persistence"]:
            e = d[m].to_numpy(float) - y
            rows.append(dict(patient=p, horizon=h, model=m, n_common=len(d),
                             RMSE=np.sqrt(np.mean(e**2)), MAE=np.mean(np.abs(e)),
                             MARD=100*np.mean(np.abs(e)/y)))
res = pd.DataFrame(rows)
res.to_csv("output/common_timestamp_comparison.csv", index=False)

for h in HORIZONS:
    print(f"\n=== {h} min | common windows per patient (min/max): "
          f"{res[(res.horizon==h)].n_common.min()}/{res[(res.horizon==h)].n_common.max()}")
    for name, sub in (("12 patients", res), ("6 paper patients", res[res.patient.isin(SIX)])):
        g = sub[sub.horizon==h].groupby("model")[["RMSE","MAE","MARD"]].agg(["mean","std"]).round(2)
        print(name); print(g.to_string())
    w = res[res.horizon==h].pivot(index="patient", columns="model", values="RMSE")
    for a, b in [("CRNN","GluNet_regression"),("CRNN","GluNet_literal"),
                 ("GluNet_regression","GluNet_literal"),("CRNN","Persistence"),
                 ("GluNet_regression","Persistence")]:
        diff = w[a] - w[b]
        p = wilcoxon(w[a], w[b]).pvalue
        print(f"RMSE {a} - {b}: mean {diff.mean():+.2f} mg/dL, {int((diff<0).sum())}/12 patients favour {a}, Wilcoxon p={p:.4f}")
