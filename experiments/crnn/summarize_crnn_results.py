"""Patient-wise + mean±SD table: CRNN vs last-observation persistence, same timestamps."""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.crnn.clinical_metrics import clarke_error_grid_zones
from experiments.crnn.time_lag_metrics import evaluate as lag_evaluate

ROOT = Path("output/crnn_initial")
HORIZONS = (30, 60)


def metrics(df, pred_col, patient, horizon, tag):
    y = df["target_glucose"].to_numpy(float)
    p = df[pred_col].to_numpy(float)
    z = clarke_error_grid_zones(y, p)
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t) / "pred.csv"
        d = df.copy()
        d["predicted_glucose"] = p
        d.to_csv(tmp, index=False)
        lag = lag_evaluate(tmp, patient, horizon)
    row = {
        "model": tag, "patient": patient, "horizon": horizon, "n": len(df),
        "MAE": np.mean(np.abs(p - y)),
        "RMSE": np.sqrt(np.mean((p - y) ** 2)),
        "MARD_%": 100 * np.mean(np.abs(p - y) / y),
        "lag_min": lag["time_lag_min"],
        "effective_horizon_min": lag["effective_prediction_horizon_min"],
        "lag_status": lag["status"],
    }
    for k in "ABCDE":
        row[f"CEG_{k}_%"] = 100 * np.mean(z == k)
    return row


rows = []
for h in HORIZONS:
    for d in sorted(ROOT.glob("patient_*")):
        f = d / f"horizon_{h}min" / "test_predictions.csv"
        if not f.exists():
            continue
        pid = d.name.replace("patient_", "")
        df = pd.read_csv(f)
        rows.append(metrics(df, "predicted_glucose", pid, h, "CRNN"))
        rows.append(metrics(df, "current_glucose", pid, h, "Persistence"))

res = pd.DataFrame(rows)
res.to_csv(ROOT / "patientwise_crnn_vs_persistence.csv", index=False)

num = ["MAE", "RMSE", "MARD_%", "lag_min", "effective_horizon_min"] + [f"CEG_{k}_%" for k in "ABCDE"]
summ = res.groupby(["horizon", "model"])[num].agg(["mean", "std"]).round(2)
summ.to_csv(ROOT / "summary_mean_sd.csv")
print(summ.T.to_string())

for h in HORIZONS:
    a = res[(res.horizon == h) & (res.model == "CRNN")].set_index("patient")["RMSE"]
    b = res[(res.horizon == h) & (res.model == "Persistence")].set_index("patient")["RMSE"]
    print(f"\n{h} min: CRNN beats persistence (RMSE) in {(a < b).sum()}/{len(a)} patients;"
          f" mean RMSE gain {np.mean(b - a):.2f} mg/dL")
