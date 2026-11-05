"""Build patient-wise LaTeX tables and a figure for the manuscript.

Reads  output/benchmark/patientwise.csv   (written by experiments/benchmark_all.py)
Writes report/tables/patientwise_{RMSE,MARD}_{30,60}.tex  and  report/figures/rmse_by_patient.pdf
Run from the repository root:  python experiments/make_report_assets.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path("output/benchmark/patientwise.csv")
OUT = Path("report")
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(parents=True, exist_ok=True)

ORDER = ["CRNN", "CRNN_bs256", "GluNet_regression", "GluNet_literal",
         "GRU", "LSTM", "BiLSTM", "TCN", "Transformer", "Persistence"]
LABEL = {"CRNN": "CRNN", "CRNN_bs256": "CRNN-256", "GluNet_regression": "GN-reg",
         "GluNet_literal": "GN-lit", "GRU": "GRU", "LSTM": "LSTM", "BiLSTM": "BiLSTM",
         "TCN": "TCN", "Transformer": "Trans.", "Persistence": "Persist."}
res = pd.read_csv(SRC)
res["patient"] = res["patient"].astype(str)

for h in sorted(res.horizon.unique()):
    for metric, unit in (("RMSE", "mg/dL"), ("MARD", "\\%")):
        piv = res[res.horizon == h].pivot(index="patient", columns="model", values=metric)
        models = [m for m in ORDER if m in piv.columns]
        piv = piv[models].sort_index(key=lambda x: x.astype(int))
        n = res[(res.horizon == h) & (res.model == "CRNN")].set_index("patient")["n_common"]
        lines = [r"\begin{table*}[t]\centering",
                 rf"\caption{{Patient-wise {metric} ({unit}) at {h}~min on common test windows; best per patient in bold. "
                 r"CRNN: primary configuration; CRNN-256: batch-256 ablation; GN-reg/GN-lit: GluNet regression/literal. "
                 rf"Source: \texttt{{output/benchmark/patientwise.csv}}.}}",
                 r"\scriptsize\setlength{\tabcolsep}{3pt}",
                 r"\begin{tabular}{lr" + "c" * len(models) + r"}\toprule",
                 "Patient & $n$ & " + " & ".join(LABEL[m] for m in models) + r"\\\midrule"]
        for p, row in piv.iterrows():
            best = row.min()
            cells = [(rf"\textbf{{{v:.2f}}}" if v == best else f"{v:.2f}") for v in row]
            lines.append(f"{p} & {int(n.get(p, 0))} & " + " & ".join(cells) + r"\\")
        mean, sd = piv.mean(), piv.std(ddof=1)
        lines.append(r"\midrule")
        lines.append(r"Mean$\pm$SD & & " + " & ".join(f"{a:.2f}$\\pm${b:.2f}" for a, b in zip(mean, sd)) + r"\\")
        lines += [r"\bottomrule", r"\end{tabular}\end{table*}"]
        (OUT / "tables" / f"patientwise_{metric}_{h}.tex").write_text("\n".join(lines) + "\n")

# Figure: per-patient RMSE across models (lines join the same patient)
horizons = sorted(res.horizon.unique())
ncols = min(3, len(horizons)); nrows = -(-len(horizons) // ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 3.0 * nrows), sharey=False)
axes = np.atleast_1d(axes).ravel()
for ax, h in zip(axes, horizons):
    piv = res[res.horizon == h].pivot(index="patient", columns="model", values="RMSE")
    models = [m for m in ORDER if m in piv.columns]
    piv = piv[models]
    x = np.arange(len(models))
    for _, row in piv.iterrows():
        ax.plot(x, row.values, color="0.75", lw=0.7, zorder=1)
    ax.scatter(np.repeat(x[None, :], len(piv), 0).ravel(), piv.values.ravel(), s=8, color="0.4", zorder=2)
    ax.plot(x, piv.mean().values, "k_", ms=14, mew=2, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[m] for m in models], rotation=60, ha="right", fontsize=7)
    ax.set_title(f"{h}-min horizon", fontsize=9)
    ax.set_ylabel("RMSE (mg/dL)", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
for ax in axes[len(horizons):]:
    ax.axis("off")
fig.tight_layout()
fig.savefig(OUT / "figures" / "rmse_by_patient.pdf")
print("Wrote tables to", OUT / "tables", "and figure to", OUT / "figures" / "rmse_by_patient.pdf")
