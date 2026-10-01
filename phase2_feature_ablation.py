#!/usr/bin/env python3
"""
Phase-2 controlled feature ablation.

Uses one fixed GRU model/protocol across all feature sets.
Common feature sets are evaluated on all 12 OhioT1DM patients.
Cohort-specific richer sets are evaluated only where the required
signals are available.

Run from the repository root:
    python phase2_feature_ablation.py
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from phase2_sequence_utils import build_sequences


PATIENTS_2018 = [559, 563, 570, 575, 588, 591]
PATIENTS_2020 = [540, 544, 552, 567, 584, 596]
HORIZONS = [15, 30, 60, 90, 120]
LOOKBACK = 120
STEP_MIN = 5

# Only numeric model inputs; observed flags are intentionally excluded.
COMMON = ["glucose", "carbs_last_60min", "bolus_last_60min", "basal_rate"]
FEATSETS_2018 = {
    "glucose": ["glucose"],
    "glucose_carbs": ["glucose", "carbs_last_60min"],
    "glucose_insulin": ["glucose", "bolus_last_60min", "basal_rate"],
    "glucose_carbs_insulin": COMMON,
    "glucose_hr": ["glucose", "heart_rate_mean_5m"],
    "glucose_hr_steps_carbs_sleep": [
        "glucose", "heart_rate_mean_5m", "steps_mean_5m",
        "carbs_last_60min", "is_sleeping",
    ],
    "full_2018": [
        "glucose", "heart_rate_mean_5m", "steps_mean_5m",
        "carbs_last_60min", "is_sleeping",
        "bolus_last_60min", "basal_rate",
    ],
}
FEATSETS_2020 = {
    "glucose": ["glucose"],
    "glucose_carbs": ["glucose", "carbs_last_60min"],
    "glucose_insulin": ["glucose", "bolus_last_60min", "basal_rate"],
    "glucose_carbs_insulin": COMMON,
    "glucose_acceleration": ["glucose", "accel_mean_5m"],
    "full_2020": [
        "glucose", "carbs_last_60min", "bolus_last_60min", "basal_rate",
        "accel_mean_5m", "is_sleeping",
    ],
}

# Fixed architecture for feature isolation.
HIDDEN = 96
LAYERS = 2
DROPOUT = 0.10
BATCH = 128
EPOCHS = 15
PATIENCE = 4
LR = 3e-4
WEIGHT_DECAY = 1e-4
SEED = 42


def seed_all(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FixedGRU(nn.Module):
    def __init__(self, input_dim: int, horizons: List[int]):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, HIDDEN, num_layers=LAYERS,
            batch_first=True, dropout=DROPOUT if LAYERS > 1 else 0.0
        )
        self.horizons = horizons
        self.heads = nn.ModuleDict({
            str(h): nn.Sequential(
                nn.Linear(HIDDEN, 48),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
                nn.Linear(48, 1),
            ) for h in horizons
        })

    def forward(self, x):
        z, _ = self.gru(x)
        z = z[:, -1]
        return torch.cat([self.heads[str(h)](z) for h in self.horizons], dim=1)


def impute_scale(train_df, test_df, features):
    med = train_df[features].replace([np.inf, -np.inf], np.nan).median()
    tr = train_df[features].replace([np.inf, -np.inf], np.nan).fillna(med)
    te = test_df[features].replace([np.inf, -np.inf], np.nan).fillna(med)
    scaler = StandardScaler()
    tr2 = scaler.fit_transform(tr)
    te2 = scaler.transform(te)
    return tr2, te2, scaler


def train_one(Xtr, ytr, Xv, yv, input_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FixedGRU(input_dim, HORIZONS).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.SmoothL1Loss()

    ds = TensorDataset(
        torch.tensor(Xtr, dtype=torch.float32),
        torch.tensor(ytr, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True)

    best = float("inf")
    best_state = None
    bad = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(Xv, dtype=torch.float32, device=device))
            val = float(loss_fn(pred, torch.tensor(yv, dtype=torch.float32, device=device)))
        if val < best:
            best = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, device, epoch


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mard(a, b):
    den = np.maximum(np.abs(a), 1e-6)
    return float(100 * np.mean(np.abs(a - b) / den))


def time_lag_minutes(y_true, y_pred, step=5, max_lag=120):
    """Choose lag minimizing MAE. Positive means prediction trails truth."""
    best_lag, best_err = 0, float("inf")
    for lag in range(-max_lag, max_lag + 1, step):
        if lag > 0:
            t, p = y_true[:-lag], y_pred[lag:]
        elif lag < 0:
            k = -lag
            t, p = y_true[k:], y_pred[:-k]
        else:
            t, p = y_true, y_pred
        if len(t) < 2:
            continue
        e = np.mean(np.abs(t - p))
        if e < best_err:
            best_err, best_lag = e, lag
    return int(best_lag)


def ceg_zone(y, p):
    """Approximate Clarke Error Grid zones using standard clinical boundaries."""
    zones = []
    for g, q in zip(np.asarray(y), np.asarray(p)):
        if g <= 70:
            if q <= 70:
                z = "A"
            elif q <= 180:
                z = "B"
            elif q <= 240:
                z = "E"
            else:
                z = "E"
        else:
            if q <= 70:
                z = "C" if g > 180 else "B"
            elif abs(q - g) <= 0.20 * g:
                z = "A"
            elif (q <= 180 and q >= 70) or abs(q - g) <= 0.30 * g:
                z = "B"
            elif (g > 180 and q < 70) or (g < 70 and q > 180):
                z = "D"
            else:
                z = "C"
        zones.append(z)
    return zones


def evaluate_patient(model, device, X, y):
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()
    rows = []
    for j, h in enumerate(HORIZONS):
        yt, yp = y[:, j], pred[:, j]
        zones = ceg_zone(yt, yp)
        counts = {z: zones.count(z) for z in "ABCDE"}
        row = {
            "horizon_min": h,
            "mae_mgdl": mae(yt, yp),
            "rmse_mgdl": rmse(yt, yp),
            "mard_pct": mard(yt, yp),
            "time_lag_min": time_lag_minutes(yt, yp),
        }
        n = len(zones)
        for z in "ABCDE":
            row[f"clarke_{z}_pct"] = 100 * counts[z] / max(n, 1)
        rows.append(row)
    return rows


def load_df(path):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_localize(None)
    return df.sort_values("timestamp").reset_index(drop=True)


def run_case(root, cohort, pid, split, feature_name, features):
    path = root / "data" / "phase2" / ("ohio2018" if cohort == "2018" else "ohio2020") / split / f"{pid}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = load_df(path)

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing features {missing}")

    # Drop rows where required inputs are unavailable only after train/test separation.
    # Median imputation handles ordinary missing values.
    Xscaled, _, _ = impute_scale(df, df, features)

    # Build target sequences from original dataframe so gap boundaries/timestamps are preserved.
    Xseq, yseq, meta = build_sequences(
        df,
        feature_cols=features,
        target_col="glucose",
        lookback_steps=LOOKBACK,
        horizon_steps=HORIZONS,
        step_min=STEP_MIN,
    )

    return Xseq, yseq, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=".")
    ap.add_argument("--output", default="output/phase2_feature_ablation")
    args = ap.parse_args()

    seed_all()
    root = Path(args.data_root).resolve()
    out = root / args.output
    out.mkdir(parents=True, exist_ok=True)

    # This runner intentionally requires the current gap-safe sequence utility.
    # It performs imputation/scaling separately inside each train/test pair below.
    all_rows = []

    cohort_specs = [("2018", PATIENTS_2018, FEATSETS_2018),
                    ("2020", PATIENTS_2020, FEATSETS_2020)]

    for cohort, patients, featsets in cohort_specs:
        for fname, features in featsets.items():
            print(f"\n=== {cohort} | {fname} | {features} ===")
            for pid in patients:
                tr_path = root / "data" / "phase2" / f"ohio{cohort}" / "train" / f"{pid}.csv"
                te_path = root / "data" / "phase2" / f"ohio{cohort}" / "test" / f"{pid}.csv"
                tr = load_df(tr_path)
                te = load_df(te_path)

                # Only run feature sets whose signals are genuinely represented.
                if "is_sleeping" in features:
                    sleep_available = tr["is_sleeping"].notna().any() and te["is_sleeping"].notna().any()
                    # Missing sleep files are represented as all-zero only if the preprocessing
                    # script explicitly created the column; skip unavailable sleep experiments.
                    if not sleep_available:
                        print(f"SKIP {pid}: sleep unavailable")
                        continue

                missing = [c for c in features if c not in tr.columns or c not in te.columns]
                if missing:
                    print(f"SKIP {pid}: missing {missing}")
                    continue

                # Fit preprocessing only on train.
                med = tr[features].replace([np.inf, -np.inf], np.nan).median()
                tr2 = tr.copy()
                te2 = te.copy()
                tr2[features] = tr2[features].replace([np.inf, -np.inf], np.nan).fillna(med)
                te2[features] = te2[features].replace([np.inf, -np.inf], np.nan).fillna(med)
                scaler = StandardScaler()
                tr2[features] = scaler.fit_transform(tr2[features])
                te2[features] = scaler.transform(te2[features])

                X, y, meta = build_sequences(
                    tr2,
                    feature_cols=features,
                    target_col="glucose",
                    lookback_steps=LOOKBACK,
                    horizon_steps=HORIZONS,
                    step_min=STEP_MIN,
                )
                Xt, yt, metat = build_sequences(
                    te2,
                    feature_cols=features,
                    target_col="glucose",
                    lookback_steps=LOOKBACK,
                    horizon_steps=HORIZONS,
                    step_min=STEP_MIN,
                )

                if len(X) < 20 or len(Xt) < 5:
                    print(f"SKIP {pid}: insufficient sequences train={len(X)} test={len(Xt)}")
                    continue

                nval = max(1, int(round(len(X) * 0.15)))
                Xtr, Xv = X[:-nval], X[-nval:]
                ytr, yv = y[:-nval], y[-nval:]

                model, device, epochs = train_one(Xtr, ytr, Xv, yv, len(features))
                metrics = evaluate_patient(model, device, Xt, yt)

                for r in metrics:
                    r.update({
                        "cohort": cohort,
                        "patient": pid,
                        "feature_set": fname,
                        "n_features": len(features),
                        "features": "|".join(features),
                        "n_train_sequences": len(X),
                        "n_test_sequences": len(Xt),
                        "epochs": epochs,
                    })
                    all_rows.append(r)

                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "features": features,
                        "horizons": HORIZONS,
                        "lookback": LOOKBACK,
                        "seed": SEED,
                    },
                    out / f"{cohort}_{pid}_{fname}.pt",
                )
                print(f"done patient={pid}, test_sequences={len(Xt)}")

    if not all_rows:
        raise RuntimeError("No experiments completed.")

    df = pd.DataFrame(all_rows)
    df.to_csv(out / "patient_horizon_results.csv", index=False)

    summary = (
        df.groupby(["cohort", "feature_set", "horizon_min"])
        .agg(
            mae_mean=("mae_mgdl", "mean"), mae_sd=("mae_mgdl", "std"),
            rmse_mean=("rmse_mgdl", "mean"), rmse_sd=("rmse_mgdl", "std"),
            mard_mean=("mard_pct", "mean"), mard_sd=("mard_pct", "std"),
            time_lag_mean=("time_lag_min", "mean"),
            time_lag_sd=("time_lag_min", "std"),
            clarke_A_mean=("clarke_A_pct", "mean"),
            clarke_B_mean=("clarke_B_pct", "mean"),
            clarke_C_mean=("clarke_C_pct", "mean"),
            clarke_D_mean=("clarke_D_pct", "mean"),
            clarke_E_mean=("clarke_E_pct", "mean"),
            n_patients=("patient", "nunique"),
        )
        .reset_index()
    )
    summary.to_csv(out / "feature_ablation_summary.csv", index=False)

    # Explicit all-12 common feature analysis.
    common = df[
        df["feature_set"].isin(
            ["glucose", "glucose_carbs", "glucose_insulin", "glucose_carbs_insulin"]
        )
    ].copy()
    common.to_csv(out / "common_all12_results.csv", index=False)

    with open(out / "config.json", "w") as f:
        json.dump({
            "seed": SEED,
            "patients_2018": PATIENTS_2018,
            "patients_2020": PATIENTS_2020,
            "horizons": HORIZONS,
            "lookback_min": LOOKBACK,
            "architecture": "2-layer GRU fixed across feature sets",
            "hidden": HIDDEN,
            "feature_sets_2018": FEATSETS_2018,
            "feature_sets_2020": FEATSETS_2020,
        }, f, indent=2)

    print("\nDONE")
    print(f"Results: {out / 'feature_ablation_summary.csv'}")
    print(f"All-12 common: {out / 'common_all12_results.csv'}")


if __name__ == "__main__":
    main()
