"""Run an explicitly labelled GluNet architecture candidate on OhioT1DM.

This script does NOT claim exact reproduction of Li et al. It uses the
paper's 16-step window, 256 quantized delta classes, four dilation rates,
and a five-main-layer interpretation (input projection + four gated blocks).
It keeps the predefined training/testing partitions and creates chronological
train/validation subsets from the training partition BEFORE window creation.

Output is written to a new run directory; existing runs are never overwritten.
"""
from __future__ import annotations
import argparse, copy, json, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.glunet.glunet_model_candidate import GluNetCandidate
from experiments.glunet.glunet_data import quantize_delta
from clinical_metrics import full_metrics_row

PATIENTS = ["540", "544", "552", "567", "584", "596", "559", "563", "570", "575", "588", "591"]
RAW_FEATURES = ["Glucose", "carbs_last_60min", "bolus_last_60min"]
FEATURES = RAW_FEATURES + ["time_of_day_sin", "time_of_day_cos"]
LOOKBACK = 16
STEP_MINUTES = 5


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def load_partition(root: Path, patient: str, split: str) -> pd.DataFrame:
    path = root / f"{patient}_{split}_multimodal.csv"
    if not path.exists(): raise FileNotFoundError(f"Missing partition: {path}")
    df = pd.read_csv(path)
    required = ["Timestamp", *RAW_FEATURES]
    missing = [c for c in required if c not in df.columns]
    if missing: raise ValueError(f"{path} missing columns: {missing}")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    for c in RAW_FEATURES: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Timestamp", "Glucose"]).sort_values("Timestamp").reset_index(drop=True)
    df["Glucose_raw"] = df["Glucose"].astype(float)
    if df.Timestamp.duplicated().any(): raise ValueError(f"Duplicate timestamps in {path}")
    mins = df.Timestamp.dt.hour * 60 + df.Timestamp.dt.minute
    df["time_of_day_sin"] = np.sin(2 * np.pi * mins / 1440.0)
    df["time_of_day_cos"] = np.cos(2 * np.pi * mins / 1440.0)
    return df


def fit_preprocessor(train_df: pd.DataFrame):
    """Fit median imputation and standardization on the training segment only."""
    medians = train_df[FEATURES].median(numeric_only=True).fillna(0.0)
    imputed = train_df[FEATURES].fillna(medians)
    means = imputed.mean().fillna(0.0)
    stds = imputed.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    return {"medians": medians.to_dict(), "means": means.to_dict(), "stds": stds.to_dict()}


def apply_preprocessor(df: pd.DataFrame, prep: dict) -> pd.DataFrame:
    out = df.copy()
    med = pd.Series(prep["medians"]); means = pd.Series(prep["means"]); stds = pd.Series(prep["stds"])
    out[FEATURES] = (out[FEATURES].fillna(med) - means) / stds
    return out


def make_sequences(df: pd.DataFrame, horizon: int):
    if horizon <= 0 or horizon % STEP_MINUTES: raise ValueError("Horizon must be a positive multiple of 5")
    df = df.copy().reset_index(drop=True)
    delta_t = df.Timestamp.diff().dt.total_seconds()
    continuous = (delta_t - STEP_MINUTES * 60).abs() <= 30
    continuous.iloc[0] = False
    df["segment"] = (~continuous).cumsum()
    ts = pd.DatetimeIndex(df.Timestamp)
    lookup = {t: i for i, t in enumerate(ts)}
    vals = df[FEATURES].to_numpy(dtype=np.float32)
    glucose = df.Glucose_raw.to_numpy(dtype=np.float64)
    seg = df.segment.to_numpy()
    X, y, meta = [], [], []
    step = pd.Timedelta(minutes=horizon)
    for end in range(LOOKBACK - 1, len(df)):
        start = end - LOOKBACK + 1
        if not np.all(seg[start:end + 1] == seg[end]): continue
        target_idx = lookup.get(ts[end] + step)
        if target_idx is None or seg[target_idx] != seg[end]: continue
        x = vals[start:end + 1]
        if not np.isfinite(x).all(): continue
        delta = glucose[target_idx] - glucose[end]
        if not np.isfinite(delta): continue
        X.append(x); y.append(int(quantize_delta([delta])[0]))
        meta.append({"input_start": ts[start], "input_end": ts[end], "target_timestamp": ts[target_idx],
                     "current_glucose": glucose[end], "target_glucose": glucose[target_idx],
                     "delta_glucose": delta})
    return np.asarray(X, dtype=np.float32).reshape((-1, LOOKBACK, len(FEATURES))), np.asarray(y, dtype=np.int64), pd.DataFrame(meta)


def compute_class_weights(y, n_classes: int = 256, power: float = 0.5) -> np.ndarray:
    """Compute square-root inverse-frequency weights from training labels only.

    Unobserved classes receive zero weight. Observed-class weights are
    normalized so their sample-weighted mean is one.
    """
    labels = np.asarray(y, dtype=np.int64)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("y must be a non-empty one-dimensional label array")
    if np.any((labels < 0) | (labels >= n_classes)):
        raise ValueError(f"labels must be in [0, {n_classes - 1}]")
    if power <= 0:
        raise ValueError("power must be positive")
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    weights = np.zeros(n_classes, dtype=np.float64)
    observed = counts > 0
    weights[observed] = counts[observed] ** (-power)
    weights /= np.sum(counts * weights) / counts.sum()
    return weights.astype(np.float32)


def loaders(X, y, batch, shuffle, device):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0, pin_memory=device.type == "cuda")


def train_one(Xtr, ytr, Xv, yv, args, device, run_dir, patient, horizon):
    model = GluNetCandidate(input_features=len(FEATURES)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    validation_criterion = nn.CrossEntropyLoss()
    class_weights = None
    if args.loss_mode == "sqrt_inverse_frequency":
        # Weights are fitted from ytr only; validation/test labels never enter.
        class_weights = compute_class_weights(ytr, power=args.class_weight_power)
        train_criterion = nn.CrossEntropyLoss(
            weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device)
        )
    else:
        train_criterion = nn.CrossEntropyLoss()
    tr_loader = loaders(Xtr, ytr, args.batch_size, True, device)
    va_loader = loaders(Xv, yv, args.batch_size, False, device)
    best_loss, best_state, best_epoch, patience = float("inf"), None, 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); loss_sum = count = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True); logits = model(xb); loss = train_criterion(logits, yb)
            loss.backward(); opt.step(); loss_sum += loss.item() * len(yb); count += len(yb)
        train_loss = loss_sum / max(count, 1)
        model.eval(); v_sum = v_count = 0
        val_prob_batches, val_target_batches = [], []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                l = validation_criterion(logits, yb)
                v_sum += l.item() * len(yb); v_count += len(yb)
                val_prob_batches.append(torch.softmax(logits, dim=-1).cpu().numpy())
                val_target_batches.append(yb.cpu().numpy())
        val_loss = v_sum / max(v_count, 1)

        # Diagnostic only: does not alter the objective, gradients, or checkpoint selection.
        val_probs = np.concatenate(val_prob_batches, axis=0)
        val_true = np.concatenate(val_target_batches, axis=0)
        val_pred = val_probs.argmax(axis=1)
        val_pred_counts = np.bincount(val_pred, minlength=256)
        val_true_endpoint_share = 100.0 * np.mean((val_true == 0) | (val_true == 255))
        val_pred_endpoint_share = 100.0 * np.mean((val_pred == 0) | (val_pred == 255))
        val_pred_mode_share = 100.0 * val_pred_counts.max() / max(len(val_pred), 1)
        val_entropy = -np.sum(
            val_probs * np.log(np.maximum(val_probs, 1e-12)), axis=1
        )
        val_max_probability = val_probs.max(axis=1)

        history.append({
            "epoch": epoch,
            "train_objective_loss": train_loss,
            "validation_ce": val_loss,
            "val_true_unique_classes": int(np.unique(val_true).size),
            "val_pred_unique_classes": int(np.unique(val_pred).size),
            "val_true_endpoint_share_pct": float(val_true_endpoint_share),
            "val_pred_endpoint_share_pct": float(val_pred_endpoint_share),
            "val_pred_mode_share_pct": float(val_pred_mode_share),
            "val_mean_probability_entropy": float(val_entropy.mean()),
            "val_mean_max_probability": float(val_max_probability.mean()),
        })
        if val_loss < best_loss:
            best_loss, best_state, best_epoch, patience = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else: patience += 1
        print(
            f"{patient} h{horizon} epoch {epoch}/{args.epochs}: "
            f"train CE={train_loss:.4f}; val CE={val_loss:.4f}; "
            f"val classes true/pred={np.unique(val_true).size}/{np.unique(val_pred).size}; "
            f"pred endpoints={val_pred_endpoint_share:.1f}%; "
            f"largest predicted class={val_pred_mode_share:.1f}%",
            flush=True,
        )
        if patience >= args.patience: break
    if best_state is None: raise RuntimeError("No best checkpoint was produced")
    model.load_state_dict(best_state)
    out = run_dir / f"patient{patient}_{horizon}min"
    out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    torch.save({"state_dict": best_state, "patient": patient, "horizon_minutes": horizon,
                "feature_names": FEATURES, "lookback": LOOKBACK, "channels": [32,32,64,64],
                "dilations": [1,2,4,8], "best_epoch": best_epoch, "best_validation_ce": best_loss,
                "preprocessor_fit_on_training_segment_only": True,
                "loss_mode": args.loss_mode,
                "class_weight_power": args.class_weight_power if class_weights is not None else None,
                "class_weights": class_weights.tolist() if class_weights is not None else None}, out / "best.pt")
    return model, out


def infer(model, X, device, batch_size):
    model.eval(); probs=[]
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i+batch_size]).to(device)
            probs.append(torch.softmax(model(xb), dim=-1).cpu().numpy())
    return np.concatenate(probs, axis=0) if probs else np.empty((0,256), dtype=np.float32)


def metric_row(y_true, y_pred, model_name, patient, horizon):
    err = np.asarray(y_pred) - np.asarray(y_true)
    row = {"patient": patient, "horizon_minutes": horizon, "model": model_name,
           "n_test": len(y_true), "MAE": float(np.mean(np.abs(err))),
           "RMSE": float(np.sqrt(np.mean(err**2))),
           "mean_signed_error": float(np.mean(err))}
    row.update(full_metrics_row(np.asarray(y_true), np.asarray(y_pred), sample_interval_min=5))
    return row


def run(args):
    set_seed(args.seed)
    root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / pd.Timestamp.now().strftime("candidate_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=False, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    config = vars(args).copy(); config.update({"data_root": str(root), "output_dir": str(run_dir), "device": str(device),
        "features": FEATURES, "lookback": LOOKBACK, "patients": args.patients, "horizons": args.horizons,
        "status": "candidate implementation; not claimed as exact paper reproduction"})
    (run_dir / "experiment_config.json").write_text(json.dumps(config, indent=2))
    rows=[]; diagnostics=[]
    for patient in args.patients:
        train_all = load_partition(root, patient, "training")
        test_df = load_partition(root, patient, "testing")
        cut = int(len(train_all) * (1.0 - args.validation_fraction))
        if cut < LOOKBACK + 1 or len(train_all) - cut < LOOKBACK + max(args.horizons)//STEP_MINUTES:
            raise ValueError(f"Patient {patient}: training partition too short for chronological split")
        train_df, val_df = train_all.iloc[:cut].copy(), train_all.iloc[cut:].copy()
        prep = fit_preprocessor(train_df)
        train_df, val_df, test_df = [apply_preprocessor(x, prep) for x in (train_df, val_df, test_df)]
        (run_dir / f"patient{patient}_preprocessing.json").write_text(json.dumps(prep, indent=2))
        for horizon in args.horizons:
            Xtr, ytr, mtr = make_sequences(train_df, horizon)
            Xv, yv, mv = make_sequences(val_df, horizon)
            Xte, yte, mte = make_sequences(test_df, horizon)
            if min(len(Xtr), len(Xv), len(Xte)) == 0:
                raise RuntimeError(f"Patient {patient}, horizon {horizon}: empty train/validation/test sequences; inspect missing data and split sizes")
            model, out = train_one(Xtr,ytr,Xv,yv,args,device,run_dir,patient,horizon)
            prob = infer(model, Xte, device, args.batch_size)
            pred_class = prob.argmax(axis=1)
            delta_values = np.arange(256, dtype=np.float64)/10.0 - 12.7
            delta_expected = prob @ delta_values
            current = mte.current_glucose.to_numpy(dtype=float)
            actual = mte.target_glucose.to_numpy(dtype=float)
            predictions = {
                "expected_class_value": current + delta_expected,
                "argmax_class_value": current + delta_values[pred_class],
                "persistence": current,
            }
            for decoder, forecast in predictions.items():
                row = metric_row(actual, forecast, decoder, patient, horizon)
                row["best_epoch"] = int(torch.load(out / "best.pt", map_location="cpu", weights_only=False)["best_epoch"])
                rows.append(row)
            pred_df = mte.copy()
            pred_df["predicted_class_argmax_zero_based"] = pred_class
            pred_df["predicted_class_unique_count"] = int(np.unique(pred_class).size)
            pred_df["predicted_class_entropy_mean"] = float(np.mean(-np.sum(prob * np.log(np.maximum(prob, 1e-12)), axis=1)))
            pred_df["probability_max_mean"] = float(prob.max(axis=1).mean())
            pred_df["prediction_expected_class_value"] = predictions["expected_class_value"]
            pred_df["prediction_argmax_class_value"] = predictions["argmax_class_value"]
            pred_df["prediction_persistence"] = predictions["persistence"]
            pred_df.to_csv(out / "test_predictions_and_diagnostics.csv", index=False)
            np.savez_compressed(out / "test_probabilities.npz", probabilities=prob, predicted_classes=pred_class, true_classes=yte)
            diagnostics.append({"patient":patient,"horizon_minutes":horizon,"n_train_windows":len(Xtr),"n_validation_windows":len(Xv),"n_test_windows":len(Xte),
                "train_unique_classes":int(np.unique(ytr).size),"validation_unique_classes":int(np.unique(yv).size),"test_unique_classes":int(np.unique(yte).size),
                "predicted_unique_classes":int(np.unique(pred_class).size),"predicted_class_min":int(pred_class.min()),"predicted_class_max":int(pred_class.max()),
                "mean_probability_entropy":float(np.mean(-np.sum(prob*np.log(np.maximum(prob,1e-12)),axis=1))),
                "mean_max_probability":float(prob.max(axis=1).mean())})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(run_dir / "patient_metrics_all_decoders.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(run_dir / "class_collapse_diagnostics.csv", index=False)
    metric_cols=["MAE","RMSE","MARD","TimeLag_min","CEGA_A_%","CEGA_B_%","CEGA_C_%","CEGA_D_%","CEGA_E_%"]
    summary=metrics.groupby(["horizon_minutes","model"])[metric_cols].agg(["mean","std"]).reset_index()
    summary.to_csv(run_dir / "mean_sd_summary.csv", index=False)
    print(f"Completed candidate runs. Results: {run_dir}")
    print("Important: inspect class_collapse_diagnostics.csv and validation/test class coverage before interpreting metrics.")


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="~/Desktop/OhioT1DM_data_backup")
    p.add_argument("--output-dir", default="experiments/glunet/runs/paper_candidate_v1")
    p.add_argument("--patients", nargs="+", default=PATIENTS)
    p.add_argument("--horizons", nargs="+", type=int, default=[30,60])
    p.add_argument("--validation-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--loss-mode", choices=["standard", "sqrt_inverse_frequency"], default="standard", help="Optional inverse-frequency weighting fitted on training labels only")
    p.add_argument("--class-weight-power", type=float, default=0.5, help="Exponent alpha for class weights n_c**(-alpha); used only with sqrt_inverse_frequency (default: 0.5)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--cpu", action="store_true")
    args=p.parse_args()
    if not 0.05 <= args.validation_fraction <= 0.4: p.error("validation-fraction must be between 0.05 and 0.4")
    if args.class_weight_power <= 0: p.error("class-weight-power must be positive")
    if any(h <= 0 or h % 5 for h in args.horizons): p.error("horizons must be positive multiples of 5")
    if any(p not in PATIENTS for p in args.patients): p.error(f"patients must be selected from {PATIENTS}")
    return args

if __name__ == "__main__": run(parse_args())
