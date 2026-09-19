"""
clinical_metrics.py
Clinical + statistical evaluation metrics, built to slot into your existing
sklearn/pandas stack (no new dependencies). Adds what full_results_by_patient.csv
is currently missing: MARD, time lag, and Clarke Error Grid zone distribution.

Usage alongside your existing RMSE/MAE:
    from clinical_metrics import mard, time_lag_minutes, clarke_error_grid_distribution
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List


def mard(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """Mean Absolute Relative Difference, as a percentage."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred) / np.maximum(y_true, eps)) * 100.0)


def time_lag_minutes(y_true: np.ndarray, y_pred: np.ndarray, sample_interval_min: int = 5,
                      max_lag_samples: int = 12) -> float:
    """
    Cross-correlation based lag estimate (minutes). Positive lag = prediction
    trails the true trace, which is the typical/expected direction for CGM
    forecasting models and a known clinical concern to quantify.
    """
    y_true = np.asarray(y_true, dtype=float) - np.mean(y_true)
    y_pred = np.asarray(y_pred, dtype=float) - np.mean(y_pred)

    best_lag, best_corr = 0, -np.inf
    for lag in range(0, max_lag_samples + 1):
        a, b = (y_true, y_pred) if lag == 0 else (y_true[lag:], y_pred[:-lag])
        if len(a) < 2:
            continue
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        corr = np.dot(a, b) / denom if denom > 0 else 0.0
        if corr > best_corr:
            best_corr, best_lag = corr, lag
    return float(best_lag * sample_interval_min)


def clarke_error_grid_zone(ref: float, pred: float) -> str:
    """Single-point Clarke Error Grid zone classification (Clarke et al., 1987)."""
    if ref <= 0 or pred < 0:
        return "E"
    if (pred <= 70 and ref <= 70) or (0.8 * ref <= pred <= 1.2 * ref):
        return "A"
    if (ref >= 180 and pred <= 70) or (ref <= 70 and pred >= 180):
        return "E"
    if (70 <= ref <= 290 and pred >= ref + 110) or \
       (130 <= ref <= 180 and pred <= (7.0 / 5.0) * ref - 182):
        return "C"
    if (ref >= 240 and 70 <= pred <= 180) or (ref <= 70 and 70 <= pred <= 180):
        return "D"
    return "B"


def clarke_error_grid_distribution(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Percentage of points in each CEGA zone (A-E), summing to 100."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    zones = [clarke_error_grid_zone(r, p) for r, p in zip(y_true, y_pred)]
    n = len(zones)
    return {z: (zones.count(z) / n * 100.0 if n else 0.0) for z in ["A", "B", "C", "D", "E"]}


def full_metrics_row(y_true: np.ndarray, y_pred: np.ndarray, sample_interval_min: int = 5) -> Dict[str, float]:
    """
    Convenience function: returns a flat dict you can append directly to a
    results row alongside your existing RMSE/MAE/R2 computation, e.g.:

        row.update(full_metrics_row(actual, preds))
    """
    cega = clarke_error_grid_distribution(y_true, y_pred)
    out = {
        "MARD": round(mard(y_true, y_pred), 2),
        "TimeLag_min": round(time_lag_minutes(y_true, y_pred, sample_interval_min), 2),
    }
    out.update({f"CEGA_{z}_%": round(v, 2) for z, v in cega.items()})
    return out


def summary_mean_sd(full_df: pd.DataFrame, group_cols: List[str], metric_cols: List[str]) -> pd.DataFrame:
    """
    Mirrors the mean+/-std summary pattern already used in
    multimodel_compare_all.py, but for the fuller metric set and formatted
    as 'mean ± sd' strings for direct reporting (matches the table format
    requested: Model | RMSE | MAE | MARD | Time Lag | Clarke Error Grid).
    """
    def fmt(s):
        return f"{s.mean():.2f} \u00b1 {(s.std(ddof=1) if len(s) > 1 else 0.0):.2f}"

    rows = []
    for keys, g in full_df.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        row["n_patients"] = len(g)
        for col in metric_cols:
            row[col] = fmt(g[col])
        rows.append(row)
    return pd.DataFrame(rows)
