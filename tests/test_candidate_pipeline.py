"""Synthetic tests for the new GluNet candidate runner; no clinical data used."""
import numpy as np
import pandas as pd
import torch
from experiments.glunet.glunet_model_candidate import GluNetCandidate
from experiments.glunet.train_glunet_candidate_all_patients import (
    FEATURES, RAW_FEATURES, fit_preprocessor, apply_preprocessor, make_sequences,
)


def frame(n=100):
    ts = pd.date_range("2025-01-01", periods=n, freq="5min")
    g = 110 + 4*np.sin(np.arange(n)/12)
    df = pd.DataFrame({"Timestamp":ts, "Glucose":g, "Glucose_raw":g,
                       "carbs_last_60min":np.zeros(n), "bolus_last_60min":np.zeros(n)})
    mins = df.Timestamp.dt.hour*60 + df.Timestamp.dt.minute
    df["time_of_day_sin"] = np.sin(2*np.pi*mins/1440)
    df["time_of_day_cos"] = np.cos(2*np.pi*mins/1440)
    return df


def test_preprocessor_uses_training_only_stats_and_keeps_raw_target():
    tr = frame(60)
    va = frame(40); va["Glucose"] += 500; va["Glucose_raw"] += 500
    prep = fit_preprocessor(tr)
    transformed = apply_preprocessor(va, prep)
    assert transformed.Glucose_raw.iloc[0] > 500
    assert np.isclose(prep["means"]["Glucose"], tr["Glucose"].mean())  # exact training-only mean
    assert np.isfinite(transformed[FEATURES].to_numpy()).all()


def test_candidate_sequences_are_aligned_to_exact_horizon():
    df = frame()
    prep = fit_preprocessor(df.iloc[:60])
    df = apply_preprocessor(df, prep)
    X, y, meta = make_sequences(df, 30)
    assert len(X) == len(y) == len(meta) > 0
    assert X.shape[1:] == (16, len(FEATURES))
    assert (pd.to_datetime(meta.target_timestamp) - pd.to_datetime(meta.input_end)).eq(pd.Timedelta(minutes=30)).all()
    assert np.isfinite(X).all()


def test_candidate_model_outputs_256_logits():
    model = GluNetCandidate(input_features=len(FEATURES))
    out = model(torch.zeros(2,16,len(FEATURES)))
    assert out.shape == (2,256)
    assert torch.isfinite(out).all()
