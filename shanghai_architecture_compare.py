"""
shanghai_architecture_compare.py
Architecture comparison on the Shanghai T1DM cohort (12 patients), using the
SAME protocol as multimodel_architecture_compare.py on OhioT1DM -- same
architectures, same horizons in minutes, same clinical metric suite -- so
the two datasets' results are directly comparable (this is the cross-dataset
generalizability check from the original brief).

Key protocol difference from Ohio, driven by the data itself (not a choice):
    - Shanghai is 15-min sampled (vs Ohio's 5-min), so WINDOW and HORIZONS
      are expressed in STEPS differently, but represent the SAME wall-clock
      lookback (60 min) and horizons (15/30/60 min).
    - Feature set is Glucose-only. Shanghai's other columns (insulin doses,
      diet notes) are sparse event markers, not the dense per-timestep
      channels Ohio has -- an insulin-inclusive feature set for Shanghai
      would need its own extraction logic (bolus-last-60min, active basal
      rate, analogous to what was built for Ohio) as a follow-up. Glucose-
      only is also directly comparable to Ohio's "A: Glucose only" baseline,
      which makes this the right first comparison to run.
    - T2DM (100 patients) is NOT included here -- T1DM is the disease-
      matched comparison to OhioT1DM. Run a separate pass on T2DM patients
      afterwards if a secondary generalizability check is wanted.

Usage:
    python shanghai_preprocessing.py --raw_dir /path/to/Shanghai_DATASET --out_dir ./data_shanghai_converted
    python shanghai_architecture_compare.py --data_dir ./data_shanghai_converted
"""

import os
import argparse
os.environ["TF_METAL_DEVICE_PLACEMENT"] = "0"
import tensorflow as tf
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential, Model
from keras.layers import (
    LSTM, GRU, Bidirectional, Dense, Dropout, Input, Conv1D,
    MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Add,
)
from keras.callbacks import EarlyStopping

from shanghai_preprocessing import (
    load_shanghai_file, clean_and_resample, segment_contiguous,
    parse_patient_visit, SAMPLE_INTERVAL_MIN,
)
from clinical_metrics import full_metrics_row, summary_mean_sd

WINDOW = 4                                          # 60 min lookback @ 15-min sampling
HORIZONS = {"15min": 1, "30min": 2, "60min": 4}      # steps @ 15-min sampling
EPOCHS = 10
BATCH_SIZE = 64

os.makedirs("output", exist_ok=True)


def build_lstm(input_dim):
    m = Sequential([LSTM(64, input_shape=(WINDOW, input_dim)), Dropout(0.2),
                     Dense(32, activation="relu"), Dense(1)])
    m.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return m


def build_gru(input_dim):
    m = Sequential([GRU(64, input_shape=(WINDOW, input_dim)), Dropout(0.2),
                     Dense(32, activation="relu"), Dense(1)])
    m.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return m


def build_bilstm(input_dim):
    m = Sequential([Bidirectional(LSTM(64), input_shape=(WINDOW, input_dim)), Dropout(0.2),
                     Dense(32, activation="relu"), Dense(1)])
    m.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return m


def build_tcn(input_dim):
    inp = Input(shape=(WINDOW, input_dim))
    x = inp
    for dilation in (1, 2):   # fewer dilation levels than Ohio -- WINDOW=4 is short
        x = Conv1D(32, kernel_size=2, padding="causal", dilation_rate=dilation, activation="relu")(x)
        x = Dropout(0.2)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation="relu")(x)
    out = Dense(1)(x)
    m = Model(inp, out)
    m.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return m


def build_transformer(input_dim, d_model=32, num_heads=4, ff_dim=64):
    inp = Input(shape=(WINDOW, input_dim))
    x = Dense(d_model)(inp)
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(x, x)
    x = Add()([x, attn])
    x = LayerNormalization()(x)
    ff = Dense(ff_dim, activation="relu")(x)
    ff = Dense(d_model)(ff)
    x = Add()([x, ff])
    x = LayerNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation="relu")(x)
    out = Dense(1)(x)
    m = Model(inp, out)
    m.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return m


ARCHITECTURES = {
    "LSTM": build_lstm, "GRU": build_gru, "BiLSTM": build_bilstm,
    "TCN": build_tcn, "Transformer": build_transformer,
}


def load_t1dm_patients(data_dir):
    """One row per T1DM patient, using their earliest visit only (visit '0')
    for a clean single-recording-per-patient comparison, matching Ohio's
    one-file-per-patient structure. Patients with only later visits are
    included using their earliest available visit."""
    files_by_patient = {}
    for f in sorted(__import__("glob").glob(os.path.join(data_dir, "*.xlsx"))):
        dataset, patient_id, visit = parse_patient_visit(f)
        if dataset != "T1DM":
            continue
        files_by_patient.setdefault(patient_id, []).append((visit, f))

    patients = {}
    for patient_id, visits in files_by_patient.items():
        visits.sort(key=lambda v: v[0])
        patients[patient_id] = visits[0][1]   # earliest visit
    return patients


def process_one_patient(patient_id, path):
    df = load_shanghai_file(path)
    df = clean_and_resample(df)
    min_len = WINDOW + max(HORIZONS.values()) + 10
    segments = segment_contiguous(df, min_len)
    if not segments:
        print(f"Skipping patient {patient_id}: no usable contiguous segment.")
        return []

    # use the single longest contiguous segment, split chronologically 80/20 train/test
    segment = max(segments, key=len)
    split_idx = int(len(segment) * 0.8)
    train_vals = segment["glucose"].values[:split_idx]
    test_vals = segment["glucose"].values[split_idx:]

    mean, std = train_vals.mean(), train_vals.std()
    std = std if std > 1e-6 else 1.0

    rows = []
    for horizon_name, horizon_steps in HORIZONS.items():
        def make_windows(vals):
            X, y = [], []
            for i in range(len(vals) - WINDOW - horizon_steps + 1):
                X.append(vals[i:i + WINDOW])
                y.append(vals[i + WINDOW + horizon_steps - 1])
            return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

        train_scaled = (train_vals - mean) / std
        test_scaled = (test_vals - mean) / std
        X_train, y_train = make_windows(train_scaled)
        X_test, y_test = make_windows(test_scaled)
        X_train = X_train[..., None]
        X_test = X_test[..., None]

        if len(X_train) < 30 or len(X_test) < 10:
            print(f"Patient {patient_id} | {horizon_name}: not enough data, skipping.")
            continue

        for arch_name, build_fn in ARCHITECTURES.items():
            print(f"Patient {patient_id} | Horizon {horizon_name} | Architecture {arch_name}")
            model = build_fn(input_dim=1)
            model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                      validation_split=0.1,
                      callbacks=[EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
                      verbose=0)

            preds_scaled = model.predict(X_test, verbose=0).flatten()
            preds = preds_scaled * std + mean
            actual = y_test * std + mean

            row = {
                "Patient": patient_id, "Horizon": horizon_name, "Architecture": arch_name,
                "RMSE": round(float(np.sqrt(mean_squared_error(actual, preds))), 2),
                "MAE": round(float(mean_absolute_error(actual, preds)), 2),
            }
            row.update(full_metrics_row(actual, preds, sample_interval_min=SAMPLE_INTERVAL_MIN))
            rows.append(row)
            print(f"  RMSE={row['RMSE']:.2f}  MARD={row['MARD']:.2f}%")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Output dir from shanghai_preprocessing.py")
    args = parser.parse_args()

    patients = load_t1dm_patients(args.data_dir)
    print(f"Found {len(patients)} T1DM patients: {sorted(patients.keys())}")

    import concurrent.futures
    max_workers = max(1, (os.cpu_count() or 2) - 1)

    all_rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one_patient, pid, path): pid for pid, path in patients.items()}
        for future in concurrent.futures.as_completed(futures):
            all_rows.extend(future.result())

    full_df = pd.DataFrame(all_rows)
    if full_df.empty:
        raise SystemExit("No results produced -- check --data_dir points to converted .xlsx files.")

    full_df.to_csv("output/shanghai_t1dm_architecture_results.csv", index=False)
    print("\nSaved: output/shanghai_t1dm_architecture_results.csv")

    metric_cols = ["RMSE", "MAE", "MARD", "TimeLag_min", "CEGA_A_%", "CEGA_B_%", "CEGA_C_%", "CEGA_D_%", "CEGA_E_%"]
    summary = summary_mean_sd(full_df, group_cols=["Horizon", "Architecture"], metric_cols=metric_cols)
    summary.to_csv("output/shanghai_t1dm_architecture_summary.csv", index=False)
    print("Saved: output/shanghai_t1dm_architecture_summary.csv")
    print(summary.to_string(index=False))
