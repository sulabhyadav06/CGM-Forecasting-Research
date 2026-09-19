"""
multimodel_architecture_compare.py -- UPDATED

Changes from the previous version:
  1. Default feature set no longer includes HeartRate/Steps -- those are
     100% missing for the 6 2020 patients (different wristband, confirmed
     against the actual XML). The default now uses features available for
     ALL 12 patients: Glucose + carbs_last_60min + is_sleeping +
     bolus_last_60min + basal_rate (insulin replaces HR/Steps as the added
     physiological signal -- arguably more directly relevant to glucose
     dynamics anyway).
  2. A second, separate run (FEATURE_COLS_2018_RICH) adds HeartRate + Steps
     back in, restricted to the 6 2018 patients only, so you still get an
     apples-to-apples comparison of "does wearable data help" without
     silently training 2020 patients on placeholder values.
  3. Requires the v2 parse_xml.py (with insulin extraction) to have been run
     so that bolus_last_60min / basal_rate columns exist in the CSVs.

Outputs:
    output/architecture_results_by_patient.csv       (all 12 patients, insulin feature set)
    output/architecture_results_2018_wearable.csv     (6 2018 patients, +HR/Steps)
    output/architecture_summary_mean_sd.csv
    output/architecture_summary_2018_wearable_mean_sd.csv
"""

import os
# fix: tensorflow-metal's GPU backend is frequently SLOWER than CPU for
# small recurrent models (LSTM/GRU with few units, tiny batches) -- the
# overhead of dispatching each small batch to the GPU outweighs any
# parallelism benefit. Force CPU for this workload.
os.environ["TF_METAL_DEVICE_PLACEMENT"] = "0"
import tensorflow as tf
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass
print("Visible devices:", tf.config.get_visible_devices())

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

from clinical_metrics import full_metrics_row, summary_mean_sd

PATIENT_IDS_2018 = ["559", "563", "570", "575", "588", "591"]
PATIENT_IDS_2020 = ["540", "544", "552", "567", "584", "596"]
PATIENT_IDS = PATIENT_IDS_2018 + PATIENT_IDS_2020

HORIZONS = {"15min": 3, "30min": 6, "60min": 12}   # steps @ 5-min sampling
WINDOW = 12                                        # 60 min lookback
EPOCHS = 10
BATCH_SIZE = 128
SAMPLE_INTERVAL_MIN = 5

# Available for all 12 patients (insulin replaces HR/Steps as the added signal)
FEATURE_COLS_ALL12 = ["Glucose", "carbs_last_60min", "is_sleeping", "bolus_last_60min", "basal_rate"]
# Richer set, 2018 cohort only (HR/Steps unavailable for 2020 patients)
FEATURE_COLS_2018_RICH = ["Glucose", "HeartRate", "Steps", "carbs_last_60min", "is_sleeping",
                           "bolus_last_60min", "basal_rate"]

os.makedirs("output", exist_ok=True)


def create_sequences(X, y, window=WINDOW):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i: i + window])
        ys.append(y[i + window])
    return np.array(Xs), np.array(ys)


def build_lstm(input_dim):
    m = Sequential([
        LSTM(64, input_shape=(WINDOW, input_dim)),
        Dropout(0.2), Dense(32, activation="relu"), Dense(1),
    ])
    m.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return m


def build_gru(input_dim):
    m = Sequential([
        GRU(64, input_shape=(WINDOW, input_dim)),
        Dropout(0.2), Dense(32, activation="relu"), Dense(1),
    ])
    m.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return m


def build_bilstm(input_dim):
    m = Sequential([
        Bidirectional(LSTM(64), input_shape=(WINDOW, input_dim)),
        Dropout(0.2), Dense(32, activation="relu"), Dense(1),
    ])
    m.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return m


def build_tcn(input_dim):
    inp = Input(shape=(WINDOW, input_dim))
    x = inp
    for dilation in (1, 2, 4):
        x = Conv1D(32, kernel_size=3, padding="causal", dilation_rate=dilation, activation="relu")(x)
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
    "LSTM": build_lstm,
    "GRU": build_gru,
    "BiLSTM": build_bilstm,
    "TCN": build_tcn,
    "Transformer": build_transformer,
}


def process_one_patient(args):
    """
    Runs every horizon x architecture combination for ONE patient.
    Pulled out to a top-level function (rather than nested in
    run_comparison) so it can be pickled and run in a separate process --
    required for ProcessPoolExecutor, especially on macOS's 'spawn' start
    method.
    """
    patient_id, feature_cols, out_prefix = args
    rows = []

    train_path = f"data/{patient_id}_training_multimodal.csv"
    test_path = f"data/{patient_id}_testing_multimodal.csv"

    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print(f"Skipping patient {patient_id} - CSVs not found. Run parse_xml.py for this patient first.")
        return rows

    train_df = pd.read_csv(train_path, parse_dates=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    test_df = pd.read_csv(test_path, parse_dates=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    missing_cols = [c for c in feature_cols if c not in train_df.columns]
    if missing_cols:
        print(f"Skipping patient {patient_id}: missing columns {missing_cols} "
              f"(re-run parse_xml.py with insulin extraction to generate these).")
        return rows

    all_nan_cols = [c for c in feature_cols if train_df[c].isna().all() or test_df[c].isna().all()]
    if all_nan_cols:
        print(f"Skipping patient {patient_id}: columns {all_nan_cols} are entirely missing "
              f"for this patient (likely an MDI patient with no pump data).")
        return rows

    for horizon_name, horizon_steps in HORIZONS.items():
        t_train, t_test = train_df.copy(), test_df.copy()
        t_train["target"] = t_train["Glucose"].shift(-horizon_steps)
        t_test["target"] = t_test["Glucose"].shift(-horizon_steps)
        t_train = t_train.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)
        t_test = t_test.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)

        if len(t_train) == 0 or len(t_test) == 0:
            print(f"Patient {patient_id} | {horizon_name}: 0 rows after dropna, skipping.")
            continue

        scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
        X_train_raw = scaler_X.fit_transform(t_train[feature_cols].values)
        y_train_raw = scaler_y.fit_transform(t_train[["target"]].values)
        X_test_raw = scaler_X.transform(t_test[feature_cols].values)
        y_test_raw = scaler_y.transform(t_test[["target"]].values)

        X_train, y_train = create_sequences(X_train_raw, y_train_raw)
        X_test, y_test = create_sequences(X_test_raw, y_test_raw)

        if len(X_train) < 50 or len(X_test) < 20:
            print(f"Patient {patient_id} | {horizon_name}: not enough data, skipping.")
            continue

        for arch_name, build_fn in ARCHITECTURES.items():
            print(f"[{out_prefix}] Patient {patient_id} | Horizon {horizon_name} | Architecture {arch_name}")
            model = build_fn(input_dim=len(feature_cols))
            model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                      validation_split=0.1,
                      callbacks=[EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
                      verbose=0)

            preds_scaled = model.predict(X_test, verbose=0)
            preds = scaler_y.inverse_transform(preds_scaled).flatten()
            actual = scaler_y.inverse_transform(y_test).flatten()

            row = {
                "Patient": patient_id,
                "Horizon": horizon_name,
                "Architecture": arch_name,
                "RMSE": round(float(np.sqrt(mean_squared_error(actual, preds))), 2),
                "MAE": round(float(mean_absolute_error(actual, preds)), 2),
            }
            row.update(full_metrics_row(actual, preds, sample_interval_min=SAMPLE_INTERVAL_MIN))
            rows.append(row)
            print(f"  Patient {patient_id} {horizon_name} {arch_name}: "
                  f"RMSE={row['RMSE']:.2f}  MARD={row['MARD']:.2f}%")

    return rows


def run_comparison(patient_ids, feature_cols, out_prefix, max_workers=None):
    """
    Runs process_one_patient for each patient IN PARALLEL across separate
    processes (each patient's work is fully independent, so this is a safe,
    high-value parallelization). max_workers defaults to
    os.cpu_count() - 1 (leaves one core free for the OS/other work).
    """
    import concurrent.futures

    max_workers = max_workers or max(1, (os.cpu_count() or 2) - 1)
    print(f"Running {len(patient_ids)} patients across {max_workers} parallel processes...")

    tasks = [(pid, feature_cols, out_prefix) for pid in patient_ids]
    all_rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for patient_rows in executor.map(process_one_patient, tasks):
            all_rows.extend(patient_rows)

    full_df = pd.DataFrame(all_rows)
    # fix: round all numeric columns to 2 decimal places for readability
    numeric_cols = full_df.select_dtypes(include="number").columns
    full_df[numeric_cols] = full_df[numeric_cols].round(2)
    full_df.to_csv(f"output/{out_prefix}_results.csv", index=False)
    print(f"\nSaved: output/{out_prefix}_results.csv")

    if len(full_df):
        metric_cols = ["RMSE", "MAE", "MARD", "TimeLag_min", "CEGA_A_%", "CEGA_B_%", "CEGA_C_%", "CEGA_D_%", "CEGA_E_%"]
        summary = summary_mean_sd(full_df, group_cols=["Horizon", "Architecture"], metric_cols=metric_cols)
        summary.to_csv(f"output/{out_prefix}_summary_mean_sd.csv", index=False)
        print(f"Saved: output/{out_prefix}_summary_mean_sd.csv")
        print(summary.to_string(index=False))
    return full_df


if __name__ == "__main__":
    print("=" * 70)
    print("PASS 1: all 12 patients, insulin-based feature set (no HR/Steps)")
    print("=" * 70)
    run_comparison(PATIENT_IDS, FEATURE_COLS_ALL12, out_prefix="architecture_all12")

    print("\n" + "=" * 70)
    print("PASS 2: 2018 cohort only (6 patients), richer feature set incl. HR/Steps")
    print("=" * 70)
    run_comparison(PATIENT_IDS_2018, FEATURE_COLS_2018_RICH, out_prefix="architecture_2018_wearable")


# --------------------------------------------------------------------------- #
# Why N-BEATS/N-HiTS and TFT are not in this pass
# --------------------------------------------------------------------------- #
# N-BEATS/N-HiTS: designed for univariate series; running them on the full
# multimodal feature set would require a separate architecture (they don't
# take multivariate input the same way as the RNN/attention models above).
# Cleanest to add as a *second* comparison using Glucose-only input, run
# alongside "A: Glucose only" from multimodel_compare_all.py rather than
# mixed into this multimodal architecture table.
#
# TFT: needs static/known-future covariate handling (best via
# pytorch-forecasting, which is a different stack than Keras) — worth adding
# once these five architectures' results are validated and you decide it's
# worth the extra dependency.
