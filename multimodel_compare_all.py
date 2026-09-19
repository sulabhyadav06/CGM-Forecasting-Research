"""
multimodel_compare_all.py -- UPDATED

Changes from your original version:
  1. Adds insulin features (bolus_last_60min, basal_rate) -- new feature set
     "E: Glucose + Insulin (bolus+basal)" and an updated "F: Full multimodal
     (HR+Steps+Carbs+Sleep+Insulin)".
  2. PATIENT_IDS now covers all 12 (6x 2018 + 6x 2020).
  3. HORIZONS now includes 15min alongside 30/60min.
  4. Cohort-aware feature sets: HeartRate/Steps are 100% missing for the 6
     2020 patients (different wristband -- see our earlier discussion), so
     any feature set containing them is SKIPPED for 2020 patients rather
     than silently trained on placeholder/constant values. Feature sets
     that don't depend on HR/Steps (A, C, E) still run on all 12.
  5. Requires parse_xml.py to have been re-run with the insulin extraction
     added (see parse_xml_v2_with_insulin.py) so that
     data/<id>_{training,testing}_multimodal.csv contains bolus_last_60min
     and basal_rate columns. If you haven't re-generated those CSVs yet,
     feature sets E and F will KeyError -- regenerate first.

Outputs (unchanged structure, same filenames):
    output/full_results_by_patient.csv
    output/summary_mean_std.csv
    output/rmse_summary_chart.png
"""

import os
# fix: tensorflow-metal's GPU backend is frequently SLOWER than CPU for
# small recurrent models -- force CPU for this workload (see architecture
# comparison script for the full explanation).
os.environ["TF_METAL_DEVICE_PLACEMENT"] = "0"
import tensorflow as tf
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass
print("Visible devices:", tf.config.get_visible_devices())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

PATIENT_IDS_2018 = ["559", "563", "570", "575", "588", "591"]
PATIENT_IDS_2020 = ["540", "544", "552", "567", "584", "596"]
PATIENT_IDS = PATIENT_IDS_2018 + PATIENT_IDS_2020

HORIZONS = {"15min": 3, "30min": 6, "60min": 12}
WINDOW = 12
EPOCHS = 15
BATCH_SIZE = 128

os.makedirs("output", exist_ok=True)

FEATURE_SETS = {
    "A: Glucose only": ["Glucose"],
    "B: Glucose + HeartRate": ["Glucose", "HeartRate"],
    "C: Glucose + Carbs(meals)": ["Glucose", "carbs_last_60min"],
    "D: Glucose + HR + Steps + Carbs + Sleep": [
        "Glucose", "HeartRate", "Steps", "carbs_last_60min", "is_sleeping"
    ],
    "E: Glucose + Insulin (bolus+basal)": ["Glucose", "bolus_last_60min", "basal_rate"],
    "F: Full multimodal (HR+Steps+Carbs+Sleep+Insulin)": [
        "Glucose", "HeartRate", "Steps", "carbs_last_60min", "is_sleeping",
        "bolus_last_60min", "basal_rate",
    ],
}

# Feature sets requiring HeartRate/Steps -- unavailable for the 6 2020 patients
REQUIRES_WEARABLE = {"B: Glucose + HeartRate", "D: Glucose + HR + Steps + Carbs + Sleep",
                      "F: Full multimodal (HR+Steps+Carbs+Sleep+Insulin)"}


def create_sequences(X, y, window=WINDOW):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i: i + window])
        ys.append(y[i + window])
    return np.array(Xs), np.array(ys)


def build_lstm(input_dim):
    model = Sequential([
        LSTM(64, input_shape=(WINDOW, input_dim), return_sequences=False),
        Dropout(0.2), Dense(32, activation="relu"), Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def process_one_patient(patient_id):
    """
    Runs every horizon x feature-set combination for ONE patient. Pulled
    out to a top-level function so it can be pickled and run in a separate
    process via ProcessPoolExecutor -- patients are fully independent, so
    this is a safe, high-value way to use multiple CPU cores instead of
    training everything in one sequential loop.
    """
    is_2020_patient = patient_id in PATIENT_IDS_2020
    train_path = f"data/{patient_id}_training_multimodal.csv"
    test_path = f"data/{patient_id}_testing_multimodal.csv"
    rows = []

    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print(f"Skipping patient {patient_id} - CSVs not found. Run parse_xml.py for this patient first.")
        return rows

    train_df = pd.read_csv(train_path, parse_dates=["Timestamp"])
    test_df = pd.read_csv(test_path, parse_dates=["Timestamp"])
    for df in (train_df, test_df):
        df.sort_values("Timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

    for horizon_name, horizon_steps in HORIZONS.items():
        base_train, base_test = train_df.copy(), test_df.copy()
        base_train["target"] = base_train["Glucose"].shift(-horizon_steps)
        base_test["target"] = base_test["Glucose"].shift(-horizon_steps)

        for model_name, cols in FEATURE_SETS.items():
            if is_2020_patient and model_name in REQUIRES_WEARABLE:
                print(f"Skipping {model_name} for patient {patient_id} "
                      f"(2020 cohort has no HeartRate/Steps data -- different wristband).")
                continue

            missing_cols = [c for c in cols if c not in base_train.columns]
            if missing_cols:
                print(f"Skipping {model_name} for patient {patient_id}: missing columns {missing_cols} "
                      f"(re-run parse_xml.py with insulin extraction to generate these).")
                continue

            all_nan_cols = [c for c in cols if base_train[c].isna().all() or base_test[c].isna().all()]
            if all_nan_cols:
                print(f"Skipping {model_name} for patient {patient_id}: columns {all_nan_cols} "
                      f"are entirely missing for this patient (likely an MDI patient with no "
                      f"pump data, or a wearable-less 2020 patient).")
                continue

            t_train = base_train.dropna(subset=cols + ["target"]).reset_index(drop=True)
            t_test = base_test.dropna(subset=cols + ["target"]).reset_index(drop=True)

            if len(t_train) == 0 or len(t_test) == 0:
                print(f"Skipping {model_name} for patient {patient_id}: 0 rows remain after dropna.")
                continue

            print(f"Patient {patient_id} | Horizon {horizon_name} | Model {model_name}")

            scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
            X_train_raw = scaler_X.fit_transform(t_train[cols].values)
            y_train_raw = scaler_y.fit_transform(t_train[["target"]].values)
            X_test_raw = scaler_X.transform(t_test[cols].values)
            y_test_raw = scaler_y.transform(t_test[["target"]].values)

            X_train, y_train = create_sequences(X_train_raw, y_train_raw)
            X_test, y_test = create_sequences(X_test_raw, y_test_raw)

            if len(X_train) < 50 or len(X_test) < 20:
                print(f"  Not enough data for patient {patient_id}/{model_name}, skipping.")
                continue

            model = build_lstm(input_dim=len(cols))
            model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1,
                      callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
                      verbose=0)

            preds_scaled = model.predict(X_test, verbose=0)
            preds = scaler_y.inverse_transform(preds_scaled).flatten()
            actual = scaler_y.inverse_transform(y_test).flatten()

            rmse = round(float(np.sqrt(mean_squared_error(actual, preds))), 2)
            mae = round(float(mean_absolute_error(actual, preds)), 2)
            r2 = round(float(r2_score(actual, preds)), 2)
            print(f"  Patient {patient_id} {horizon_name} {model_name}: RMSE={rmse:.2f} MAE={mae:.2f} R2={r2:.2f}")

            rows.append({
                "Patient": patient_id, "Cohort": "2020" if is_2020_patient else "2018",
                "Horizon": horizon_name, "Model": model_name,
                "RMSE": rmse, "MAE": mae, "R2": r2,
            })
    return rows


if __name__ == "__main__":
    import concurrent.futures

    max_workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"Running {len(PATIENT_IDS)} patients across {max_workers} parallel processes...")

    all_rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for patient_rows in executor.map(process_one_patient, PATIENT_IDS):
            all_rows.extend(patient_rows)

    full_df = pd.DataFrame(all_rows)
    if full_df.empty:
        raise SystemExit(
            "\nNo results were produced -- every patient was skipped.\n"
            "This almost always means data/<id>_training_multimodal.csv and "
            "data/<id>_testing_multimodal.csv don't exist yet.\n"
            "Run parse_xml.py first (with the insulin extraction added), confirm "
            "it prints 24 'Saved: ...' lines with no 'Missing' lines, and check "
            "that the CSVs actually appear in data/ before re-running this script."
        )

    summary = full_df.groupby(["Horizon", "Model"])["RMSE"].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["Horizon", "Model", "RMSE_mean", "RMSE_std", "N_patients"]
    summary["RMSE_mean"] = summary["RMSE_mean"].round(2)
    summary["RMSE_std"] = summary["RMSE_std"].round(2)

    # fix: round all numeric columns to 2 decimal places for readability
    full_df_numeric_cols = full_df.select_dtypes(include="number").columns
    full_df[full_df_numeric_cols] = full_df[full_df_numeric_cols].round(2)
    full_df.to_csv("output/full_results_by_patient.csv", index=False)
    print("\nSaved: output/full_results_by_patient.csv")

    summary.to_csv("output/summary_mean_std.csv", index=False)
    print("\n" + "=" * 70)
    print("SUMMARY: Mean +/- Std RMSE across patients")
    print("=" * 70)
    print(summary.to_string(index=False))
    print("\nNote: N_patients for feature sets B/D/F will be <=6 (2018 cohort only);")
    print("A/C/E will be up to 12 (both cohorts) -- not directly comparable N's,")
    print("keep this in mind when reading the summary table.")

    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(18, 5), sharey=True)
    for ax, horizon_name in zip(axes, HORIZONS.keys()):
        sub = summary[summary["Horizon"] == horizon_name]
        ax.bar(sub["Model"], sub["RMSE_mean"], yerr=sub["RMSE_std"], capsize=5)
        ax.set_title(f"Horizon: {horizon_name}")
        ax.set_ylabel("RMSE (mg/dL)")
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig("output/rmse_summary_chart.png")
    print("\nSaved: output/rmse_summary_chart.png")
