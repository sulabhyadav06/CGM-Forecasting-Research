"""
Full multimodal comparison across all OhioT1DM patients and both horizons.
Requires that parse_xml.py has already been run for EACH patient, for BOTH
"training" and "testing" splits, producing files like:
    data/570_training_multimodal.csv
    data/570_testing_multimodal.csv
    data/563_training_multimodal.csv
    ... etc

Outputs:
    output/full_results_by_patient.csv   (every patient x horizon x model)
    output/summary_mean_std.csv          (mean +/- std RMSE per model x horizon)
    output/rmse_summary_chart.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

PATIENT_IDS = ["559", "563", "570", "575", "588", "591"]
HORIZONS = {"30min": 6, "60min": 12}
WINDOW = 12
EPOCHS = 15
BATCH_SIZE = 32

os.makedirs("output", exist_ok=True)

FEATURE_SETS = {
    "A: Glucose only": ["Glucose"],
    "B: Glucose + HeartRate": ["Glucose", "HeartRate"],
    "C: Glucose + Carbs(meals)": ["Glucose", "carbs_last_60min"],
    "D: Glucose + HR + Steps + Carbs + Sleep": [
        "Glucose", "HeartRate", "Steps", "carbs_last_60min", "is_sleeping"
    ],
}


def create_sequences(X, y, window=WINDOW):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i : i + window])
        ys.append(y[i + window])
    return np.array(Xs), np.array(ys)


def build_lstm(input_dim):
    model = Sequential([
        LSTM(64, input_shape=(WINDOW, input_dim), return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


all_rows = []

for patient_id in PATIENT_IDS:
    train_path = f"data/{patient_id}_training_multimodal.csv"
    test_path = f"data/{patient_id}_testing_multimodal.csv"

    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print(f"Skipping patient {patient_id} - CSVs not found. Run parse_xml.py for this patient first.")
        continue

    train_df = pd.read_csv(train_path, parse_dates=["Timestamp"])
    test_df = pd.read_csv(test_path, parse_dates=["Timestamp"])

    for df in (train_df, test_df):
        df.sort_values("Timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

    for horizon_name, horizon_steps in HORIZONS.items():
        t_train = train_df.copy()
        t_test = test_df.copy()
        t_train["target"] = t_train["Glucose"].shift(-horizon_steps)
        t_test["target"] = t_test["Glucose"].shift(-horizon_steps)
        t_train.dropna(inplace=True)
        t_test.dropna(inplace=True)
        t_train.reset_index(drop=True, inplace=True)
        t_test.reset_index(drop=True, inplace=True)

        for model_name, cols in FEATURE_SETS.items():
            print(f"\nPatient {patient_id} | Horizon {horizon_name} | Model {model_name}")

            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()

            X_train_raw = scaler_X.fit_transform(t_train[cols].values)
            y_train_raw = scaler_y.fit_transform(t_train[["target"]].values)
            X_test_raw = scaler_X.transform(t_test[cols].values)
            y_test_raw = scaler_y.transform(t_test[["target"]].values)

            X_train, y_train = create_sequences(X_train_raw, y_train_raw)
            X_test, y_test = create_sequences(X_test_raw, y_test_raw)

            if len(X_train) < 50 or len(X_test) < 20:
                print("  Not enough data, skipping.")
                continue

            model = build_lstm(input_dim=len(cols))
            model.fit(
                X_train, y_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_split=0.1,
                verbose=0,
            )

            preds_scaled = model.predict(X_test, verbose=0)
            preds = scaler_y.inverse_transform(preds_scaled).flatten()
            actual = scaler_y.inverse_transform(y_test).flatten()

            rmse = np.sqrt(mean_squared_error(actual, preds))
            mae = mean_absolute_error(actual, preds)
            r2 = r2_score(actual, preds)

            print(f"  RMSE: {rmse:.2f} | MAE: {mae:.2f} | R2: {r2:.3f}")

            all_rows.append({
                "Patient": patient_id,
                "Horizon": horizon_name,
                "Model": model_name,
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2,
            })

# ---------------------------------------------------------------
# Save full results
# ---------------------------------------------------------------
full_df = pd.DataFrame(all_rows)
full_df.to_csv("output/full_results_by_patient.csv", index=False)
print("\nSaved: output/full_results_by_patient.csv")

# ---------------------------------------------------------------
# Summary: mean +/- std RMSE per Model x Horizon across patients
# ---------------------------------------------------------------
summary = full_df.groupby(["Horizon", "Model"])["RMSE"].agg(["mean", "std", "count"]).reset_index()
summary.columns = ["Horizon", "Model", "RMSE_mean", "RMSE_std", "N_patients"]
summary.to_csv("output/summary_mean_std.csv", index=False)

print("\n" + "=" * 70)
print("SUMMARY: Mean +/- Std RMSE across patients")
print("=" * 70)
print(summary.to_string(index=False))

# ---------------------------------------------------------------
# Chart: grouped bar chart, RMSE mean with error bars, per horizon
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, len(HORIZONS), figsize=(14, 5), sharey=True)
for ax, horizon_name in zip(axes, HORIZONS.keys()):
    sub = summary[summary["Horizon"] == horizon_name]
    ax.bar(sub["Model"], sub["RMSE_mean"], yerr=sub["RMSE_std"], capsize=5,
           color=["gray", "steelblue", "orange", "green"])
    ax.set_title(f"Horizon: {horizon_name}")
    ax.set_ylabel("RMSE (mg/dL)")
    ax.tick_params(axis="x", rotation=25)

plt.tight_layout()
plt.savefig("output/rmse_summary_chart.png")
print("\nSaved: output/rmse_summary_chart.png")