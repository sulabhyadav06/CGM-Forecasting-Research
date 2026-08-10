"""
Model A/B/C/D comparison for CGM forecasting.
Trains identical LSTM architectures on different feature subsets to isolate
the effect of adding heart rate, carbs, steps, and sleep.

Requires: data/570_training_multimodal.csv and data/570_testing_multimodal.csv
(produced by parse_xml.py with SPLIT="training" and SPLIT="testing")
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

PATIENT_ID = "570"
WINDOW = 12          # 1 hour of history (12 x 5-min)
HORIZON_STEPS = 6    # 30-min ahead
EPOCHS = 15
BATCH_SIZE = 32

os.makedirs("output", exist_ok=True)

# ---------------------------------------------------------------
# 1. Load train/test CSVs produced by parse_xml.py
# ---------------------------------------------------------------
train_df = pd.read_csv(f"data/{PATIENT_ID}_training_multimodal.csv", parse_dates=["Timestamp"])
test_df = pd.read_csv(f"data/{PATIENT_ID}_testing_multimodal.csv", parse_dates=["Timestamp"])

for df in (train_df, test_df):
    df.sort_values("Timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["target_30"] = df["Glucose"].shift(-HORIZON_STEPS)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

# ---------------------------------------------------------------
# 2. Define the 4 feature sets
# ---------------------------------------------------------------
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


# ---------------------------------------------------------------
# 3. Train + evaluate each model
# ---------------------------------------------------------------
results = []
predictions_store = {}

for name, cols in FEATURE_SETS.items():
    print(f"\n{'='*60}\nTraining Model {name}\nFeatures: {cols}\n{'='*60}")

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_raw = scaler_X.fit_transform(train_df[cols].values)
    y_train_raw = scaler_y.fit_transform(train_df[["target_30"]].values)

    X_test_raw = scaler_X.transform(test_df[cols].values)
    y_test_raw = scaler_y.transform(test_df[["target_30"]].values)

    X_train, y_train = create_sequences(X_train_raw, y_train_raw)
    X_test, y_test = create_sequences(X_test_raw, y_test_raw)

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

    print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | R2: {r2:.3f}")

    results.append({"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2})
    predictions_store[name] = (actual, preds)

# ---------------------------------------------------------------
# 4. Results table
# ---------------------------------------------------------------
results_df = pd.DataFrame(results)
print("\n" + "=" * 60)
print("MODEL A/B/C/D COMPARISON RESULTS")
print("=" * 60)
print(results_df.to_string(index=False))
results_df.to_csv("output/model_ABCD_results.csv", index=False)

# ---------------------------------------------------------------
# 5. Bar chart of RMSE
# ---------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["RMSE"], color=["gray", "steelblue", "orange", "green"])
plt.ylabel("RMSE (mg/dL)")
plt.title(f"Patient {PATIENT_ID}: RMSE by Feature Set (30-min horizon)")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("output/rmse_comparison_ABCD.png")
print("\nSaved: output/model_ABCD_results.csv and output/rmse_comparison_ABCD.png")

# ---------------------------------------------------------------
# 6. Actual vs Predicted plot for best (Model D) and worst (Model A)
# ---------------------------------------------------------------
plt.figure(figsize=(10, 4))
actual_a, preds_a = predictions_store["A: Glucose only"]
actual_d, preds_d = predictions_store["D: Glucose + HR + Steps + Carbs + Sleep"]
plt.plot(actual_a[:150], label="Actual", color="black")
plt.plot(preds_a[:150], label="Model A (glucose only)", linestyle="--", color="gray")
plt.plot(preds_d[:150], label="Model D (multimodal)", linestyle="--", color="green")
plt.legend()
plt.title(f"Patient {PATIENT_ID}: Actual vs Predicted (first 150 test points)")
plt.xlabel("Time step")
plt.ylabel("Glucose (mg/dL)")
plt.tight_layout()
plt.savefig("output/actual_vs_predicted_ABCD.png")
print("Saved: output/actual_vs_predicted_ABCD.png")