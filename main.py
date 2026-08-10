import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import load_model
from lstm_model import create_sequences, build_lstm_model

# 1. Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("output", exist_ok=True)

# 2. Fast Load Data
if os.path.exists("data/cgm_data.parquet"):
    df = pd.read_parquet("data/cgm_data.parquet")
else:
    df = pd.read_csv("data/cgm_data.csv")

df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

# 3. Create 30-min Target Horizon (6 steps ahead @ 5-min intervals)
df["target_30"] = df["Glucose"].shift(-6)
df = df.dropna().reset_index(drop=True)

glucose_vals = df["Glucose"].values
target_vals = df["target_30"].values

# 4. Create Sequences
X, y = create_sequences(glucose_vals, target_vals, window_size=12)
split_idx = int(len(X) * 0.8)

# Baseline Data (2D)
X_flat = X.reshape((X.shape[0], X.shape[1]))
X_train_flat, X_test_flat = X_flat[:split_idx], X_flat[split_idx:]

# LSTM Data (3D)
X_train_3d = X[:split_idx].reshape((split_idx, 12, 1))
X_test_3d = X[split_idx:].reshape((len(X) - split_idx, 12, 1))

y_train, y_test = y[:split_idx], y[split_idx:]

# --- MODEL 1: Linear Regression (Cached) ---
lr_model_path = "models/linear_regression.joblib"
if os.path.exists(lr_model_path):
    print("Loading cached Linear Regression model...")
    lr = joblib.load(lr_model_path)
else:
    print("Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train_flat, y_train)
    joblib.dump(lr, lr_model_path)

lr_preds = lr.predict(X_test_flat)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
lr_mae = mean_absolute_error(y_test, lr_preds)

# --- MODEL 2: Random Forest (Cached) ---
rf_model_path = "models/random_forest.joblib"
if os.path.exists(rf_model_path):
    print("Loading cached Random Forest model...")
    rf = joblib.load(rf_model_path)
else:
    print("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=20, n_jobs=-1, random_state=42)
    rf.fit(X_train_flat, y_train)
    joblib.dump(rf, rf_model_path)

rf_preds = rf.predict(X_test_flat)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
rf_mae = mean_absolute_error(y_test, rf_preds)

# --- MODEL 3: LSTM Neural Network (Cached) ---
lstm_model_path = "models/lstm_model.keras"
if os.path.exists(lstm_model_path):
    print("Loading cached LSTM model...")
    lstm = load_model(lstm_model_path)
else:
    print("Training LSTM Model...")
    lstm = build_lstm_model(input_shape=(12, 1))
    lstm.fit(
        X_train_3d,
        y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test_3d, y_test),
        verbose=1
    )
    lstm.save(lstm_model_path)

lstm_preds = lstm.predict(X_test_3d).flatten()
lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_preds))
lstm_mae = mean_absolute_error(y_test, lstm_preds)

# --- RESULTS SUMMARY ---
results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "Baseline LSTM"],
    "RMSE (mg/dL)": [lr_rmse, rf_rmse, lstm_rmse],
    "MAE (mg/dL)": [lr_mae, rf_mae, lstm_mae]
})

print("\n" + "="*45)
print("             MODEL EVALUATION TABLE            ")
print("="*45)
print(results)
print("="*45)

results.to_csv("output/model_results.csv", index=False)

# Save Plot
plt.figure(figsize=(10, 4))
plt.plot(y_test[:100], label="Actual Glucose", color="black")
plt.plot(lstm_preds[:100], label="LSTM Prediction", color="red", linestyle="--")
plt.title("Actual vs Predicted Glucose (30-min Horizon)")
plt.xlabel("Time Steps")
plt.ylabel("Glucose (mg/dL)")
plt.legend()
plt.tight_layout()
plt.savefig("output/actual_vs_predicted.png")
print("\nSuccess! Results saved to output/ folder.")