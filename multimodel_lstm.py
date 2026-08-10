import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Enable multi-core CPU threading for fast PyTorch execution
torch.set_num_threads(os.cpu_count())
device = torch.device("cpu")
print("Using CPU with multi-threading acceleration")

# 1. Load Data & Filter Zero Dropouts Automatically
df = pd.read_csv("data/cgm_data.csv")

# Filter out non-physiological values (0s and negative values)
df = df[df["Glucose"] > 30].copy()

df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

# 2. Feature Engineering
df["Glucose_Velocity"] = df["Glucose"].diff().fillna(0)
hours = df["Timestamp"].dt.hour + df["Timestamp"].dt.minute / 60.0
df["Hour_Sin"] = np.sin(2 * np.pi * hours / 24.0)
df["Hour_Cos"] = np.cos(2 * np.pi * hours / 24.0)

# Target: 30-min Horizon (6 steps ahead)
df["target_30"] = df["Glucose"].shift(-6)
df = df.dropna().reset_index(drop=True)

# Target: 30-min Horizon (6 steps ahead)
df["target_30"] = df["Glucose"].shift(-6)
df = df.dropna().reset_index(drop=True)

feature_cols = ["Glucose", "Glucose_Velocity", "Hour_Sin", "Hour_Cos"]
print(f"Features Engineered: {feature_cols}")

# 3. Scale Features
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

scaled_X_data = scaler_X.fit_transform(df[feature_cols].values)
scaled_y_data = scaler_y.fit_transform(df[["target_30"]].values)

# 4. Ultra-Fast Vectorised Sequence Generator
def create_fast_sequences(X_data, y_data, window_size=12):
    # Vectorised numpy sliding window (100x faster than python list loop)
    shape = (len(X_data) - window_size, window_size, X_data.shape[1])
    strides = (X_data.strides[0], X_data.strides[0], X_data.strides[1])
    X_seq = np.lib.stride_tricks.as_strided(X_data, shape=shape, strides=strides)
    y_seq = y_data[window_size:]
    return X_seq, y_seq

X_seq, y_seq = create_fast_sequences(scaled_X_data, scaled_y_data, window_size=12)

# Train / Test Split
split_idx = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

# Convert to Tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

dataset = TensorDataset(X_train_t, y_train_t)
loader = DataLoader(dataset, batch_size=512, shuffle=True)

# 5. Lightweight PyTorch LSTM
class MultimodalLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=1):
        super(MultimodalLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = MultimodalLSTM(input_dim=len(feature_cols)).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# 6. Fast Training Loop (5 Epochs)
print("Training Multimodal LSTM Model...")
model.train()
for epoch in range(5):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

# 7. Evaluate
model.eval()
with torch.no_grad():
    scaled_preds = model(X_test_t).numpy()

actual_preds = scaler_y.inverse_transform(scaled_preds)
actual_y_test = scaler_y.inverse_transform(y_test)

rmse = np.sqrt(mean_squared_error(actual_y_test, actual_preds))
mae = mean_absolute_error(actual_y_test, actual_preds)

print("\n" + "="*45)
print("       MULTIMODAL LSTM EVALUATION RESULTS       ")
print("="*45)
print(f" Multimodal LSTM RMSE : {rmse:.2f} mg/dL")
print(f" Multimodal LSTM MAE  : {mae:.2f} mg/dL")
print("="*45)

# Save Comparison Plot
os.makedirs("output", exist_ok=True)
plt.figure(figsize=(10, 4))
plt.plot(actual_y_test[:100], label="Actual Glucose", color="black")
plt.plot(actual_preds[:100], label="Multimodal LSTM Pred", color="green", linestyle="--")
plt.title("Multimodal LSTM: Actual vs Predicted Glucose (30-min Horizon)")