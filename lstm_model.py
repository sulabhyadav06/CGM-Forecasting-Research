import numpy as np
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential


def create_sequences(data, target, window_size=12):
    """Converts continuous glucose data into 3D sequential sliding windows."""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(target[i + window_size])
    return np.array(X), np.array(y)


def build_lstm_model(input_shape=(12, 1)):
    """Builds the baseline LSTM model."""
    model = Sequential(
        [
            LSTM(64, input_shape=input_shape, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model