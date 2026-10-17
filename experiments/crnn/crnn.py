"""CRNN glucose forecasting model.

Architecture follows the dimensions in Table V of:
Li et al., "Convolutional Recurrent Neural Networks for Glucose Prediction",
IEEE Journal of Biomedical and Health Informatics, 2020.

Unspecified implementation details must be checked against the full paper.
Input shape:  (batch, 24, features)
Output shape: (batch, 1), predicted glucose change.
"""

import torch
from torch import nn

try:
    from experiments.crnn.modified_lstm import ModifiedLSTM
except ModuleNotFoundError:
    from modified_lstm import ModifiedLSTM

class CRNN(nn.Module):
    """Convolutional recurrent network for personalized glucose prediction."""

    def __init__(self, n_features: int, dropout: float = 0.0):
        super().__init__()

        if n_features < 1:
            raise ValueError("n_features must be at least 1")

        # Preserve sequence length through each convolution so that the
        # three pooling layers yield 24 -> 12 -> 6 -> 3 time steps.
        self.conv1 = nn.Conv1d(
            in_channels=n_features,
            out_channels=8,
            kernel_size=4,
            padding="same",
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(
            in_channels=8,
            out_channels=16,
            kernel_size=4,
            padding="same",
        )
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(
            in_channels=16,
            out_channels=32,
            kernel_size=4,
            padding="same",
        )
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.lstm = ModifiedLSTM(
    input_size=32,
    hidden_size=64,
    batch_first=True,
    )

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 1)

        # These ReLU choices are provisional adaptations until verified
        # against the paper's full architecture description.
        self.activation = nn.ReLU()

    def forward(self, x):
        """Return predicted glucose change for one forecast horizon."""
        if x.ndim != 3:
            raise ValueError(
                f"Expected (batch, steps, features), got {tuple(x.shape)}"
            )

        if x.shape[1] != 24:
            raise ValueError(
                f"Expected 24 input steps, got {x.shape[1]}"
            )

        # PyTorch Conv1d expects (batch, channels, steps).
        x = x.transpose(1, 2)

        x = self.pool1(self.activation(self.conv1(x)))
        x = self.pool2(self.activation(self.conv2(x)))
        x = self.pool3(self.activation(self.conv3(x)))

        # Back to (batch, steps, channels) for the LSTM.
        x = x.transpose(1, 2)
        sequence, _ = self.lstm(x)

         # Apply dropout after the recurrent layer, as described in the paper.
        sequence = self.dropout(sequence)

         # Use the last recurrent output for the many-to-one prediction.
        x = sequence[:, -1, :]
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)


if __name__ == "__main__":
    torch.manual_seed(42)

    model = CRNN(n_features=3)
    example = torch.randn(8, 24, 3)
    prediction = model(example)

    print("Input shape:", tuple(example.shape))
    print("Output shape:", tuple(prediction.shape))
    print("Trainable parameters:", sum(
        p.numel() for p in model.parameters() if p.requires_grad
    ))

    assert prediction.shape == (8, 1)
    print("CRNN shape smoke test passed.")
