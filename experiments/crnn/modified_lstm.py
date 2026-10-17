"""Modified LSTM output connection for CRNN glucose forecasting.

Based on:
Li et al., "Convolutional Recurrent Neural Networks
for Glucose Prediction", IEEE JBHI, 2020.

Paper's modified output:
    h_t = o_t * tanh(c_t)
    y_t = h_t + x_t

The input x_t and hidden state h_t must have matching dimensions
for the residual addition. This module uses a projection when
the dimensions differ.
"""

import torch
from torch import nn


class ModifiedLSTM(nn.Module):
    """LSTM with the paper's modified output connection."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        batch_first: bool = True,
    ):
        super().__init__()

        if input_size < 1 or hidden_size < 1:
            raise ValueError("input_size and hidden_size must be positive")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Standard LSTM gate parameters.
        self.weight_ih = nn.Parameter(
            torch.empty(4 * hidden_size, input_size)
        )
        self.weight_hh = nn.Parameter(
            torch.empty(4 * hidden_size, hidden_size)
        )
        self.bias_ih = nn.Parameter(torch.empty(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(4 * hidden_size))

        # Project input to hidden size if residual dimensions differ.
        if input_size != hidden_size:
            self.residual_projection = nn.Linear(
                input_size, hidden_size, bias=False
            )
        else:
            self.residual_projection = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize trainable parameters."""
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.orthogonal_(self.weight_hh)
        nn.init.zeros_(self.bias_ih)
        nn.init.zeros_(self.bias_hh)

        # Standard LSTM forget-gate bias initialization.
        with torch.no_grad():
            self.bias_ih[self.hidden_size:2 * self.hidden_size].fill_(1.0)

    def forward(self, x):
        """Process a sequence and return modified outputs and final state."""
        if x.ndim != 3:
            raise ValueError(
                f"Expected a 3D input tensor, got {tuple(x.shape)}"
            )

        if not self.batch_first:
            x = x.transpose(0, 1)

        batch_size, sequence_length, _ = x.shape

        h = x.new_zeros(batch_size, self.hidden_size)
        c = x.new_zeros(batch_size, self.hidden_size)

        outputs = []

        for t in range(sequence_length):
            x_t = x[:, t, :]

            gates = (
                torch.nn.functional.linear(x_t, self.weight_ih, self.bias_ih)
                + torch.nn.functional.linear(h, self.weight_hh, self.bias_hh)
            )

            i, f, g, o = gates.chunk(4, dim=-1)

            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            g = torch.tanh(g)
            o = torch.sigmoid(o)

            # Standard LSTM cell-state update.
            c = f * c + i * g

            # Internal hidden state, as in the paper.
            h = o * torch.tanh(c)

            # Modified output connection.
            y_t = h + self.residual_projection(x_t)

            outputs.append(y_t)

        output = torch.stack(outputs, dim=1)

        if not self.batch_first:
            output = output.transpose(0, 1)

        return output, (h, c)

