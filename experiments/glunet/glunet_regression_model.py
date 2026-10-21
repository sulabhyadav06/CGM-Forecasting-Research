"""Continuous glucose-change regression using a GluNet-inspired backbone.

This is an experimental regression variant, not an exact reproduction
of the published GluNet classification model.
"""

import torch
from torch import nn

from experiments.glunet.glunet_model import (
    GluNetResidualBlock,
    DEFAULT_DILATIONS,
)


class GluNetRegressor(nn.Module):
    """Predict continuous glucose change in mg/dL.

    Input:
        x: (batch, time, features)

    Output:
        predicted glucose change: (batch,)
    """

    def __init__(
        self,
        input_features=3,
        channels=32,
        dilations=DEFAULT_DILATIONS,
        kernel_size=2,
        dropout=0.0,
    ):
        super().__init__()

        if input_features < 1:
            raise ValueError("input_features must be positive")
        if channels < 1:
            raise ValueError("channels must be positive")

        self.input_features = input_features
        self.channels = channels
        self.dilations = tuple(dilations)

        self.input_projection = nn.Conv1d(
            input_features,
            channels,
            kernel_size=1,
        )

        self.blocks = nn.ModuleList([
            GluNetResidualBlock(
                in_channels=channels,
                channels=channels,
                dilation=dilation,
                kernel_size=kernel_size,
                dropout=dropout,
            )
            for dilation in self.dilations
        ])

        self.postprocess = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.ReLU(),
        )

        self.regressor = nn.Linear(channels, 1)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(
                "Expected (batch, time, features), "
                f"received {tuple(x.shape)}"
            )

        if x.shape[-1] != self.input_features:
            raise ValueError(
                f"Expected {self.input_features} features, "
                f"received {x.shape[-1]}"
            )

        x = x.transpose(1, 2)
        x = self.input_projection(x)

        skip_outputs = []

        for block in self.blocks:
            x, skip = block(x)
            skip_outputs.append(skip)

        combined_skip = torch.stack(
            skip_outputs,
            dim=0,
        ).sum(dim=0)

        features = self.postprocess(combined_skip)
        last_timestep = features[:, :, -1]

        return self.regressor(last_timestep).squeeze(-1)
