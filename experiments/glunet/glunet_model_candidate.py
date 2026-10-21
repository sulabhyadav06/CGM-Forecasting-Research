"""Experimental GluNet architecture candidate.

This is a documented architectural hypothesis, not a verified exact
reproduction. The input projection is counted as layer 1, followed by
four gated causal convolutions with dilations 1, 2, 4, 8.

Provisional choices still requiring source verification:
- input projection counts toward the paper's five-layer description;
- skip paths are projected to 64 channels and summed;
- the point forecast uses expected class value;
- the existing 256-class label convention is retained.
"""

from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

N_CLASSES = 256
DILATIONS = (1, 2, 4, 8)
CHANNELS = (32, 32, 64, 64)


class GatedCausalConv1d(nn.Module):
    """Causal convolution with tanh/sigmoid gating."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()
        self.padding = dilation  # kernel size = 2
        self.filter_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=2,
            dilation=dilation, padding=0
        )
        self.gate_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=2,
            dilation=dilation, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.padding, 0))
        return torch.tanh(self.filter_conv(x)) * torch.sigmoid(
            self.gate_conv(x)
        )


class GatedResidualBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int,
        dilation: int, skip_channels: int = 64
    ):
        super().__init__()
        self.gated = GatedCausalConv1d(
            in_channels, out_channels, dilation
        )
        self.residual_projection = (
            nn.Identity() if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )
        self.skip_projection = nn.Conv1d(
            out_channels, skip_channels, kernel_size=1
        )

    def forward(self, x: torch.Tensor):
        z = self.gated(x)
        residual = self.residual_projection(x) + z
        skip = self.skip_projection(z)
        return residual, skip


class GluNetCandidate(nn.Module):
    """Five-main-layer GluNet candidate with 256 output classes.

    Input: (batch, time, features)
    Output: unnormalized class logits, shape (batch, 256)
    """

    def __init__(
        self,
        input_features: int = 3,
        lookback: int = 16,
        dilations: Sequence[int] = DILATIONS,
        channels: Sequence[int] = CHANNELS,
        n_classes: int = N_CLASSES,
    ):
        super().__init__()
        if input_features < 1:
            raise ValueError("input_features must be positive")
        if lookback < 1:
            raise ValueError("lookback must be positive")
        if tuple(dilations) != DILATIONS:
            raise ValueError(f"Expected dilations {DILATIONS}")
        if tuple(channels) != CHANNELS:
            raise ValueError(f"Expected channel layout {CHANNELS}")
        if n_classes != N_CLASSES:
            raise ValueError("Expected 256 output classes")

        self.input_features = input_features
        self.lookback = lookback
        self.n_classes = n_classes

        # Candidate interpretation: this projection is layer 1.
        self.input_projection = nn.Conv1d(
            input_features, channels[0], kernel_size=1
        )

        self.blocks = nn.ModuleList()
        in_channels = channels[0]
        for dilation, out_channels in zip(dilations, channels):
            self.blocks.append(
                GatedResidualBlock(
                    in_channels, out_channels, dilation,
                    skip_channels=channels[-1]
                )
            )
            in_channels = out_channels

        self.postprocess = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(channels[-1], channels[-1], kernel_size=1),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(channels[-1], n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Input must have shape (batch, time, features)")
        if x.shape[1] != self.lookback:
            raise ValueError(f"Expected {self.lookback} input time steps")
        if x.shape[2] != self.input_features:
            raise ValueError(f"Expected {self.input_features} features")

        x = self.input_projection(x.transpose(1, 2))
        skip_sum = None
        for block in self.blocks:
            x, skip = block(x)
            skip_sum = skip if skip_sum is None else skip_sum + skip

        features = self.postprocess(skip_sum)
        return self.classifier(features[:, :, -1])

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self(x), dim=-1)

    @torch.no_grad()
    def predict_delta_provisional(self, x: torch.Tensor) -> torch.Tensor:
        """Provisional expected-class decoder; not verified paper postprocessing."""
        probabilities = self.predict_proba(x)
        delta_values = (
            torch.arange(self.n_classes, device=x.device,
                         dtype=probabilities.dtype) + 1
        ) / 10.0 - 12.8
        return (probabilities * delta_values.unsqueeze(0)).sum(dim=-1)

    @torch.no_grad()
    def predict_delta_argmax_provisional(
        self, x: torch.Tensor
        ) -> torch.Tensor:
        """Recover the glucose change for the most probable class.

        Provisional decoder: the supplied paper excerpt does not
        explicitly establish whether inference uses argmax.
        """
        predicted_classes = self(x).argmax(dim=-1)
        paper_labels = predicted_classes.to(torch.float32) + 1.0
        return paper_labels / 10.0 - 12.8
