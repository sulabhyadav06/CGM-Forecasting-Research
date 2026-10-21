"""GluNet-inspired causal dilated CNN for glucose-change classification.

Reference:
K. Li et al., "GluNet: A Deep Learning Framework for Accurate
Glucose Forecasting," IEEE Journal of Biomedical and Health
Informatics, 2020.

The paper specifies causal dilated convolutions, gated activations,
residual and parameterized skip connections, and a 256-class output.

Implementation note:
The paper does not fully specify all layer-level dimensions and
implementation details. Choices made here are documented adaptations,
not claims of exact source-code reproduction.
"""

from typing import Dict, Sequence

import torch
from torch import nn
import torch.nn.functional as F


N_CLASSES = 256
DEFAULT_DILATIONS = (1, 2, 4, 8)


class GatedDilatedConv1d(nn.Module):
    """Causal dilated convolution with a tanh-sigmoid gate."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        dilation: int = 1,
    ):
        super().__init__()

        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")
        if dilation < 1:
            raise ValueError("dilation must be positive")

        self.padding = (kernel_size - 1) * dilation

        self.filter_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )
        self.gate_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Left-only padding prevents access to future timesteps.
        x = F.pad(x, (self.padding, 0))

        filter_value = torch.tanh(self.filter_conv(x))
        gate_value = torch.sigmoid(self.gate_conv(x))

        return filter_value * gate_value


class GluNetResidualBlock(nn.Module):
    """Gated causal convolution with residual and skip projections."""

    def __init__(
        self,
        in_channels: int,
        channels: int,
        dilation: int,
        kernel_size: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.gated_conv = GatedDilatedConv1d(
            in_channels=in_channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.dropout = nn.Dropout(dropout)

        # Projection allows the residual path to change channel width.
        self.residual_projection = nn.Conv1d(
            channels, in_channels, kernel_size=1
        )
        self.skip_projection = nn.Conv1d(
            channels, channels, kernel_size=1
        )

    def forward(self, x: torch.Tensor):
        gated = self.dropout(self.gated_conv(x))

        residual = x + self.residual_projection(gated)
        skip = self.skip_projection(gated)

        return residual, skip


class GluNet(nn.Module):
    """GluNet-style probabilistic glucose-change forecasting network.

    Input:
        x: float tensor with shape (batch, time, features)

    Output:
        logits: tensor with shape (batch, 256)

    This module returns logits. Use cross-entropy loss during training;
    softmax is applied when probabilities are required for inference.
    """

    def __init__(
        self,
        input_features: int = 3,
        channels: int = 32,
        dilations: Sequence[int] = DEFAULT_DILATIONS,
        kernel_size: int = 2,
        dropout: float = 0.0,
        n_classes: int = N_CLASSES,
    ):
        super().__init__()

        if input_features < 1:
            raise ValueError("input_features must be positive")
        if channels < 1:
            raise ValueError("channels must be positive")
        if not dilations or any(d < 1 for d in dilations):
            raise ValueError("dilations must be positive integers")
        if n_classes != N_CLASSES:
            raise ValueError("This reproduction expects 256 output classes")

        self.input_features = input_features
        self.channels = channels
        self.dilations = tuple(dilations)
        self.n_classes = n_classes

        self.input_projection = nn.Conv1d(
            input_features, channels, kernel_size=1
        )

        self.blocks = nn.ModuleList()
        for dilation in self.dilations:
            self.blocks.append(
                GluNetResidualBlock(
                    in_channels=channels,
                    channels=channels,
                    dilation=dilation,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
            )

        self.postprocess = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "Expected input shape (batch, time, features), "
                f"received {tuple(x.shape)}"
            )
        if x.shape[-1] != self.input_features:
            raise ValueError(
                f"Expected {self.input_features} features, "
                f"received {x.shape[-1]}"
            )

        # Convert (B, T, F) to Conv1d layout (B, F, T).
        x = x.transpose(1, 2)
        x = self.input_projection(x)

        skip_outputs = []
        for block in self.blocks:
            x, skip = block(x)
            skip_outputs.append(skip)

        # Parameterized skip paths are learned by each skip projection.
        combined_skip = torch.stack(skip_outputs, dim=0).sum(dim=0)

        # Use the latest valid timestep for the forecast distribution.
        features = self.postprocess(combined_skip)
        last_timestep = features[:, :, -1]

        return self.classifier(last_timestep)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities for inference."""
        return torch.softmax(self.forward(x), dim=-1)

    @torch.no_grad()
    def predict_delta(self, x: torch.Tensor) -> torch.Tensor:
        """Return expected quantized glucose change in mg/dL.

        Uses the expected value of the predicted class distribution.
        This is an explicit inference choice; the paper's exact
        probability-to-point-estimate procedure is not fully specified
        in the implementation details available here.
        """
        probabilities = self.predict_proba(x)
        class_indices = torch.arange(
            self.n_classes,
            device=x.device,
            dtype=probabilities.dtype,
        )
        # Zero-based index i represents paper label i+1.
        delta_values = (class_indices + 1.0) / 10.0 - 12.8

        return (probabilities * delta_values.unsqueeze(0)).sum(dim=-1)

    @torch.no_grad()
    def predict_glucose(
        self,
        x: torch.Tensor,
        current_glucose: torch.Tensor,
    ) -> torch.Tensor:
        """Recover an absolute glucose forecast from current glucose."""
        delta = self.predict_delta(x)
        return current_glucose.reshape(-1).to(delta.device) + delta


def count_parameters(model: nn.Module) -> int:
    """Count trainable model parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
