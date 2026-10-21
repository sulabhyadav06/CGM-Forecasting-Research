"""GluNet source-aligned reconstruction candidate; not author-code-identical."""
from __future__ import annotations
from typing import Sequence
import torch
from torch import nn
import torch.nn.functional as F

N_CLASSES = 256
DILATIONS = (1, 2, 4, 8)
LAYER_CHANNELS = (32, 32, 64, 64)


class CausalConv1d(nn.Module):
    """Left-padded temporal convolution; output length equals input length."""
    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=1, bias=True):
        super().__init__()
        if kernel_size < 1 or dilation < 1:
            raise ValueError("kernel_size and dilation must be positive")
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(F.pad(x, (self.left_padding, 0)))


class GatedDilatedLayer(nn.Module):
    """Tanh/sigmoid gated causal convolution with residual and learned skip."""
    def __init__(self, in_channels, out_channels, skip_channels, dilation,
                 kernel_size=2, dropout=0.0):
        super().__init__()
        self.filter_conv = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.gate_conv = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.residual_projection = (nn.Identity() if in_channels == out_channels
                                    else nn.Conv1d(in_channels, out_channels, 1))
        self.skip_projection = nn.Conv1d(out_channels, skip_channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gated = torch.tanh(self.filter_conv(x)) * torch.sigmoid(self.gate_conv(x))
        gated = self.dropout(gated)
        return self.residual_projection(x) + gated, self.skip_projection(gated)


class GluNetV2(nn.Module):
    """Five temporal layers + probabilistic 256-class output head.

    forward() returns logits. Apply softmax only for inference.
    """
    def __init__(self, input_features=3, lookback=16,
                 dilations: Sequence[int] = DILATIONS,
                 layer_channels: Sequence[int] = LAYER_CHANNELS,
                 n_classes=256, dropout=0.0):
        super().__init__()
        if input_features < 1 or lookback < 1:
            raise ValueError("input_features and lookback must be positive")
        if tuple(dilations) != DILATIONS:
            raise ValueError(f"Expected dilations {DILATIONS}")
        if tuple(layer_channels) != LAYER_CHANNELS:
            raise ValueError(f"Expected layer channels {LAYER_CHANNELS}")
        if n_classes != N_CLASSES:
            raise ValueError("Expected 256 classes")
        self.input_features, self.lookback, self.n_classes = input_features, lookback, n_classes
        # Distinct initial temporal causal convolution; not a 1x1 projection.
        self.initial_causal_conv = CausalConv1d(input_features, 32, kernel_size=2)
        blocks, in_channels = [], 32
        for dilation, out_channels in zip(dilations, layer_channels):
            blocks.append(GatedDilatedLayer(in_channels, out_channels, 64,
                                            dilation, kernel_size=2, dropout=dropout))
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.postprocess_conv1 = nn.Conv1d(64, 64, kernel_size=1)
        self.postprocess_conv2 = nn.Conv1d(64, n_classes, kernel_size=1)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(f"Expected (batch,time,features), got {tuple(x.shape)}")
        if x.shape[1] != self.lookback:
            raise ValueError(f"Expected lookback {self.lookback}, got {x.shape[1]}")
        if x.shape[2] != self.input_features:
            raise ValueError(f"Expected {self.input_features} features, got {x.shape[2]}")
        x = self.initial_causal_conv(x.transpose(1, 2))
        skip_sum = None
        for block in self.blocks:
            x, skip = block(x)
            skip_sum = skip if skip_sum is None else skip_sum + skip
        z = F.relu(skip_sum)
        z = F.relu(self.postprocess_conv1(z))
        return self.postprocess_conv2(z)[:, :, -1]

    @torch.no_grad()
    def predict_proba(self, x):
        return torch.softmax(self.forward(x), dim=-1)
