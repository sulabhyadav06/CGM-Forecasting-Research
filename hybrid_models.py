from __future__ import annotations

import math
import torch
from torch import nn


# ============================================================
# Positional Encoding
# ============================================================

class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.

    Input:
        x: [batch, sequence_length, d_model]
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(
                position * div_term[: pe[:, 1::2].shape[1]]
            )

        self.register_buffer(
            "pe",
            pe.unsqueeze(0),
            persistent=False,
        )

    def forward(self, x):
        seq_len = x.size(1)

        if seq_len > self.pe.size(1):
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"maximum positional encoding length {self.pe.size(1)}"
            )

        return x + self.pe[:, :seq_len]


# ============================================================
# TCN Branch
# ============================================================

class TCNBlock(nn.Module):
    """
    Residual causal TCN block.

    Input/output:
        [B, C, T]
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.left_padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.left_padding,
        )

        self.norm = nn.BatchNorm1d(channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.conv(x)

        # Remove future/right-side positions introduced
        # by symmetric PyTorch padding.
        if self.left_padding > 0:
            x = x[..., :-self.left_padding]

        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x + residual


class TCNBranch(nn.Module):
    """
    Multi-scale temporal convolution branch.

    Role:
        Capture local and short-to-medium temporal glucose dynamics.
    """

    def __init__(
        self,
        d_model: int = 56,
        levels: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                TCNBlock(
                    channels=d_model,
                    kernel_size=3,
                    dilation=2 ** i,
                    dropout=dropout,
                )
                for i in range(levels)
            ]
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x:
            [B, T, D]

        returns:
            [B, D]
        """

        # [B,T,D] -> [B,D,T]
        x = x.transpose(1, 2)

        for block in self.blocks:
            x = block(x)

        # Final temporal representation
        x = x[:, :, -1]

        # [B,D]
        x = self.norm(x)

        return x


# ============================================================
# GRU Branch
# ============================================================

class GRUBranch(nn.Module):
    """
    GRU branch.

    Role:
        Capture sequential physiological dynamics.
    """

    def __init__(
        self,
        d_model: int = 56,
        hidden_size: int = 56,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, x):
        """
        x:
            [B,T,D]

        returns:
            [B,D]
        """

        output, _ = self.gru(x)

        # Last hidden representation
        x = output[:, -1, :]

        return self.projection(x)


# ============================================================
# Improved Transformer Branch
# ============================================================

class TransformerBranch(nn.Module):
    """
    Improved Transformer branch.

    Role:
        Capture longer-range temporal relationships
        and interactions across modalities.

    Uses:
        - sinusoidal positional encoding
        - Transformer encoder
        - attention pooling
    """

    def __init__(
        self,
        d_model: int = 56,
        heads: int = 2,
        layers: int = 1,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()

        self.position = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=max_len,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
        )

        self.norm = nn.LayerNorm(d_model)

        # Learn which timesteps matter most
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        """
        x:
            [B,T,D]

        returns:
            [B,D]
        """

        x = self.position(x)

        x = self.encoder(x)

        x = self.norm(x)

        attention_scores = self.attention_pool(x)

        weights = torch.softmax(
            attention_scores,
            dim=1,
        )

        x = (x * weights).sum(dim=1)

        return x


# ============================================================
# Adaptive Gated Fusion
# ============================================================

class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion of TCN, GRU and Transformer representations.

    Instead of:
        fused = concat(TCN, GRU, Transformer)

    we learn:
        alpha_TCN
        alpha_GRU
        alpha_Transformer

    with:
        alpha_1 + alpha_2 + alpha_3 = 1
    """

    def __init__(
        self,
        d_model: int = 56,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )

        # Learnable branch projections
        self.tcn_projection = nn.Linear(
            d_model,
            d_model,
        )

        self.gru_projection = nn.Linear(
            d_model,
            d_model,
        )

        self.transformer_projection = nn.Linear(
            d_model,
            d_model,
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tcn_repr,
        gru_repr,
        transformer_repr,
    ):

        tcn_repr = self.tcn_projection(tcn_repr)

        gru_repr = self.gru_projection(gru_repr)

        transformer_repr = self.transformer_projection(
            transformer_repr
        )

        combined = torch.cat(
            [
                tcn_repr,
                gru_repr,
                transformer_repr,
            ],
            dim=-1,
        )

        gate_logits = self.gate(combined)

        gate_weights = torch.softmax(
            gate_logits,
            dim=-1,
        )

        fused = (
            gate_weights[:, 0:1] * tcn_repr
            + gate_weights[:, 1:2] * gru_repr
            + gate_weights[:, 2:3] * transformer_repr
        )

        fused = self.norm(fused)

        return fused, gate_weights


# ============================================================
# Multi-Horizon Prediction Head
# ============================================================

class MultiHorizonHead(nn.Module):
    """
    Shared representation followed by separate
    prediction heads for each forecast horizon.
    """

    def __init__(
        self,
        d_model: int = 56,
        horizons=(15, 30, 60, 90, 120),
        dropout: float = 0.1,
    ):
        super().__init__()

        self.horizons = tuple(horizons)

        self.shared = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.heads = nn.ModuleDict(
            {
                str(h): nn.Linear(d_model, 1)
                for h in self.horizons
            }
        )

    def forward(self, x):

        x = self.shared(x)

        predictions = []

        for horizon in self.horizons:

            predictions.append(
                self.heads[str(horizon)](x)
            )

        return torch.cat(
            predictions,
            dim=-1,
        )


# ============================================================
# Complete Hybrid Model
# ============================================================

class HybridTCNGRUTransformer(nn.Module):
    """
    Proposed Phase-2 hybrid architecture.

    Architecture:

        Multimodal input
              |
        Feature embedding
              |
        +------+------+------+
        |      |             |
       TCN    GRU     Transformer
        |      |             |
        +------+------+------+
               |
        Adaptive gated fusion
               |
        Shared representation
               |
        +------+------+------+------+
        |      |      |      |      |
       15     30     60     90     120 min
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 56,
        gru_hidden: int = 56,
        gru_layers: int = 2,
        tcn_levels: int = 3,
        transformer_heads: int = 2,
        transformer_layers: int = 1,
        horizons=(15, 30, 60, 90, 120),
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()

        self.horizons = tuple(horizons)

        # ----------------------------------------------------
        # Feature embedding
        # ----------------------------------------------------

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # ----------------------------------------------------
        # Three complementary branches
        # ----------------------------------------------------

        self.tcn = TCNBranch(
            d_model=d_model,
            levels=tcn_levels,
            dropout=dropout,
        )

        self.gru = GRUBranch(
            d_model=d_model,
            hidden_size=gru_hidden,
            layers=gru_layers,
            dropout=dropout,
        )

        self.transformer = TransformerBranch(
            d_model=d_model,
            heads=transformer_heads,
            layers=transformer_layers,
            dropout=dropout,
            max_len=max_len,
        )

        # ----------------------------------------------------
        # Adaptive fusion
        # ----------------------------------------------------

        self.fusion = AdaptiveFusion(
            d_model=d_model,
            dropout=dropout,
        )

        # ----------------------------------------------------
        # Multi-horizon prediction
        # ----------------------------------------------------

        self.head = MultiHorizonHead(
            d_model=d_model,
            horizons=horizons,
            dropout=dropout,
        )

    def forward(
        self,
        x,
        return_gates=False,
    ):

        # x = [B,T,input_dim]
        x = self.embedding(x)

        # Three representations
        tcn_repr = self.tcn(x)

        gru_repr = self.gru(x)

        transformer_repr = self.transformer(x)

        # Adaptive fusion
        fused, gates = self.fusion(
            tcn_repr,
            gru_repr,
            transformer_repr,
        )

        # Multi-horizon prediction
        prediction = self.head(fused)

        if return_gates:
            return prediction, gates

        return prediction


# ============================================================
# Utility
# ============================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )