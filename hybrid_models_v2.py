from __future__ import annotations
import math
import torch
from torch import nn

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)
    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            raise ValueError('Sequence exceeds positional-encoding length.')
        return x + self.pe[:, :x.size(1)]

class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.left = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=self.left, dilation=dilation)
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        r = x
        x = self.conv(x)
        if self.left: x = x[..., :-self.left]
        return r + self.drop(self.act(self.norm(x)))

class TCNBranch(nn.Module):
    def __init__(self, d_model=56, levels=3, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([TCNBlock(d_model, 3, 2**i, dropout) for i in range(levels)])
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        x = x.transpose(1, 2)
        for b in self.blocks: x = b(x)
        return self.norm(x[:, :, -1])

class GRUBranch(nn.Module):
    def __init__(self, d_model=56, hidden_size=56, layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(d_model, hidden_size, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.proj = nn.Sequential(nn.Linear(hidden_size, d_model), nn.LayerNorm(d_model), nn.GELU())
    def forward(self, x):
        y, _ = self.gru(x)
        return self.proj(y[:, -1])

class TransformerBranch(nn.Module):
    def __init__(self, d_model=56, heads=2, layers=1, dropout=0.1, max_len=512):
        super().__init__()
        self.pos = SinusoidalPositionalEncoding(d_model, max_len)
        enc = nn.TransformerEncoderLayer(d_model, heads, 4*d_model, dropout, activation='gelu', batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(d_model)
        self.pool = nn.Sequential(nn.Linear(d_model, d_model//2), nn.Tanh(), nn.Linear(d_model//2, 1))
    def forward(self, x):
        x = self.norm(self.encoder(self.pos(x)))
        w = torch.softmax(self.pool(x), dim=1)
        return (x*w).sum(dim=1)

class HorizonAwareResidualFusion(nn.Module):
    def __init__(self, d_model=56, horizon_dim=16, n_horizons=5, dropout=0.1):
        super().__init__()
        self.horizon = nn.Embedding(n_horizons, horizon_dim)
        self.gate = nn.Sequential(nn.Linear(3*d_model+horizon_dim, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 2))
        self.tcn_delta = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout))
        self.tr_delta = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)
        self.tcn_scale = nn.Parameter(torch.tensor(0.1))
        self.tr_scale = nn.Parameter(torch.tensor(0.1))
    def forward(self, tcn, gru, tr, hidx):
        h = self.horizon(hidx)
        gates = torch.sigmoid(self.gate(torch.cat([tcn, gru, tr, h], dim=-1)))
        fused = gru + self.tcn_scale*gates[:, :1]*self.tcn_delta(tcn) + self.tr_scale*gates[:, 1:]*self.tr_delta(tr)
        return self.norm(fused), gates

class HorizonAwareResidualHybrid(nn.Module):
    def __init__(self, input_dim, d_model=56, gru_hidden=56, gru_layers=2, tcn_levels=3, transformer_heads=2, transformer_layers=1, horizons=(15,30,60,90,120), dropout=0.1, max_len=512, horizon_embedding_dim=16):
        super().__init__()
        self.horizons = tuple(horizons)
        self.embedding = nn.Sequential(nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.tcn = TCNBranch(d_model, tcn_levels, dropout)
        self.gru = GRUBranch(d_model, gru_hidden, gru_layers, dropout)
        self.transformer = TransformerBranch(d_model, transformer_heads, transformer_layers, dropout, max_len)
        self.fusion = HorizonAwareResidualFusion(d_model, horizon_embedding_dim, len(self.horizons), dropout)
        self.heads = nn.ModuleDict({str(h): nn.Sequential(nn.Linear(d_model,d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model,1)) for h in self.horizons})
        self.hmap = {h:i for i,h in enumerate(self.horizons)}
    def forward(self, x, return_gates=False):
        x = self.embedding(x)
        tcn, gru, tr = self.tcn(x), self.gru(x), self.transformer(x)
        preds, gates = [], []
        for h in self.horizons:
            idx = torch.full((x.size(0),), self.hmap[h], dtype=torch.long, device=x.device)
            fused, g = self.fusion(tcn, gru, tr, idx)
            preds.append(self.heads[str(h)](fused))
            gates.append(g)
        pred = torch.cat(preds, dim=-1)
        gate = torch.stack(gates, dim=1)
        return (pred, gate) if return_gates else pred

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
