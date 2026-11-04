"""Baseline sequence models for the CRNN benchmark pipeline.

All models take (batch, 24, n_features) and return (batch, 1): the predicted
glucose CHANGE, exactly like CRNN, so they run through the same data, split,
normalisation, training loop and saved-prediction code. Inputs only contain
samples up to the forecast origin (no future samples).
"""
import torch
from torch import nn

from experiments.crnn.crnn import CRNN


class _Head(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(width, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


class RecurrentBaseline(nn.Module):
    def __init__(self, n_features, kind="lstm", hidden=64,
                 bidirectional=False, dropout=0.2):
        super().__init__()
        rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[kind]
        self.bi = bidirectional
        self.hidden = hidden
        self.rnn = rnn(n_features, hidden, batch_first=True,
                       bidirectional=bidirectional)
        self.head = _Head(hidden * (2 if bidirectional else 1), dropout)

    def forward(self, x):
        out, _ = self.rnn(x)
        if self.bi:
            feat = torch.cat([out[:, -1, :self.hidden], out[:, 0, self.hidden:]], dim=1)
        else:
            feat = out[:, -1, :]
        return self.head(feat)


class _CausalBlock(nn.Module):
    def __init__(self, cin, cout, dilation, dropout, k=3):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv1 = nn.Conv1d(cin, cout, k, dilation=dilation)
        self.conv2 = nn.Conv1d(cout, cout, k, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        y = nn.functional.pad(x, (self.pad, 0))
        y = self.drop(self.act(self.conv1(y)))
        y = nn.functional.pad(y, (self.pad, 0))
        y = self.drop(self.act(self.conv2(y)))
        return self.act(y + self.skip(x))


class TCNBaseline(nn.Module):
    def __init__(self, n_features, channels=32, dilations=(1, 2, 4, 8), dropout=0.2):
        super().__init__()
        blocks, cin = [], n_features
        for d in dilations:
            blocks.append(_CausalBlock(cin, channels, d, dropout))
            cin = channels
        self.blocks = nn.Sequential(*blocks)
        self.head = _Head(channels, dropout)

    def forward(self, x):
        h = self.blocks(x.transpose(1, 2))
        return self.head(h[:, :, -1])


class TransformerBaseline(nn.Module):
    def __init__(self, n_features, steps=24, d_model=32, heads=4,
                 layers=2, ff=64, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, steps, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = _Head(d_model, dropout)

    def forward(self, x):
        h = self.enc(self.proj(x) + self.pos)
        return self.head(h[:, -1, :])


def build_model(name, n_features, dropout):
    name = name.lower()
    if name == "crnn":
        return CRNN(n_features=n_features, dropout=dropout)
    if name == "gru":
        return RecurrentBaseline(n_features, "gru", dropout=dropout)
    if name == "lstm":
        return RecurrentBaseline(n_features, "lstm", dropout=dropout)
    if name == "bilstm":
        return RecurrentBaseline(n_features, "lstm", bidirectional=True, dropout=dropout)
    if name == "tcn":
        return TCNBaseline(n_features, dropout=dropout)
    if name == "transformer":
        return TransformerBaseline(n_features, dropout=dropout)
    raise ValueError(f"Unknown model: {name}")
