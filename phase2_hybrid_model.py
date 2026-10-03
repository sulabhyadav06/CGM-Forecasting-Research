
from __future__ import annotations
import torch
from torch import nn
from hybrid_models import TCNBranch, GRUBranch, TransformerBranch, AdaptiveFusion, MultiHorizonHead

class PersonalizedHybridTCNGRUTransformer(nn.Module):
    """
    Professor-specified Phase-2 model:
    cohort-specific feature projection -> shared TCN/GRU/Transformer
    -> patient-conditioned adaptive fusion -> multi-horizon heads.
    """
    def __init__(self, input_dims, n_patients=12, d_model=64,
                 gru_hidden=96, gru_layers=2, tcn_levels=3,
                 transformer_heads=4, transformer_layers=1,
                 horizons=(15,30,60,90,120), dropout=0.1,
                 max_len=512):
        super().__init__()
        self.horizons = tuple(horizons)

        self.projections = nn.ModuleDict({
            str(k): nn.Sequential(
                nn.Linear(v, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )
            for k, v in input_dims.items()
        })

        self.patient_embedding = nn.Embedding(n_patients, d_model)

        self.tcn = TCNBranch(d_model=d_model, levels=tcn_levels, dropout=dropout)
        self.gru = GRUBranch(d_model=d_model, hidden_size=gru_hidden,
                             layers=gru_layers, dropout=dropout)
        self.transformer = TransformerBranch(
            d_model=d_model, heads=transformer_heads,
            layers=transformer_layers, dropout=dropout,
            max_len=max_len,
        )

        self.fusion = AdaptiveFusion(d_model=d_model, dropout=dropout)

        self.personalize = nn.Sequential(
            nn.Linear(2*d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.head = MultiHorizonHead(
            d_model=d_model, horizons=horizons, dropout=dropout
        )

    def forward(self, x, patient_id, cohort_id, return_gates=False):
        # x is [B,T,F] and cohort_id selects the appropriate input projection.
        projected = torch.empty(
            x.size(0), x.size(1), self.patient_embedding.embedding_dim,
            device=x.device, dtype=x.dtype
        )
        for cohort in torch.unique(cohort_id):
            mask = cohort_id == cohort
            projected[mask] = self.projections[str(int(cohort.item()))](x[mask])

        tcn = self.tcn(projected)
        gru = self.gru(projected)
        transformer = self.transformer(projected)

        fused, gates = self.fusion(tcn, gru, transformer)
        p = self.patient_embedding(patient_id)
        fused = self.personalize(torch.cat([fused, p], dim=-1))
        prediction = self.head(fused)

        if return_gates:
            return prediction, gates
        return prediction

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
