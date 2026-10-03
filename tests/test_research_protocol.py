"""Lightweight protocol regression tests for CGM Forecasting Research."""
import numpy as np
import pandas as pd
import torch
from phase2_sequence_utils import build_sequences
from clinical_metrics import time_lag_minutes
from hybrid_ablation_experiments import AblationModel


def test_horizons_and_gap_safe_sequences():
    n=240
    ts=pd.date_range("2026-01-01",periods=n,freq="5min").to_numpy(dtype="datetime64[ns]")
    ts[150:]+=np.timedelta64(25,"m")
    g=np.arange(n,dtype=float)+100
    df=pd.DataFrame({"timestamp":ts,"glucose":g,"x":g/10})
    X,y,meta=build_sequences(df,["x"],"glucose",24,[3,6,12,18,24],5,"timestamp")
    assert len(X)==len(y)==len(meta)>0 and y.shape[1]==5
    assert meta.segment_id.nunique()==2
    for _,r in meta.iterrows():
        for h,t in zip([15,30,60,90,120],r.target_times):
            assert (pd.Timestamp(t)-pd.Timestamp(r.input_end)).total_seconds()/60==h


def test_known_signed_lag():
    rng=np.random.default_rng(42); truth=rng.normal(size=500)
    pred=np.r_[np.zeros(3),truth[:-3]]
    assert time_lag_minutes(truth,pred,5,8)==15


def test_all_architectures_have_five_outputs_and_adaptive_gates_sum_to_one():
    names=["gru","tcn","transformer","tcn_gru","gru_transformer","tcn_transformer","tcn_gru_transformer","adaptive_tcn_gru_transformer"]
    x=torch.randn(2,12,5)
    for name in names:
        m=AblationModel(input_dim=5,architecture=name,d_model=16,gru_hidden=16,gru_layers=1,tcn_levels=2,transformer_heads=2,transformer_layers=1,horizons=(15,30,60,90,120),max_len=12)
        assert m(x).shape==(2,5)
        if name=="adaptive_tcn_gru_transformer":
            _,g=m(x,return_gates=True)
            assert g.shape==(2,3)
            assert torch.allclose(g.sum(-1),torch.ones(2),atol=1e-6)
