"""Lightweight protocol regression tests for CGM Forecasting Research."""
import numpy as np
import pandas as pd
import torch
import xml.etree.ElementTree as ET
from parse_xml import parse_bolus
from ohio2020_acceleration_preprocessing import aggregate_acceleration_to_cgm
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


def test_bolus_parser_uses_ts_begin():
    root=ET.fromstring('<patient><bolus><event ts_begin="01-01-2026 12:00:00" dose="1.25" /><event ts_begin="01-01-2026 12:10:00" dose="2.0" /></bolus></patient>')
    result=parse_bolus(root)
    assert len(result)==2 and np.isclose(result.dose.sum(),3.25)


def test_acceleration_aggregation_excludes_endpoint_future_event():
    cgm=pd.DataFrame({"timestamp":pd.to_datetime(["2026-01-01 00:05:00","2026-01-01 00:10:00"]),"glucose":[100,110]})
    accel=pd.DataFrame({"timestamp":pd.to_datetime(["2026-01-01 00:01:00","2026-01-01 00:02:00","2026-01-01 00:05:00","2026-01-01 00:06:00"]),"value":[1.,3.,99.,5.]})
    out=aggregate_acceleration_to_cgm(cgm,accel)
    assert out.loc[0,"accel_mean_5m"]==2.0
    assert out.loc[1,"accel_mean_5m"]==52.0


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
