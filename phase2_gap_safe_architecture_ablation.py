
"""
Phase-2 gap-safe architecture/ablation comparison.

Runs all required architectures on all 12 OhioT1DM patients using the
Phase-2 multimodal data, identical preprocessing, sequence rules, split,
optimization, and horizons.

Architectures:
LSTM, GRU, BiLSTM, TCN, Transformer,
TCN-GRU, GRU-Transformer, TCN-Transformer,
TCN-GRU-Transformer, Adaptive-Hybrid.

The hybrid ablations are trained jointly for all horizons with the existing
AblationModel; the five classical architectures are trained separately per
horizon. Results are saved incrementally.
"""
from __future__ import annotations
import argparse, copy, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from phase2_sequence_utils import build_sequences
from hybrid_ablation_experiments import AblationModel
from hybrid_models import TCNBranch, TransformerBranch

FEATURES = {
    "ohio2018": [
        "glucose","carbs_last_60min","bolus_last_60min","basal_rate",
        "heart_rate_mean_5m","heart_rate_std_5m","heart_rate_min_5m",
        "heart_rate_max_5m","heart_rate_count_5m","heart_rate_coverage_5m",
        "heart_rate_missing_5m","steps_mean_5m","steps_std_5m",
        "steps_min_5m","steps_max_5m","steps_count_5m","steps_coverage_5m",
        "steps_missing_5m"],
    "ohio2020": [
        "glucose","carbs_last_60min","bolus_last_60min","basal_rate",
        "accel_mean_5m","accel_std_5m","accel_min_5m","accel_max_5m",
        "accel_count_5m","accel_coverage_5m","accel_missing_5m"],
}
PATIENTS = {
    "ohio2018":["559","563","570","575","588","591"],
    "ohio2020":["540","544","552","567","584","596"],
}
CLASSICAL = ["LSTM","GRU","BiLSTM","TCN","Transformer"]
HYBRIDS = ["tcn_gru","gru_transformer","tcn_transformer",
           "tcn_gru_transformer","adaptive_tcn_gru_transformer"]

class Standardizer:
    def __init__(self): self.med={}; self.mean={}; self.std={}
    def fit(self, df, cols):
        for c in cols:
            s=pd.to_numeric(df[c],errors="coerce")
            m=float(s.median()) if s.notna().any() else 0.
            z=s.fillna(m); mu=float(z.mean()); sd=float(z.std(ddof=0))
            if not np.isfinite(sd) or sd<1e-8: sd=1.
            self.med[c]=m; self.mean[c]=mu; self.std[c]=sd
        return self
    def transform(self, df, cols):
        out=df.copy()
        for c in cols:
            s=pd.to_numeric(out[c],errors="coerce").fillna(self.med[c])
            out[c]=((s-self.mean[c])/self.std[c]).astype(np.float32)
        return out

class Seq(Dataset):
    def __init__(self,X,Y):
        self.X=np.asarray(X,np.float32); self.Y=np.asarray(Y,np.float32)
    def __len__(self): return len(self.Y)
    def __getitem__(self,i):
        return torch.from_numpy(self.X[i]),torch.from_numpy(self.Y[i])

class SingleHead(nn.Module):
    def __init__(self, kind, input_dim, output_dim, hidden=64, layers=2, dropout=.1,
                 tf_heads=2, tf_layers=1):
        super().__init__(); self.kind=kind
        self.emb=nn.Sequential(nn.Linear(input_dim,hidden),nn.LayerNorm(hidden),nn.GELU())
        if kind in ("LSTM","GRU","BiLSTM"):
            rnn=nn.LSTM if kind!="GRU" else nn.GRU
            h=hidden//2 if kind=="BiLSTM" else hidden
            self.rnn=rnn(hidden,h,num_layers=layers,batch_first=True,
                         dropout=dropout if layers>1 else 0.,
                         bidirectional=(kind=="BiLSTM"))
            out=h*(2 if kind=="BiLSTM" else 1)
            self.proj=nn.Sequential(nn.Linear(out,hidden),nn.LayerNorm(hidden),nn.GELU())
        elif kind=="TCN":
            self.branch=TCNBranch(d_model=hidden,levels=layers,dropout=dropout)
            self.proj=nn.Identity()
        elif kind=="Transformer":
            self.branch=TransformerBranch(d_model=hidden,heads=tf_heads,
                                          layers=tf_layers,dropout=dropout,max_len=512)
            self.proj=nn.Identity()
        self.head=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),
                                nn.Dropout(dropout),nn.Linear(hidden,output_dim))
    def forward(self,x):
        x=self.emb(x)
        if self.kind in ("LSTM","GRU","BiLSTM"):
            z,_=self.rnn(x); z=self.proj(z[:,-1])
        else:
            z=self.proj(self.branch(x))
        return self.head(z)

def seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def gap_sequences(df, features, lookback, horizons, scaler, target_mean, target_std):
    work=scaler.transform(df,features)
    raw=pd.to_numeric(df["glucose"],errors="coerce")
    work["glucose"]=((raw-target_mean)/target_std).astype(np.float32)
    # Build one max-horizon gap-safe sequence and collect all targets.
    work=work.copy()
    ts=pd.to_datetime(work["timestamp"],utc=True).dt.tz_localize(None)
    ts=ts.to_numpy(dtype="datetime64[ns]")
    X=work[features].to_numpy(np.float32); y=work["glucose"].to_numpy(np.float32)
    L=lookback//5; hs=[h//5 for h in horizons]; maxh=max(hs)
    xs=[]; ys=[]
    expected=np.timedelta64(5,"m")
    for end in range(L-1,len(work)-maxh):
        a=end-L+1; b=end+maxh
        its=ts[a:end+1]; fts=ts[end:b+1]
        if len(its)!=L or len(fts)!=maxh+1: continue
        if not np.all(np.diff(its)==expected) or not np.all(np.diff(fts)==expected): continue
        xx=X[a:end+1]; yy=[y[end+h] for h in hs]
        if np.isfinite(xx).all() and np.isfinite(yy).all():
            xs.append(xx); ys.append(yy)
    return Seq(xs,ys)

def split(ds, frac=.15, purge=0):
    n=len(ds); cut=int(n*(1-frac)); tr=max(1,cut-purge)
    return Seq(ds.X[:tr],ds.Y[:tr]),Seq(ds.X[cut:],ds.Y[cut:])

def train(model,tr,va,args,device):
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    best=float("inf"); state=None; bad=0
    lossfn=nn.SmoothL1Loss()
    for ep in range(1,args.epochs+1):
        model.train(); total=0.; n=0
        for x,y in DataLoader(tr,args.batch_size,shuffle=True):
            x=x.to(device); y=y.to(device); opt.zero_grad(set_to_none=True)
            p=model(x); loss=lossfn(p,y)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip); opt.step()
            total+=loss.item()*len(y); n+=len(y)
        model.eval(); vl=0.; vn=0
        with torch.no_grad():
            for x,y in DataLoader(va,args.batch_size):
                p=model(x.to(device)); loss=lossfn(p,y.to(device)); vl+=loss.item()*len(y); vn+=len(y)
        vl/=max(vn,1)
        print(f"    epoch {ep:02d} val={vl:.5f}")
        if vl<best-1e-7:
            best=vl; state=copy.deepcopy(model.state_dict()); bad=0
        else:
            bad+=1
            if bad>=args.patience: break
    model.load_state_dict(state)
    return model

@torch.no_grad()
def predict(model,ds,args,device):
    model.eval(); out=[]; y=[]
    for x,t in DataLoader(ds,args.batch_size):
        out.append(model(x.to(device)).cpu().numpy()); y.append(t.numpy())
    return np.concatenate(out),np.concatenate(y)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-root",type=Path,default=Path("data/phase2"))
    ap.add_argument("--lookback",type=int,default=120)
    ap.add_argument("--horizons",type=int,nargs="+",default=[15,30,60,90,120])
    ap.add_argument("--epochs",type=int,default=15); ap.add_argument("--patience",type=int,default=3)
    ap.add_argument("--batch-size",type=int,default=128); ap.add_argument("--hidden",type=int,default=64)
    ap.add_argument("--layers",type=int,default=2); ap.add_argument("--dropout",type=float,default=.1)
    ap.add_argument("--lr",type=float,default=1e-3); ap.add_argument("--weight-decay",type=float,default=1e-4)
    ap.add_argument("--grad-clip",type=float,default=1.); ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--output-dir",type=Path,default=Path("output/phase2_gap_safe_architecture"))
    args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); print("Device:",device)
    rows=[]
    for cohort,patients in PATIENTS.items():
        feats=FEATURES[cohort]
        for patient in patients:
            print("\n"+"="*78); print(cohort,patient); print("="*78)
            trdf=pd.read_csv(args.data_root/cohort/"train"/f"{patient}.csv",parse_dates=["timestamp"])
            tedf=pd.read_csv(args.data_root/cohort/"test"/f"{patient}.csv",parse_dates=["timestamp"])
            # Chronological validation split first; all imputation/scaling statistics
            # are fit only on the training partition.
            trdf=trdf.sort_values("timestamp").reset_index(drop=True)
            cut=max(1,int(len(trdf)*0.85))
            fit_cut=max(1,cut-max(args.horizons)//5)
            fit_df=trdf.iloc[:fit_cut].copy()
            scaler=Standardizer().fit(fit_df,feats)
            glucose=pd.to_numeric(fit_df.glucose,errors="coerce")
            tm=float(glucose.mean()); ts=float(glucose.std(ddof=0))
            if not np.isfinite(ts) or ts<1e-8: ts=1.
            trseq=gap_sequences(trdf,feats,args.lookback,args.horizons,scaler,tm,ts)
            teseq=gap_sequences(tedf,feats,args.lookback,args.horizons,scaler,tm,ts)
            purge=args.lookback//5+max(args.horizons)//5
            train_ds,val_ds=split(trseq,.15,purge)
            print("sequences:",len(train_ds),len(val_ds),len(teseq))
            for name in CLASSICAL:
                seed(args.seed)
                model=SingleHead(name,len(feats),len(args.horizons),args.hidden,args.layers,args.dropout).to(device)
                params=sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"  {name} joint multi-horizon params={params:,}")
                model=train(model,train_ds,val_ds,args,device)
                p,y=predict(model,teseq,args,device)
                for hi,h in enumerate(args.horizons):
                    pp=p[:,hi]*ts+tm; yy=y[:,hi]*ts+tm; e=pp-yy
                    rows.append(dict(cohort=cohort,patient=patient,architecture=name,
                        horizon_min=h,parameters=params,mae_mgdl=float(np.mean(abs(e))),
                        rmse_mgdl=float(np.sqrt(np.mean(e**2))),n_test=len(yy)))
                pd.DataFrame(rows).to_csv(args.output_dir/"all_results.csv",index=False)
            # Hybrid ablations: joint multi-horizon.
            for name in HYBRIDS:
                seed(args.seed)
                model=AblationModel(input_dim=len(feats),architecture=name,d_model=56,
                    gru_hidden=56,gru_layers=2,tcn_levels=3,transformer_heads=2,
                    transformer_layers=1,horizons=tuple(args.horizons),dropout=.1,
                    max_len=args.lookback//5).to(device)
                params=sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"  {name} joint params={params:,}")
                # custom joint trainer
                opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
                best=float("inf"); state=None; bad=0
                for ep in range(1,args.epochs+1):
                    model.train(); total=0.; n=0
                    for x,y in DataLoader(train_ds,args.batch_size,shuffle=True):
                        x=x.to(device); y=y.to(device); opt.zero_grad(set_to_none=True)
                        p=model(x); loss=nn.SmoothL1Loss()(p,y); loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip); opt.step()
                        total+=loss.item()*len(y); n+=len(y)
                    model.eval(); vl=0.; vn=0
                    with torch.no_grad():
                        for x,y in DataLoader(val_ds,args.batch_size):
                            l=nn.SmoothL1Loss()(model(x.to(device)),y.to(device)); vl+=l.item()*len(y); vn+=len(y)
                    vl/=max(vn,1); print(f"    epoch {ep:02d} val={vl:.5f}")
                    if vl<best-1e-7: best=vl; state=copy.deepcopy(model.state_dict()); bad=0
                    else:
                        bad+=1
                        if bad>=args.patience: break
                model.load_state_dict(state)
                p,y=predict(model,teseq,args,device)
                for hi,h in enumerate(args.horizons):
                    pp=p[:,hi]*ts+tm; yy=y[:,hi]*ts+tm; e=pp-yy
                    rows.append(dict(cohort=cohort,patient=patient,architecture=name,
                        horizon_min=h,parameters=params,mae_mgdl=float(np.mean(abs(e))),
                        rmse_mgdl=float(np.sqrt(np.mean(e**2))),n_test=len(yy)))
                pd.DataFrame(rows).to_csv(args.output_dir/"all_results.csv",index=False)
    df=pd.DataFrame(rows); df.to_csv(args.output_dir/"all_results.csv",index=False)
    summary=df.groupby(["architecture","horizon_min"])[["mae_mgdl","rmse_mgdl"]].agg(["mean","std","count"]).reset_index()
    summary.to_csv(args.output_dir/"architecture_mean_sd.csv",index=False)
    print("\nFINAL ARCHITECTURE RESULTS"); print(summary.to_string(index=False))

if __name__=="__main__": main()
