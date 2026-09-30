# Joint 6-patient matched-GRU validation runner.
# Same 85/15 chronological split, 120-min lookback, horizons 15/30/60/90/120,
# patient-specific train-only scaling, seed, batch size, optimizer and early stopping
# as Full-v5. Does not load held-out test files.

import argparse, json, random
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

FEATURES = [
    "glucose","carbs_last_60min","bolus_last_60min","basal_rate",
    "heart_rate_mean_5m","heart_rate_std_5m","heart_rate_min_5m",
    "heart_rate_max_5m","heart_rate_count_5m","heart_rate_coverage_5m",
    "heart_rate_missing_5m","steps_mean_5m","steps_std_5m",
    "steps_min_5m","steps_max_5m","steps_count_5m",
    "steps_coverage_5m","steps_missing_5m"]
PATIENTS=[559,563,570,575,588,591]
HORIZONS=[15,30,60,90,120]
STEP=5

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def read_csv(path):
    d=pd.read_csv(path)
    d["timestamp"]=pd.to_datetime(d["timestamp"],errors="coerce")
    d=d.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    for c in FEATURES: d[c]=pd.to_numeric(d[c],errors="coerce")
    return d.dropna(subset=FEATURES).reset_index(drop=True)

def seqs(d, lookback, horizons):
    x=d[FEATURES].to_numpy(np.float32); y=d.glucose.to_numpy(np.float32)
    ts=d.timestamp.to_numpy(); L=lookback//STEP; H=[h//STEP for h in horizons]
    xs,ys=[],[]
    for end in range(L-1,len(d)-max(H)):
        st=end-L+1
        dt=np.diff(ts[st:end+max(H)+1]).astype("timedelta64[s]").astype(np.int64)
        if len(dt) and np.any(dt != STEP*60): continue
        xs.append(x[st:end+1]); ys.append([y[end+h] for h in H])
    return np.asarray(xs,np.float32),np.asarray(ys,np.float32)

class DS(Dataset):
    def __init__(self,parts):
        self.x=torch.from_numpy(np.concatenate([p[0] for p in parts]))
        self.y=torch.from_numpy(np.concatenate([p[1] for p in parts]))
        self.p=torch.from_numpy(np.concatenate([np.full(len(p[0]),p[2],np.int64) for p in parts]))
    def __len__(self): return len(self.x)
    def __getitem__(self,i): return self.x[i],self.p[i],self.y[i]

class MatchedGRU(nn.Module):
    def __init__(self,input_dim,hidden=152,layers=2,n_h=5,drop=0.10):
        super().__init__()
        self.gru=nn.GRU(input_dim,hidden,num_layers=layers,batch_first=True,
                        dropout=drop if layers>1 else 0)
        self.pemb=nn.Embedding(6,16)
        self.hemb=nn.Embedding(n_h,32)
        self.fuse=nn.Sequential(nn.Linear(hidden+16,128),nn.GELU(),
                                nn.Dropout(drop),nn.Linear(128,64),nn.GELU())
        self.heads=nn.ModuleList([
            nn.Sequential(nn.Linear(96,32),nn.GELU(),nn.Linear(32,1))
            for _ in range(n_h)])
    def forward(self,x,p):
        z,_=self.gru(x); z=z[:,-1]
        z=self.fuse(torch.cat([z,self.pemb(p)],-1))
        return torch.stack([
            head(torch.cat([z,self.hemb.weight[i].expand(len(x),-1)],-1)).squeeze(-1)
            for i,head in enumerate(self.heads)],1)

def nparams(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-root",default="data/phase2/ohio2018")
    ap.add_argument("--lookback",type=int,default=120)
    ap.add_argument("--epochs",type=int,default=30)
    ap.add_argument("--patience",type=int,default=6)
    ap.add_argument("--batch-size",type=int,default=128)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--lr",type=float,default=3e-4)
    a=ap.parse_args(); seed_all(a.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root=Path(a.data_root); trparts=[]; vparts=[]; scalers={}; counts={}
    for pi,pid in enumerate(PATIENTS):
        d=read_csv(root/"train"/f"{pid}.csv"); split=int(.85*len(d))
        tr=d.iloc[:split].copy(); va=d.iloc[split:].copy()
        mu=tr[FEATURES].mean(); sd=tr[FEATURES].std().replace(0,1.0)
        tr[FEATURES]=(tr[FEATURES]-mu)/sd; va[FEATURES]=(va[FEATURES]-mu)/sd
        X,y=seqs(tr,a.lookback,HORIZONS); Xv,yv=seqs(va,a.lookback,HORIZONS)
        trparts.append((X,y,pi)); vparts.append((Xv,yv,pi))
        scalers[pid]={"mean":mu.to_dict(),"std":sd.to_dict()}
        counts[pid]={"train":len(X),"val":len(Xv)}
    loader=DataLoader(DS(trparts),batch_size=a.batch_size,shuffle=True)
    model=MatchedGRU(len(FEATURES),hidden=152,layers=2,n_h=5).to(device)
    print("device:",device); print("parameters:",nparams(model))
    print("patient sequence counts:",counts)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    best=float("inf"); best_state=None; bad=0
    for ep in range(1,a.epochs+1):
        model.train(); total=0.; n=0
        for x,p,y in loader:
            x,p,y=x.to(device),p.to(device),y.to(device); opt.zero_grad()
            loss=F.smooth_l1_loss(model(x,p),y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            total+=loss.item()*len(x); n+=len(x)
        model.eval(); maes=[]; rmses=[]
        with torch.no_grad():
            for pi,pid in enumerate(PATIENTS):
                X,y,_=vparts[pi]
                vl=DataLoader(DS([(X,y,pi)]),batch_size=a.batch_size)
                ps=[]; ys=[]
                for x,p,t in vl:
                    ps.append(model(x.to(device),p.to(device)).cpu().numpy()); ys.append(t.numpy())
                pred=np.concatenate(ps); yy=np.concatenate(ys)
                mu=scalers[pid]["mean"]["glucose"]; sd=scalers[pid]["std"]["glucose"]
                pred=pred*sd+mu; yy=yy*sd+mu
                e=pred-yy; maes.append(np.mean(np.abs(e),0)); rmses.append(np.sqrt(np.mean(e**2,0)))
        mae=np.stack(maes).mean(0); rmse=np.stack(rmses).mean(0); score=float(mae.mean())
        print(f"epoch={ep:02d} train={total/n:.5f} val_mean_mae={score:.4f} MAE={np.round(mae,3)} RMSE={np.round(rmse,3)}")
        if score<best:
            best=score; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=a.patience: print("early stopping"); break
    out=Path("output/matched_gru_joint_validation"); out.mkdir(parents=True,exist_ok=True)
    ck=out/"matched_gru_joint_best.pt"
    torch.save({"state_dict":best_state,"patients":PATIENTS,"lookback":a.lookback,
                "horizons":HORIZONS,"features":FEATURES,"parameters":nparams(model),
                "seed":a.seed,"scalers":scalers},ck)
    (out/"summary.json").write_text(json.dumps({"parameters":nparams(model),"best_validation_mean_mae":best,"patients":PATIENTS,"horizons":HORIZONS,"checkpoint":str(ck),"sequence_counts":counts},indent=2))
    print("saved:",ck)

if __name__=="__main__": main()
