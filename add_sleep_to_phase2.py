import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd

PATIENTS_2018 = [559, 563, 570, 575, 588, 591]
PATIENTS_2020 = [540, 544, 552, 567, 584, 596]

def parse_ts(v):
    if v is None:
        return pd.NaT
    v = str(v).strip()
    for fmt in ("%d-%m-%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        x = pd.to_datetime(v, format=fmt, errors="coerce", utc=True)
        if not pd.isna(x):
            return x
    return pd.to_datetime(v, errors="coerce", utc=True)

def find_xml(root, patient, split):
    name = f"{patient}-ws-{split}.xml"
    for p in (root/name, root/"OhioT1DM_data_backup"/name):
        if p.exists():
            return p
    raise FileNotFoundError(name)

def parse_sleep(path):
    root = ET.parse(path).getroot()
    rows = []
    for section in root:
        if section.tag.split("}")[-1].lower() not in {"sleep", "basis_sleep"}:
            continue
        for e in section:
            if e.tag.split("}")[-1].lower() != "event":
                continue
            start = parse_ts(e.attrib.get("ts_end"))
            end = parse_ts(e.attrib.get("ts_begin"))
            if pd.isna(start) or pd.isna(end) or start == end:
                continue
            if end < start:
                start, end = end, start
            rows.append((start, end))
    return pd.DataFrame(rows, columns=["sleep_start","sleep_end"]).drop_duplicates()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--phase2-root", default="data/phase2")
    args = ap.parse_args()
    root, phase2 = Path(args.data_root), Path(args.phase2_root)
    audit = []
    for cohort, patients in {"ohio2018":PATIENTS_2018, "ohio2020":PATIENTS_2020}.items():
        for split, folder in [("training","train"),("testing","test")]:
            for patient in patients:
                xp = find_xml(root, patient, split)
                cp = phase2/cohort/folder/f"{patient}.csv"
                df = pd.read_csv(cp)
                sleep = parse_sleep(xp)
                ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
                state = np.zeros(len(df), dtype=np.int8)
                for a,b in sleep.itertuples(index=False, name=None):
                    state[((ts >= a) & (ts <= b)).to_numpy()] = 1
                df["is_sleeping"] = state
                df.to_csv(cp, index=False)
                audit.append({"cohort":cohort,"patient":patient,"split":split,
                              "sleep_intervals":len(sleep),"rows":len(df),
                              "sleep_pct":100*state.mean()})
                print(f"{cohort} {patient} {split}: {len(sleep)} intervals, {state.sum()}/{len(state)} sleeping rows")
    out = phase2/"sleep_feature_audit.csv"
    pd.DataFrame(audit).to_csv(out,index=False)
    print(f"\nAudit saved: {out}")

if __name__ == "__main__":
    main()
