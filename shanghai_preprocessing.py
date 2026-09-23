"""
shanghai_preprocessing.py
Loader + preprocessing for the Shanghai_T1DM/T2DM dataset. See SHANGHAI_EDA.md
for the full exploratory writeup this is based on. Key facts:
    - 15-minute CGM sampling (uniform across all files).
    - Column 0 = Date, column 1 = CGM (mg/dl) -- read by POSITION, since a
      few files have minor header-name variants.
    - Near-zero missingness; long gaps (>20 min) are rare but handled the
      same way as the OhioT1DM pipeline (segment rather than bridge).
    - Most source files are legacy .xls (OLE2 binary) -- convert to .xlsx
      first via LibreOffice headless (see convert_xls_to_xlsx) or install
      `xlrd`.
"""

from __future__ import annotations
import glob
import os
import subprocess
import numpy as np
import pandas as pd

SAMPLE_INTERVAL_MIN = 15


def convert_xls_to_xlsx(raw_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    xls_files = glob.glob(os.path.join(raw_dir, "**", "*.xls"), recursive=True)
    if xls_files:
        subprocess.run(
            ["soffice", "--headless", "--convert-to", "xlsx", "--outdir", out_dir] + xls_files,
            check=True,
        )
    for xlsx_file in glob.glob(os.path.join(raw_dir, "**", "*.xlsx"), recursive=True):
        dest = os.path.join(out_dir, os.path.basename(xlsx_file))
        if not os.path.exists(dest):
            import shutil
            shutil.copy(xlsx_file, dest)


def parse_patient_visit(path: str) -> tuple[str, str, str]:
    """'2074_1_20210720.xlsx' -> ('T2DM' or 'T1DM', '2074', '1')"""
    fname = os.path.splitext(os.path.basename(path))[0]
    parts = fname.split("_")
    patient_id, visit = parts[0], parts[1]
    dataset = "T1DM" if patient_id.startswith("1") else "T2DM"
    return dataset, patient_id, visit


def load_shanghai_file(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    date_col, cgm_col = df.columns[0], df.columns[1]
    df = df.rename(columns={date_col: "timestamp", cgm_col: "glucose"})[["timestamp", "glucose"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["glucose"] = pd.to_numeric(df["glucose"], errors="coerce")
    return df.dropna().sort_values("timestamp").reset_index(drop=True)


def clean_and_resample(df: pd.DataFrame, freq_min: int = SAMPLE_INTERVAL_MIN,
                        max_gap_fill_steps: int = 2) -> pd.DataFrame:
    df = df.drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()
    df = df[(df["glucose"] >= 20) & (df["glucose"] <= 600)]
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=f"{freq_min}min")
    df = df.reindex(full_index)
    df["glucose"] = df["glucose"].interpolate(method="linear", limit=max_gap_fill_steps)
    df.index.name = "timestamp"
    return df.reset_index()


def segment_contiguous(df: pd.DataFrame, min_len: int) -> list[pd.DataFrame]:
    is_na = df["glucose"].isna()
    seg_id = is_na.cumsum()
    return [g.reset_index(drop=True) for _, g in df[~is_na].groupby(seg_id[~is_na]) if len(g) >= min_len]


def build_all_subjects_table(converted_dir: str) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(os.path.join(converted_dir, "*.xlsx"))):
        dataset, patient_id, visit = parse_patient_visit(f)
        df = load_shanghai_file(f)
        rows.append({
            "dataset": dataset, "patient_id": patient_id, "visit": visit,
            "path": f, "n_rows": len(df),
            "duration_days": round((df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 86400, 1) if len(df) else np.nan,
            "cgm_mean": round(df["glucose"].mean(), 1) if len(df) else np.nan,
            "cgm_std": round(df["glucose"].std(), 1) if len(df) else np.nan,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--out_dir", default="./data_shanghai_converted")
    args = parser.parse_args()
    print("Converting legacy .xls files to .xlsx ...")
    convert_xls_to_xlsx(args.raw_dir, args.out_dir)
    table = build_all_subjects_table(args.out_dir)
    if table.empty:
        raise SystemExit(
            f"\nNo Shanghai data files found under --raw_dir '{args.raw_dir}'.\n"
            f"Check that this path actually points to your unzipped Shanghai_DATASET "
            f"folder (containing Shanghai_T1DM/ and Shanghai_T2DM/ subfolders), not a "
            f"placeholder or typo'd path."
        )
    table.to_csv(os.path.join(args.out_dir, "shanghai_subjects_summary.csv"), index=False)
    print(table.groupby("dataset")[["n_rows", "duration_days", "cgm_mean", "cgm_std"]].mean())
