"""
OhioT1DM 2020 acceleration preprocessing.

Purpose
-------
Parse the scalar acceleration magnitude in the OhioT1DM 2020 XML files and
causally aggregate it to the 5-minute CGM timestamps.

The resulting features are:
    accel_mean_5m
    accel_std_5m
    accel_max_5m
    accel_min_5m
    accel_count_5m
    accel_coverage_5m
    accel_missing_5m

Important:
- CGM timestamps are the reference grid.
- Only acceleration observations strictly before the CGM timestamp are used.
- No centered/future-looking window is used.
- Missing acceleration is retained explicitly through count/coverage/missingness.
- Raw OhioT1DM data should remain outside the public repository because it is
  distributed under a data-use agreement.
"""

from pathlib import Path
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd


def parse_events(xml_path, section, ts_attr="ts", value_attr="value"):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    node = next((x for x in root.iter() if x.tag.lower() == section.lower()), None)
    if node is None:
        return pd.DataFrame(columns=["timestamp", "value"])

    rows = []
    for event in node:
        ts = event.attrib.get(ts_attr)
        value = event.attrib.get(value_attr)
        if ts is None or value is None:
            continue
        rows.append(
            (
                pd.to_datetime(ts, format="%d-%m-%Y %H:%M:%S"),
                float(value),
            )
        )

    return (
        pd.DataFrame(rows, columns=["timestamp", "value"])
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def aggregate_acceleration_to_cgm(cgm, acceleration):
    cgm = cgm.sort_values("timestamp").reset_index(drop=True)
    acceleration = acceleration.sort_values("timestamp").reset_index(drop=True)

    if acceleration.empty:
        for col in [
            "accel_mean_5m",
            "accel_std_5m",
            "accel_max_5m",
            "accel_min_5m",
            "accel_count_5m",
            "accel_coverage_5m",
        ]:
            cgm[col] = np.nan
        cgm["accel_missing_5m"] = 1
        return cgm

    # Put both series on one time index and calculate a strictly-causal
    # trailing 5-minute window. closed='left' excludes acceleration at t.
    s = acceleration.set_index("timestamp")["value"].sort_index()
    combined_index = s.index.union(cgm["timestamp"]).sort_values()
    s = s.reindex(combined_index)

    rolling = s.rolling("5min", closed="left")

    stats = pd.DataFrame(
        {
            "timestamp": combined_index,
            "accel_mean_5m": rolling.mean().to_numpy(),
            "accel_std_5m": rolling.std(ddof=0).to_numpy(),
            "accel_max_5m": rolling.max().to_numpy(),
            "accel_min_5m": rolling.min().to_numpy(),
            "accel_count_5m": rolling.count().to_numpy(),
        }
    )

    out = cgm.merge(stats, on="timestamp", how="left")
    out["accel_coverage_5m"] = (out["accel_count_5m"] / 5.0).clip(0, 1)
    out["accel_missing_5m"] = (
        out["accel_count_5m"].fillna(0).eq(0).astype(int)
    )
    return out


def process_file(xml_path, output_path):
    cgm = parse_events(xml_path, "glucose_level")
    acceleration = parse_events(xml_path, "acceleration")
    aligned = aggregate_acceleration_to_cgm(cgm, acceleration)
    aligned.to_csv(output_path, index=False)
    return aligned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True,
                        help="Path containing 2020/train and 2020/test")
    parser.add_argument("--output-root", default="output/acceleration_2020")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for split in ["train", "test"]:
        split_dir = data_root / split
        out_dir = output_root / split
        out_dir.mkdir(parents=True, exist_ok=True)

        for xml_path in sorted(split_dir.glob("*.xml")):
            patient = xml_path.name.split("-")[0]
            out_path = out_dir / f"{patient}_acceleration_aligned.csv"
            process_file(xml_path, out_path)
            print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
