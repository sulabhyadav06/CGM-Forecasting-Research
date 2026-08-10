"""
OhioT1DM multimodal parser.
Extracts glucose, heart rate, steps, meals, and sleep for ONE patient
and merges them onto the glucose timeline (5-min grid).
"""

import glob
import os
import xml.etree.ElementTree as ET
import pandas as pd

DATA_DIR = "data"
PATIENT_ID = "570"   # <-- change this to whichever patient you want to process
SPLIT = "testing"    # "training" or "testing"


def parse_event_tag(root, tag_name, ts_attr="ts", val_attr="value"):
    """Generic extractor for simple <event ts=".." value=".."/> style tags."""
    node = root.find(tag_name)
    records = []
    if node is None:
        return pd.DataFrame(columns=["Timestamp", tag_name])
    for event in node.findall("event"):
        ts = event.attrib.get(ts_attr)
        val = event.attrib.get(val_attr)
        if ts is None or val is None:
            continue
        records.append({"Timestamp": ts, tag_name: val})
    df = pd.DataFrame(records)
    if not df.empty:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
        df[tag_name] = pd.to_numeric(df[tag_name], errors="coerce")
        df = df.dropna().sort_values("Timestamp").reset_index(drop=True)
    return df


def parse_meal(root):
    """Meals: ts + carbs. Sparse events."""
    node = root.find("meal")
    records = []
    if node is None:
        return pd.DataFrame(columns=["Timestamp", "carbs"])
    for event in node.findall("event"):
        ts = event.attrib.get("ts")
        carbs = event.attrib.get("carbs")
        if ts is None or carbs is None:
            continue
        records.append({"Timestamp": ts, "carbs": carbs})
    df = pd.DataFrame(records)
    if not df.empty:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
        df["carbs"] = pd.to_numeric(df["carbs"], errors="coerce")
        df = df.dropna().sort_values("Timestamp").reset_index(drop=True)
    return df


def parse_sleep(root, tag_name="sleep"):
    """Sleep: ts_begin + ts_end ranges. Returns list of (start, end) tuples."""
    node = root.find(tag_name)
    ranges = []
    if node is None:
        return ranges
    for event in node.findall("event"):
        start = event.attrib.get("ts_begin") or event.attrib.get("ts")
        end = event.attrib.get("ts_end")
        if start is None or end is None:
            continue
        start = pd.to_datetime(start, format="%d-%m-%Y %H:%M:%S", errors="coerce")
        end = pd.to_datetime(end, format="%d-%m-%Y %H:%M:%S", errors="coerce")
        if pd.isna(start) or pd.isna(end):
            continue
        ranges.append((start, end))
    return ranges


def is_sleeping(timestamp, sleep_ranges):
    for start, end in sleep_ranges:
        if start <= timestamp <= end:
            return 1
    return 0


def build_patient_dataframe(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    glucose_df = parse_event_tag(root, "glucose_level", val_attr="value")
    glucose_df = glucose_df.rename(columns={"glucose_level": "Glucose"})

    hr_df = parse_event_tag(root, "basis_heart_rate", val_attr="value")
    hr_df = hr_df.rename(columns={"basis_heart_rate": "HeartRate"})

    steps_df = parse_event_tag(root, "basis_steps", val_attr="value")
    steps_df = steps_df.rename(columns={"basis_steps": "Steps"})

    meal_df = parse_meal(root)
    sleep_ranges = parse_sleep(root, "sleep") + parse_sleep(root, "basis_sleep")

    if glucose_df.empty:
        raise ValueError(f"No glucose_level events found in {xml_path}")

    # Merge HR and Steps onto glucose timeline (nearest match within 5 min tolerance)
    merged = pd.merge_asof(
        glucose_df.sort_values("Timestamp"),
        hr_df.sort_values("Timestamp"),
        on="Timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("5min"),
    )
    merged = pd.merge_asof(
        merged.sort_values("Timestamp"),
        steps_df.sort_values("Timestamp"),
        on="Timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("5min"),
    )

    # Carbs in last 60 minutes (rolling sum of meal carbs)
    merged["carbs_last_60min"] = 0.0
    if not meal_df.empty:
        for _, meal in meal_df.iterrows():
            window = (merged["Timestamp"] >= meal["Timestamp"]) & (
                merged["Timestamp"] <= meal["Timestamp"] + pd.Timedelta("60min")
            )
            merged.loc[window, "carbs_last_60min"] += meal["carbs"]

    # Sleep flag
    merged["is_sleeping"] = merged["Timestamp"].apply(lambda t: is_sleeping(t, sleep_ranges))

    # Fill remaining gaps in HR/Steps (forward fill, then back fill for leading NaNs)
    merged["HeartRate"] = merged["HeartRate"].ffill().bfill()
    merged["Steps"] = merged["Steps"].ffill().bfill()

    merged = merged.dropna(subset=["Glucose"]).reset_index(drop=True)
    return merged


if __name__ == "__main__":
    for pid in ["559", "563", "570", "575", "588", "591"]:
        for split in ["training", "testing"]:
            xml_path = os.path.join(DATA_DIR, f"{pid}-ws-{split}.xml")
            if not os.path.exists(xml_path):
                print(f"Missing: {xml_path}")
                continue
            print(f"Parsing {xml_path} ...")
            df = build_patient_dataframe(xml_path)
            out_path = os.path.join(DATA_DIR, f"{pid}_{split}_multimodal.csv")
            df.to_csv(out_path, index=False)
            print(f"Saved: {out_path}")