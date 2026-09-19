import glob
import os
import xml.etree.ElementTree as ET
import pandas as pd

DATA_DIR = "data"


def parse_event_tag(root, tag_name, ts_attr="ts", val_attr="value"):
    node = root.find(tag_name)
    records = []
    if node is None:
        empty = pd.DataFrame(columns=["Timestamp", tag_name])
        empty["Timestamp"] = empty["Timestamp"].astype("datetime64[ns]")
        empty[tag_name] = empty[tag_name].astype("float64")
        return empty
    for event in node.findall("event"):
        ts = event.attrib.get(ts_attr)
        val = event.attrib.get(val_attr)
        if ts is None or val is None:
            continue
        records.append({"Timestamp": ts, tag_name: val})
    df = pd.DataFrame(records, columns=["Timestamp", tag_name])
    if not df.empty:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
        df[tag_name] = pd.to_numeric(df[tag_name], errors="coerce")
        df = df.dropna().sort_values("Timestamp").reset_index(drop=True)
    df["Timestamp"] = df["Timestamp"].astype("datetime64[ns]")
    return df


def parse_meal(root):
    node = root.find("meal")
    records = []
    if node is None:
        return pd.DataFrame(columns=["Timestamp", "carbs"])
    for event in node.findall("event"):
        ts, carbs = event.attrib.get("ts"), event.attrib.get("carbs")
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
    node = root.find(tag_name)
    ranges = []
    if node is None:
        return ranges
    for event in node.findall("event"):
        start = event.attrib.get("ts_begin") or event.attrib.get("tbegin") or event.attrib.get("ts")
        end = event.attrib.get("ts_end") or event.attrib.get("tend")
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


def parse_bolus(root):
    node = root.find("bolus")
    records = []
    if node is None:
        df = pd.DataFrame(columns=["Timestamp", "dose"])
        df["Timestamp"] = df["Timestamp"].astype("datetime64[ns]")
        df["dose"] = df["dose"].astype("float64")
        return df
    for event in node.findall("event"):
        ts, dose = event.attrib.get("ts_begin"), event.attrib.get("dose")
        if ts is None or dose is None:
            continue
        records.append({"Timestamp": ts, "dose": dose})
    df = pd.DataFrame(records, columns=["Timestamp", "dose"])
    if not df.empty:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
        df["dose"] = pd.to_numeric(df["dose"], errors="coerce")
        df = df.dropna().sort_values("Timestamp").reset_index(drop=True)
    df["Timestamp"] = df["Timestamp"].astype("datetime64[ns]")
    return df


def parse_basal_rate(root):
    node = root.find("basal")
    records = []
    if node is not None:
        for event in node.findall("event"):
            ts, val = event.attrib.get("ts"), event.attrib.get("value")
            if ts is None or val is None:
                continue
            records.append({"Timestamp": ts, "rate": val})
    df = pd.DataFrame(records, columns=["Timestamp", "rate"])
    if not df.empty:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
        df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
        df = df.dropna().sort_values("Timestamp").reset_index(drop=True)
    df["Timestamp"] = df["Timestamp"].astype("datetime64[ns]")
    return df


def parse_temp_basal_ranges(root):
    node = root.find("temp_basal")
    ranges = []
    if node is None:
        return ranges
    for event in node.findall("event"):
        start, end, rate = event.attrib.get("ts_begin"), event.attrib.get("ts_end"), event.attrib.get("value")
        if start is None or end is None or rate is None:
            continue
        start = pd.to_datetime(start, format="%d-%m-%Y %H:%M:%S", errors="coerce")
        end = pd.to_datetime(end, format="%d-%m-%Y %H:%M:%S", errors="coerce")
        rate = pd.to_numeric(rate, errors="coerce")
        if pd.isna(start) or pd.isna(end) or pd.isna(rate):
            continue
        ranges.append((start, end, rate))
    return ranges


def active_basal_rate(timestamp, standing_rate, temp_ranges):
    for start, end, rate in temp_ranges:
        if start <= timestamp <= end:
            return rate
    return standing_rate


def build_patient_dataframe(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    glucose_df = parse_event_tag(root, "glucose_level", val_attr="value").rename(columns={"glucose_level": "Glucose"})
    hr_df = parse_event_tag(root, "basis_heart_rate", val_attr="value").rename(columns={"basis_heart_rate": "HeartRate"})
    steps_df = parse_event_tag(root, "basis_steps", val_attr="value").rename(columns={"basis_steps": "Steps"})
    meal_df = parse_meal(root)
    sleep_ranges = parse_sleep(root, "sleep") + parse_sleep(root, "basis_sleep")
    bolus_df = parse_bolus(root)
    basal_df = parse_basal_rate(root)
    temp_basal_ranges = parse_temp_basal_ranges(root)

    if glucose_df.empty:
        raise ValueError(f"No glucose_level events found in {xml_path}")

    merged = pd.merge_asof(glucose_df.sort_values("Timestamp"), hr_df.sort_values("Timestamp"),
                            on="Timestamp", direction="nearest", tolerance=pd.Timedelta("5min"))
    merged = pd.merge_asof(merged.sort_values("Timestamp"), steps_df.sort_values("Timestamp"),
                            on="Timestamp", direction="nearest", tolerance=pd.Timedelta("5min"))

    merged["carbs_last_60min"] = 0.0
    if not meal_df.empty:
        for _, meal in meal_df.iterrows():
            window = (merged["Timestamp"] >= meal["Timestamp"]) & (merged["Timestamp"] <= meal["Timestamp"] + pd.Timedelta("60min"))
            merged.loc[window, "carbs_last_60min"] += meal["carbs"]

    merged["is_sleeping"] = merged["Timestamp"].apply(lambda t: is_sleeping(t, sleep_ranges))

    merged["bolus_last_60min"] = 0.0
    if not bolus_df.empty:
        for _, b in bolus_df.iterrows():
            window = (merged["Timestamp"] >= b["Timestamp"]) & (merged["Timestamp"] <= b["Timestamp"] + pd.Timedelta("60min"))
            merged.loc[window, "bolus_last_60min"] += b["dose"]

    merged = pd.merge_asof(
        merged.sort_values("Timestamp"),
        basal_df.rename(columns={"rate": "standing_basal_rate"}).sort_values("Timestamp"),
        on="Timestamp", direction="backward",
    )
    merged["standing_basal_rate"] = merged["standing_basal_rate"].ffill().bfill()
    merged["basal_rate"] = merged.apply(
        lambda row: active_basal_rate(row["Timestamp"], row["standing_basal_rate"], temp_basal_ranges), axis=1,
    )
    merged = merged.drop(columns=["standing_basal_rate"])

    merged["HeartRate"] = merged["HeartRate"].ffill().bfill()
    merged["Steps"] = merged["Steps"].ffill().bfill()
    merged = merged.dropna(subset=["Glucose"]).reset_index(drop=True)
    return merged



if __name__ == "__main__":
    PATIENT_IDS_2018 = ["559", "563", "570", "575", "588", "591"]
    PATIENT_IDS_2020 = ["540", "544", "552", "567", "584", "596"]
    ALL_PATIENT_IDS = PATIENT_IDS_2018 + PATIENT_IDS_2020

    for pid in ALL_PATIENT_IDS:
        for split in ["training", "testing"]:
            xml_path = os.path.join(DATA_DIR, f"{pid}-ws-{split}.xml")
            if not os.path.exists(xml_path):
                print(f"Missing: {xml_path}")
                continue
            print(f"Parsing {xml_path} ...")
            df = build_patient_dataframe(xml_path)
            out_path = os.path.join(DATA_DIR, f"{pid}_{split}_multimodal.csv")
            df.to_csv(out_path, index=False)
            print(f"Saved: {out_path}  shape={df.shape}")
