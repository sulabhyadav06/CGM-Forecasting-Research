"""
Phase-2 OhioT1DM multimodal preprocessing.

2018: CGM + carbs + insulin/bolus + basal + heart rate + steps
2020: CGM + carbs + insulin/bolus + basal + scalar acceleration

OhioT1DM XML stores measurements as <event .../> elements inside named
sections (e.g. <glucose_level>, <meal>, <bolus>). The parser therefore uses
the parent section name to identify each event type.

CGM is the master 5-minute grid.
Event totals use events timestamped at t.
Sensor aggregates use the causal interval [t-5 min, t).
No scaling is performed here.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd


PATIENTS_2018 = [559, 563, 570, 575, 588, 591]
PATIENTS_2020 = [540, 544, 552, 567, 584, 596]

HISTORY = pd.Timedelta(minutes=60)
WINDOW = pd.Timedelta(minutes=5)


def parse_ts(value):
    """Parse Ohio timestamps such as 07-12-2021 01:17:00 as UTC."""
    if value is None:
        return pd.NaT

    value = str(value).strip()

    for fmt in ("%d-%m-%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        x = pd.to_datetime(
            value,
            format=fmt,
            errors="coerce",
            utc=True,
        )
        if not pd.isna(x):
            return x

    return pd.to_datetime(value, errors="coerce", utc=True)


def attr_float(event, names):
    for name in names:
        if name in event.attrib:
            try:
                return float(event.attrib[name])
            except (TypeError, ValueError):
                return np.nan
    return np.nan


def parse_xml(path: Path):
    """
    Parse OhioT1DM XML.

    Important:
        The XML has section -> event structure, for example:

        <glucose_level>
            <event ts="..." value="101"/>
        </glucose_level>

    Therefore the section name is obtained from event.getparent()-equivalent
    logic by iterating section elements and then their children.
    """
    root = ET.parse(path).getroot()

    records = {
        "glucose": [],
        "carbs": [],
        "bolus": [],
        "basal": [],
        "heart_rate": [],
        "steps": [],
        "acceleration": [],
    }

    # Iterate over sections, then events within each section.
    for section in root:
        section_name = section.tag.split("}")[-1].lower()

        for event in section:
            tag = event.tag.split("}")[-1].lower()

            # Ohio files use <event .../> in these sections.
            if tag != "event":
                continue

            timestamp = parse_ts(event.attrib.get("ts"))
            if pd.isna(timestamp):
                continue

            value = attr_float(event, ("value",))

            if section_name == "glucose_level":
                if np.isfinite(value):
                    records["glucose"].append((timestamp, value))

            elif section_name == "meal":
                value = attr_float(
                    event,
                    ("carbs", "carb_input", "carbohydrates"),
                )
                if np.isfinite(value):
                    records["carbs"].append((timestamp, value))

            elif section_name == "bolus":
                value = attr_float(
                    event,
                    ("dose", "bolus", "insulin"),
                )
                if np.isfinite(value):
                    records["bolus"].append((timestamp, value))

            elif section_name in ("basal", "temp_basal"):
                if np.isfinite(value):
                    records["basal"].append((timestamp, value))

            elif section_name in (
                "basis_heart_rate",
                "heart_rate",
            ):
                if np.isfinite(value):
                    records["heart_rate"].append((timestamp, value))

            elif section_name in (
                "basis_steps",
                "steps",
            ):
                if np.isfinite(value):
                    records["steps"].append((timestamp, value))

            elif section_name == "acceleration":
                if np.isfinite(value):
                    records["acceleration"].append(
                        (timestamp, value)
                    )

    if not records["glucose"]:
        raise ValueError(
            f"No glucose observations found in {path}. "
            "Expected <glucose_level><event ts=... value=.../></glucose_level>."
        )

    def make_df(items, column):
        if not items:
            return pd.DataFrame(
                columns=["timestamp", column]
            )

        df = pd.DataFrame(
            items,
            columns=["timestamp", column],
        )
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True,
        )

        return (
            df.dropna(subset=["timestamp"])
            .sort_values("timestamp")
            .drop_duplicates("timestamp", keep="last")
            .reset_index(drop=True)
        )

    return {
        key: make_df(value, key)
        for key, value in records.items()
    }


def event_total(grid, events, value_column):
    """
    Sum events over (t-60min, t].

    Events at exactly t are included because they are timestamped at the
    current CGM observation time.
    """
    if events.empty:
        return pd.Series(
            0.0,
            index=grid.index,
        )

    event_times = (
        events["timestamp"]
        .astype("int64")
        .to_numpy()
    )

    values = events[value_column].to_numpy(
        dtype=float
    )

    cumulative = np.concatenate(
        [[0.0], np.cumsum(values)]
    )

    grid_times = (
        grid["timestamp"]
        .astype("int64")
        .to_numpy()
    )

    left_times = (
        grid["timestamp"] - HISTORY
    ).astype("int64").to_numpy()

    right = np.searchsorted(
        event_times,
        grid_times,
        side="right",
    )

    left = np.searchsorted(
        event_times,
        left_times,
        side="right",
    )

    return pd.Series(
        cumulative[right] - cumulative[left],
        index=grid.index,
    )


def causal_aggregate(
    grid,
    observations,
    value_column,
    prefix,
):
    """
    Aggregate observations in [t-5min, t).

    This is causal: an observation timestamped exactly at t is not used to
    construct the current t sensor feature.
    """
    names = [
        f"{prefix}_mean_5m",
        f"{prefix}_std_5m",
        f"{prefix}_min_5m",
        f"{prefix}_max_5m",
        f"{prefix}_count_5m",
        f"{prefix}_coverage_5m",
        f"{prefix}_missing_5m",
    ]

    if observations.empty:
        return pd.DataFrame(
            {
                names[0]: np.nan,
                names[1]: np.nan,
                names[2]: np.nan,
                names[3]: np.nan,
                names[4]: 0,
                names[5]: 0.0,
                names[6]: 1,
            },
            index=grid.index,
        )

    obs_times = (
        observations["timestamp"]
        .astype("int64")
        .to_numpy()
    )

    obs_values = observations[value_column].to_numpy(
        dtype=float
    )

    grid_times = (
        grid["timestamp"]
        .astype("int64")
        .to_numpy()
    )

    window_ns = int(WINDOW.value)

    rows = []

    for current_time in grid_times:
        right = np.searchsorted(
            obs_times,
            current_time,
            side="left",
        )

        left = np.searchsorted(
            obs_times,
            current_time - window_ns,
            side="left",
        )

        values = obs_values[left:right]
        values = values[np.isfinite(values)]

        if len(values) == 0:
            rows.append(
                (
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    0,
                    0.0,
                    1,
                )
            )
        else:
            rows.append(
                (
                    float(np.mean(values)),
                    float(
                        np.std(values, ddof=1)
                    ) if len(values) > 1 else 0.0,
                    float(np.min(values)),
                    float(np.max(values)),
                    int(len(values)),
                    1.0,
                    0,
                )
            )

    return pd.DataFrame(
        rows,
        columns=names,
        index=grid.index,
    )


def align_basal(grid, basal):
    """
    Latest known basal state at or before each CGM timestamp.
    """
    if basal.empty:
        return pd.Series(
            np.nan,
            index=grid.index,
        )

    left = grid[
        ["timestamp"]
    ].sort_values("timestamp")

    right = basal[
        ["timestamp", "basal"]
    ].sort_values("timestamp")

    merged = pd.merge_asof(
        left,
        right,
        on="timestamp",
        direction="backward",
    )

    return merged["basal"].set_axis(
        grid.index
    )


def process(path: Path, cohort: str):
    data = parse_xml(path)

    # CGM is the master grid.
    grid = data["glucose"].copy()

    grid["gap_from_previous_min"] = (
        grid["timestamp"]
        .diff()
        .dt.total_seconds()
        / 60.0
    )

    grid["valid_5min_interval"] = (
        grid["gap_from_previous_min"].eq(5.0)
        | grid["gap_from_previous_min"].isna()
    )

    # Causal 60-minute event history.
    grid["carbs_last_60min"] = event_total(
        grid,
        data["carbs"],
        "carbs",
    )

    grid["bolus_last_60min"] = event_total(
        grid,
        data["bolus"],
        "bolus",
    )

    grid["basal_rate"] = align_basal(
        grid,
        data["basal"],
    ).to_numpy()

    if cohort == "2018":
        heart = causal_aggregate(
            grid,
            data["heart_rate"],
            "heart_rate",
            "heart_rate",
        )

        steps = causal_aggregate(
            grid,
            data["steps"],
            "steps",
            "steps",
        )

        grid = pd.concat(
            [grid, heart, steps],
            axis=1,
        )

    elif cohort == "2020":
        acceleration = causal_aggregate(
            grid,
            data["acceleration"],
            "acceleration",
            "accel",
        )

        grid = pd.concat(
            [grid, acceleration],
            axis=1,
        )

    else:
        raise ValueError(
            f"Unknown cohort: {cohort}"
        )

    grid["timestamp"] = (
        grid["timestamp"]
        .dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    )

    return grid


def find_xml(
    data_root: Path,
    patient: int,
    split: str,
):
    filename = (
        f"{patient}-ws-{split}.xml"
    )

    candidates = [
        data_root / filename,
        data_root
        / "OhioT1DM_data_backup"
        / filename,
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find {filename}. "
        f"Checked: {candidates}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        default="data",
    )

    parser.add_argument(
        "--output-root",
        default="data/phase2",
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)

    cohorts = {
        "ohio2018": (
            "2018",
            PATIENTS_2018,
        ),
        "ohio2020": (
            "2020",
            PATIENTS_2020,
        ),
    }

    audit_rows = []

    for folder, (
        cohort,
        patients,
    ) in cohorts.items():

        for split in (
            "training",
            "testing",
        ):

            output_dir = (
                output_root
                / folder
                / (
                    "train"
                    if split == "training"
                    else "test"
                )
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            for patient in patients:

                source = find_xml(
                    data_root,
                    patient,
                    split,
                )

                print(
                    f"\nProcessing "
                    f"{cohort} | "
                    f"patient={patient} | "
                    f"{split}"
                )

                df = process(
                    source,
                    cohort,
                )

                destination = (
                    output_dir
                    / f"{patient}.csv"
                )

                df.to_csv(
                    destination,
                    index=False,
                )

                audit_rows.append(
                    {
                        "cohort": cohort,
                        "patient": patient,
                        "split": split,
                        "rows": len(df),
                        "columns": len(df.columns),
                        "continuous_pct": (
                            100.0
                            * df[
                                "valid_5min_interval"
                            ].mean()
                        ),
                    }
                )

                print(
                    f"Saved {destination} "
                    f"({len(df)} rows, "
                    f"{len(df.columns)} columns)"
                )

    audit = pd.DataFrame(
        audit_rows
    )

    audit_path = (
        output_root
        / "phase2_generation_audit.csv"
    )

    audit.to_csv(
        audit_path,
        index=False,
    )

    print("\n==============================")
    print("Phase-2 preprocessing complete")
    print("==============================")
    print(audit.to_string(index=False))
    print(
        f"\nAudit saved to: {audit_path}"
    )


if __name__ == "__main__":
    main()
