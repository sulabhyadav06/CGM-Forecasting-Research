import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from phase2_multimodal_preprocessing import (
    HISTORY,
    align_basal,
    causal_aggregate,
    event_total,
    parse_xml,
)


def test_phase2_parser_extracts_bolus_events_at_ts_begin():
    root = ET.fromstring("""
    <patient>
      <glucose_level>
        <event ts="01-01-2026 12:00:00" value="100"/>
      </glucose_level>
      <bolus>
        <event ts_begin="01-01-2026 11:55:00"
               ts_end="01-01-2026 11:55:00"
               dose="1.25"/>
      </bolus>
    </patient>
    """)
    # parse_xml currently accepts a path, so use a temporary XML file.
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fixture.xml"
        path.write_text(ET.tostring(root, encoding="unicode"))
        parsed = parse_xml(path)

    assert len(parsed["bolus"]) == 1
    assert np.isclose(parsed["bolus"]["bolus"].iloc[0], 1.25)


def test_event_total_excludes_event_exactly_60_minutes_old():
    t = pd.Timestamp("2026-01-01 12:00:00", tz="UTC")
    grid = pd.DataFrame({"timestamp": [t]})
    events = pd.DataFrame({
        "timestamp": [
            t - HISTORY,
            t - pd.Timedelta(minutes=30),
            t,
        ],
        "carbs": [10.0, 20.0, 30.0],
    })

    result = event_total(grid, events, "carbs")
    assert result.iloc[0] == 50.0


def test_sensor_aggregation_excludes_current_timestamp():
    t = pd.Timestamp("2026-01-01 12:05:00", tz="UTC")
    grid = pd.DataFrame({"timestamp": [t]})
    observations = pd.DataFrame({
        "timestamp": [
            t - pd.Timedelta(minutes=4),
            t,
        ],
        "value": [2.0, 100.0],
    })

    result = causal_aggregate(grid, observations, "value", "sensor")
    assert result.loc[0, "sensor_mean_5m"] == 2.0
    assert result.loc[0, "sensor_count_5m"] == 1


def test_basal_alignment_does_not_use_future_observation():
    grid = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-01-01 12:00:00",
            "2026-01-01 12:05:00",
        ], utc=True)
    })
    basal = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-01-01 12:02:00",
        ], utc=True),
        "basal": [1.5],
    })

    result = align_basal(grid, basal)
    assert result.isna().iloc[0]
    assert result.iloc[1] == 1.5


def test_phase2_parser_preserves_temporary_basal_intervals():
    import tempfile
    from pathlib import Path

    root = ET.fromstring("""
    <patient>
      <glucose_level>
        <event ts="01-01-2026 12:00:00" value="100"/>
      </glucose_level>
      <basal>
        <event ts="01-01-2026 11:00:00" value="1.0"/>
      </basal>
      <temp_basal>
        <event ts_begin="01-01-2026 11:55:00"
               ts_end="01-01-2026 12:10:00"
               value="0.5"/>
      </temp_basal>
    </patient>
    """)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "temporary_basal.xml"
        path.write_text(ET.tostring(root, encoding="unicode"))
        parsed = parse_xml(path)

    assert "temp_basal" in parsed
    intervals = parsed["temp_basal"]
    assert len(intervals) == 1
    assert intervals.iloc[0]["value"] == 0.5
    assert intervals.iloc[0]["start"] == pd.Timestamp(
        "2026-01-01 11:55:00", tz="UTC"
    )
    assert intervals.iloc[0]["end"] == pd.Timestamp(
        "2026-01-01 12:10:00", tz="UTC"
    )


def test_temporary_basal_overrides_only_during_its_interval():
    from phase2_multimodal_preprocessing import apply_temp_basal

    grid = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-01-01 11:50:00+00:00",
            "2026-01-01 11:55:00+00:00",
            "2026-01-01 12:00:00+00:00",
            "2026-01-01 12:10:00+00:00",
        ], utc=True)
    })

    ordinary = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-01-01 11:00:00+00:00",
        ], utc=True),
        "basal": [1.0],
    })

    temporary = pd.DataFrame({
        "start": pd.to_datetime([
            "2026-01-01 11:55:00+00:00",
        ], utc=True),
        "end": pd.to_datetime([
            "2026-01-01 12:10:00+00:00",
        ], utc=True),
        "value": [0.5],
    })

    result = apply_temp_basal(grid, ordinary, temporary)

    assert result.tolist() == [1.0, 0.5, 0.5, 1.0]


def test_latest_starting_temporary_basal_wins_when_intervals_overlap():
    from phase2_multimodal_preprocessing import apply_temp_basal

    grid = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-01-01 12:02:00+00:00",
            "2026-01-01 12:07:00+00:00",
            "2026-01-01 12:12:00+00:00",
        ], utc=True)
    })

    ordinary = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-01-01 11:00:00+00:00",
        ], utc=True),
        "basal": [1.0],
    })

    temporary = pd.DataFrame({
        "start": pd.to_datetime([
            "2026-01-01 12:00:00+00:00",
            "2026-01-01 12:05:00+00:00",
        ], utc=True),
        "end": pd.to_datetime([
            "2026-01-01 12:15:00+00:00",
            "2026-01-01 12:10:00+00:00",
        ], utc=True),
        "value": [0.5, 0.8],
    })

    result = apply_temp_basal(grid, ordinary, temporary)

    assert result.tolist() == [0.5, 0.8, 0.5]
