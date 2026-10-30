from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from experiments.glunet.analyze_hypoglycemia_episodes import (
    extract_episodes,
    match_episodes,
    evaluate_episode_setting,
)


def test_extract_episodes_groups_by_gap_tolerance():
    start = datetime(2026, 1, 1)
    times = pd.date_range(start, periods=6, freq="5min")
    episodes = extract_episodes(times, np.array([1, 1, 0, 1, 0, 1], bool), 10)
    assert len(episodes) == 2
    assert episodes[0]["n_low_observations"] == 2
    assert episodes[1]["n_low_observations"] == 2


def test_extract_episodes_splits_when_gap_exceeds_tolerance():
    times = pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:05",
                            "2026-01-01 00:20"])
    episodes = extract_episodes(times, np.ones(3, dtype=bool), 10)
    assert len(episodes) == 2


def test_extract_episodes_rejects_unsorted_timestamps():
    times = pd.to_datetime(["2026-01-01 00:05", "2026-01-01 00:00"])
    with pytest.raises(ValueError, match="sorted"):
        extract_episodes(times, np.ones(2, dtype=bool))


def test_match_episodes_one_to_one_and_delay_sign():
    actual = [{"episode_id": 1, "start": pd.Timestamp("2026-01-01 00:10"),
               "end": pd.Timestamp("2026-01-01 00:20")}]
    predicted = [{"episode_id": 1, "start": pd.Timestamp("2026-01-01 00:05"),
                  "end": pd.Timestamp("2026-01-01 00:15")},
                 {"episode_id": 2, "start": pd.Timestamp("2026-01-01 00:10"),
                  "end": pd.Timestamp("2026-01-01 00:20")}]
    matches, missed, false = match_episodes(actual, predicted, 5)
    assert len(matches) == 1
    assert matches[0]["detection_delay_minutes"] == -5
    assert missed == []
    assert len(false) == 1


def test_episode_metrics_count_misses_and_false_predictions():
    times = pd.date_range("2026-01-01", periods=12, freq="5min")
    actual = np.full(12, 100.0)
    predicted = np.full(12, 100.0)
    actual[[1, 2]] = 60
    actual[[8, 9]] = 60
    predicted[[1, 2]] = 60
    predicted[[5, 6]] = 60
    summary, _, _ = evaluate_episode_setting(
        times, actual, predicted, gap_tolerance_minutes=10,
        anticipation_minutes=0, monitored_hours=1.0,
    )
    assert summary["actual_episode_count"] == 2
    assert summary["predicted_episode_count"] == 2
    assert summary["detected_episode_count"] == 1
    assert summary["missed_episode_count"] == 1
    assert summary["false_predicted_episode_count"] == 1
    assert summary["false_alarms_per_24h"] == 24.0
