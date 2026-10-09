from pathlib import Path

import pytest

from lunarops.fileio.observation_results import (
    read_observation_results,
    write_observation_results,
)


def test_observation_result_file_round_trip_has_schema_and_units(tmp_path: Path):
    rows = {
        "source-a": [
            {"normal_point_index": 2, "oc_one_way_m": 0.1, "light_time_converged": True},
            {"normal_point_index": 1, "oc_one_way_m": -0.2, "light_time_converged": False},
        ]
    }
    path = tmp_path / "rows.txt.gz"
    write_observation_results(rows, path)
    recovered = read_observation_results(path)

    assert recovered == [
        {
            "source": "source-a",
            "normal_point_index": 2,
            "oc_one_way_m": 0.1,
            "light_time_converged": True,
        },
        {
            "source": "source-a",
            "normal_point_index": 1,
            "oc_one_way_m": -0.2,
            "light_time_converged": False,
        },
    ]


def test_result_table_preserves_nullable_and_scalar_fields(tmp_path):
    rows = {
        "arc": [
            {"flag": True, "index": 1, "delta_m": 1.0e-15, "label": "station A"},
            {"flag": False, "index": 2, "label": "station B"},
        ]
    }
    path = write_observation_results(rows, tmp_path / "rows.txt")
    restored = read_observation_results(path)
    assert restored[0]["delta_m"] == 1.0e-15
    assert restored[1]["delta_m"] is None
    assert restored[0]["flag"] is True
    assert restored[0]["label"] == "station A"


def test_prediction_reader_checks_schema_and_keeps_output_precision(tmp_path):
    from lunarops.fileio.prediction_results import read_prediction_windows, write_prediction_windows

    row = {
        "station": "S",
        "reflector": "R",
        "start_utc": "start",
        "end_utc": "end",
        "start_local": "start",
        "end_local": "end",
        "sample_count": 2,
        "duration_s": 1.123456789,
    }
    path = write_prediction_windows([row], tmp_path / "window.txt")
    assert read_prediction_windows(path)[0]["duration_s"] == 1.123457
    text = path.read_text().replace("field duration_s float s", "field duration_s float m")
    path.write_text(text)
    with pytest.raises(ValueError, match="unit"):
        read_prediction_windows(path)
