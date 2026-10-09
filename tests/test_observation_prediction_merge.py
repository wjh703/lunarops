from __future__ import annotations

import pytest

from lunarops.config.context import RunContext
from lunarops.classes.time import Epoch, format_time_with_utc_offset
from lunarops.fileio.prediction_results import (
    read_prediction_results,
    read_prediction_windows,
    write_prediction_results,
)
from lunarops.programs.llr_observation_prediction_merge import llr_observation_prediction_merge


def _row(utc_t1: str, *, observable: bool) -> dict[str, object]:
    return {
        "utc_t1": utc_t1,
        "local_t1": format_time_with_utc_offset(
            Epoch.from_isot(utc_t1),
            utc_offset_hours=8.0,
            precision=3,
        ),
        "station": "S",
        "reflector": "R",
        "station_itrf_x_m": 1.0,
        "station_itrf_y_m": 2.0,
        "station_itrf_z_m": 3.0,
        "reflector_itrf_x_m": 4.0,
        "reflector_itrf_y_m": 5.0,
        "reflector_itrf_z_m": 6.0,
        "range_up_geometric_m": 7.0,
        "azimuth_deg": 8.0,
        "elevation_deg": 9.0,
        "observable": observable,
    }


def _merge_config(*inputs: str) -> dict[str, object]:
    return {
        "inputFilesPrediction": list(inputs),
        "outputFilePrediction": "merged_prediction.txt",
        "outputFileWindows": "merged_windows.txt",
        "stepSeconds": 5.0,
    }


def test_prediction_merge_sorts_rows_and_rebuilds_windows(tmp_path):
    early = tmp_path / "early.txt"
    late = tmp_path / "late.txt"
    write_prediction_results(
        [
            _row("2025-01-01T00:00:00.000", observable=True),
            _row("2025-01-01T00:00:05.000", observable=True),
        ],
        early,
    )
    write_prediction_results(
        [
            _row("2025-01-01T00:00:10.000", observable=False),
            _row("2025-01-01T00:00:15.000", observable=True),
        ],
        late,
    )

    result = llr_observation_prediction_merge(
        _merge_config(late.name, early.name),
        RunContext(working_dir=tmp_path),
    )

    assert [row["utc_t1"] for row in result["rows"]] == [
        "2025-01-01T00:00:00.000",
        "2025-01-01T00:00:05.000",
        "2025-01-01T00:00:10.000",
        "2025-01-01T00:00:15.000",
    ]
    assert [row["utc_t1"] for row in read_prediction_results(tmp_path / "merged_prediction.txt")] == [
        "2025-01-01T00:00:00.000",
        "2025-01-01T00:00:05.000",
        "2025-01-01T00:00:10.000",
        "2025-01-01T00:00:15.000",
    ]
    assert read_prediction_windows(tmp_path / "merged_windows.txt") == [
        {
            "station": "S",
            "reflector": "R",
            "start_utc": "2025-01-01T00:00:00.000",
            "end_utc": "2025-01-01T00:00:05.000",
            "start_local": "2025-01-01T08:00:00.000+08:00",
            "end_local": "2025-01-01T08:00:05.000+08:00",
            "sample_count": 2,
            "duration_s": 5.0,
        },
        {
            "station": "S",
            "reflector": "R",
            "start_utc": "2025-01-01T00:00:15.000",
            "end_utc": "2025-01-01T00:00:15.000",
            "start_local": "2025-01-01T08:00:15.000+08:00",
            "end_local": "2025-01-01T08:00:15.000+08:00",
            "sample_count": 1,
            "duration_s": 0.0,
        },
    ]


def test_prediction_merge_rejects_duplicate_epochs(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    row = _row("2025-01-01T00:00:00.000", observable=True)
    write_prediction_results([row], first)
    write_prediction_results([row], second)

    with pytest.raises(ValueError, match="duplicate utc_t1"):
        llr_observation_prediction_merge(
            _merge_config(first.name, second.name),
            RunContext(working_dir=tmp_path),
        )
