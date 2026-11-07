"""Prediction criteria, visibility windows, prediction files and their merge."""

import pytest

from lunarops.classes.observation.prediction import PredictionCriteria, build_visibility_windows
from lunarops.classes.time import Epoch, TimeScale, format_time_with_utc_offset
from lunarops.config.context import RunContext
from lunarops.fileio.prediction_results import (
    read_prediction_results,
    read_prediction_windows,
    write_prediction_results,
    write_prediction_windows,
)
from lunarops.parallel.mpi import mpi_prediction_rows
from lunarops.programs.llr_observation_prediction_merge import llr_observation_prediction_merge


def _row(utc_t1: str, *, observable: bool = True, **changes) -> dict[str, object]:
    return {
        "utc_t1": utc_t1,
        "local_t1": format_time_with_utc_offset(Epoch.from_isot(utc_t1), utc_offset_hours=8.0, precision=3),
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
        **changes,
    }


def _window(start, end, count, duration):
    local = {
        value: format_time_with_utc_offset(Epoch.from_isot(value), utc_offset_hours=8.0, precision=3)
        for value in (start, end)
    }
    return {
        "station": "S",
        "reflector": "R",
        "start_utc": start,
        "end_utc": end,
        "start_local": local[start],
        "end_local": local[end],
        "sample_count": count,
        "duration_s": duration,
    }


def test_elongation_ranges_may_wrap_through_zero():
    criteria = PredictionCriteria(allowed_elongation_ranges_deg=((350.0, 10.0),))
    assert criteria.elongation_allowed(355.0) and criteria.elongation_allowed(5.0)
    assert not criteria.elongation_allowed(180.0)


def test_windows_join_only_consecutive_observable_samples():
    rows = [_row(f"2025-01-01T00:0{minute}:00.000", observable=minute != 2) for minute in range(4)]
    assert build_visibility_windows(rows, step_seconds=60.0) == [
        _window("2025-01-01T00:00:00.000", "2025-01-01T00:01:00.000", 2, 60.0),
        _window("2025-01-01T00:03:00.000", "2025-01-01T00:03:00.000", 1, 0.0),
    ]


def test_prediction_and_window_files_round_trip(tmp_path):
    rows = [
        _row("2025-01-01T00:00:00.000", station="S 1", reflector="R/1"),
        _row("2025-01-01T00:05:00.000", station="S 1", reflector="R/1"),
    ]
    windows = [_window("2025-01-01T00:00:00.000", "2025-01-01T00:00:00.000", 1, 0.0)]
    write_prediction_results(rows, tmp_path / "prediction.txt")
    write_prediction_windows(windows, tmp_path / "windows.txt")

    assert read_prediction_results(tmp_path / "prediction.txt") == rows
    assert read_prediction_windows(tmp_path / "windows.txt") == windows


def test_window_files_keep_six_decimals_and_check_units(tmp_path):
    window = {
        "station": "S",
        "reflector": "R",
        "start_utc": "start",
        "end_utc": "end",
        "start_local": "start",
        "end_local": "end",
        "sample_count": 2,
        "duration_s": 1.123456789,
    }
    path = write_prediction_windows([window], tmp_path / "window.txt")
    assert read_prediction_windows(path)[0]["duration_s"] == 1.123457
    path.write_text(path.read_text().replace("field duration_s float s", "field duration_s float m"))
    with pytest.raises(ValueError, match="unit"):
        read_prediction_windows(path)


def test_merge_sorts_rows_and_rebuilds_windows(tmp_path):
    write_prediction_results([_row("2025-01-01T00:00:00.000"), _row("2025-01-01T00:00:05.000")], tmp_path / "early.txt")
    write_prediction_results(
        [_row("2025-01-01T00:00:10.000", observable=False), _row("2025-01-01T00:00:15.000")], tmp_path / "late.txt"
    )
    config = {
        "inputFilesPrediction": ["late.txt", "early.txt"],
        "outputFilePrediction": "merged.txt",
        "outputFileWindows": "windows.txt",
        "stepSeconds": 5.0,
    }

    result = llr_observation_prediction_merge(config, RunContext(working_dir=tmp_path))

    expected = [f"2025-01-01T00:00:{second:02d}.000" for second in (0, 5, 10, 15)]
    assert [row["utc_t1"] for row in result["rows"]] == expected
    assert [row["utc_t1"] for row in read_prediction_results(tmp_path / "merged.txt")] == expected
    assert read_prediction_windows(tmp_path / "windows.txt") == [
        _window("2025-01-01T00:00:00.000", "2025-01-01T00:00:05.000", 2, 5.0),
        _window("2025-01-01T00:00:15.000", "2025-01-01T00:00:15.000", 1, 0.0),
    ]


def test_merge_rejects_duplicate_epochs(tmp_path):
    for name in ("first.txt", "second.txt"):
        write_prediction_results([_row("2025-01-01T00:00:00.000")], tmp_path / name)
    config = {
        "inputFilesPrediction": ["first.txt", "second.txt"],
        "outputFilePrediction": "merged.txt",
        "outputFileWindows": "windows.txt",
        "stepSeconds": 5.0,
    }
    with pytest.raises(ValueError, match="duplicate utc_t1"):
        llr_observation_prediction_merge(config, RunContext(working_dir=tmp_path))


def test_mpi_prediction_rows_restore_time_order_from_out_of_order_chunks():
    class Runtime:
        """Answers every chunk, in reverse order, with rows labelled by grid index."""

        def broadcast_spec(self, kind, spec):
            return 0

        def map_tasks(self, kind, payloads, **_):
            answers = [
                {
                    "items": [
                        {"index": i, "row": {"index": i}}
                        for i in range(p["startIndex"], p["startIndex"] + p["recordCount"])
                    ]
                }
                for p in payloads
            ]
            return answers[::-1]

    epochs = [Epoch.from_isot("2025-01-01T00:00:00", scale=TimeScale.UTC).shifted(s) for s in range(5)]
    rows = mpi_prediction_rows(
        Runtime(),
        {"specId": "prediction"},
        epochs,
        step_seconds=1.0,
        station="S",
        reflector="R",
        criteria={},
        meteorology={},
        utc_offset_hours=0.0,
        chunksize=2,
        quiet=True,
    )

    assert [row["index"] for row in rows] == [0, 1, 2, 3, 4]
