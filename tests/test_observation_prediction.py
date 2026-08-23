from __future__ import annotations

import numpy as np

from lunarops.classes.displacement.terrestrial_geometry import enu2itrf, itrf2enu
from lunarops.classes.observation.prediction import PredictionCriteria, build_visibility_windows
from lunarops.fileio.prediction_results import (
    read_prediction_results,
    read_prediction_windows,
    write_prediction_results,
    write_prediction_windows,
)


def test_enu_and_itrf_rotations_are_inverses():
    enu = np.array([12.5, -31.0, 4.25])
    itrf = enu2itrf(enu, latitude_rad=0.43, longitude_rad=-1.17)
    np.testing.assert_allclose(
        itrf2enu(itrf, latitude_rad=0.43, longitude_rad=-1.17),
        enu,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_prediction_criteria_supports_wraparound_elongation_ranges():
    criteria = PredictionCriteria(allowed_elongation_ranges_deg=((350.0, 10.0),))
    assert criteria.elongation_allowed(355.0)
    assert criteria.elongation_allowed(5.0)
    assert not criteria.elongation_allowed(180.0)


def test_visibility_windows_merge_only_consecutive_observable_grid_samples():
    rows = [
        {
            "utc_t1": "2025-01-01T00:00:00.000000000",
            "local_t1": "2025-01-01T08:00:00.000000000+08:00",
            "station": "S",
            "reflector": "R",
            "observable": True,
        },
        {
            "utc_t1": "2025-01-01T00:01:00.000000000",
            "local_t1": "2025-01-01T08:01:00.000000000+08:00",
            "station": "S",
            "reflector": "R",
            "observable": True,
        },
        {
            "utc_t1": "2025-01-01T00:02:00.000000000",
            "local_t1": "2025-01-01T08:02:00.000000000+08:00",
            "station": "S",
            "reflector": "R",
            "observable": False,
        },
        {
            "utc_t1": "2025-01-01T00:03:00.000000000",
            "local_t1": "2025-01-01T08:03:00.000000000+08:00",
            "station": "S",
            "reflector": "R",
            "observable": True,
        },
    ]
    windows = build_visibility_windows(rows, step_seconds=60.0)
    assert windows == [
        {
            "station": "S",
            "reflector": "R",
            "start_utc": "2025-01-01T00:00:00.000000000",
            "end_utc": "2025-01-01T00:01:00.000000000",
            "start_local": "2025-01-01T08:00:00.000000000+08:00",
            "end_local": "2025-01-01T08:01:00.000000000+08:00",
            "sample_count": 2,
            "duration_s": 60.0,
        },
        {
            "station": "S",
            "reflector": "R",
            "start_utc": "2025-01-01T00:03:00.000000000",
            "end_utc": "2025-01-01T00:03:00.000000000",
            "start_local": "2025-01-01T08:03:00.000000000+08:00",
            "end_local": "2025-01-01T08:03:00.000000000+08:00",
            "sample_count": 1,
            "duration_s": 0.0,
        },
    ]


def test_prediction_artifacts_round_trip(tmp_path):
    prediction_rows = [
        {
            "utc_t1": "2025-01-01T00:00:00.000000000",
            "station": "S 1",
            "reflector": "R/1",
            "local_t1": "2025-01-01T08:00:00.000000000+08:00",
            "station_itrf_x_m": 1.0,
            "station_itrf_y_m": 2.0,
            "station_itrf_z_m": 3.0,
            "reflector_itrf_x_m": 4.0,
            "reflector_itrf_y_m": 5.0,
            "reflector_itrf_z_m": 6.0,
            "range_up_geometric_m": 7.0,
            "azimuth_deg": 9.0,
            "elevation_deg": 10.0,
            "observable": True,
        }
    ]
    windows = [
        {
            "station": "S 1",
            "reflector": "R/1",
            "start_utc": prediction_rows[0]["utc_t1"],
            "end_utc": prediction_rows[0]["utc_t1"],
            "start_local": prediction_rows[0]["local_t1"],
            "end_local": prediction_rows[0]["local_t1"],
            "sample_count": 1,
            "duration_s": 0.0,
        }
    ]
    prediction_rows.append({**prediction_rows[0], "utc_t1": "2025-01-01T00:05:00.000000000"})
    prediction_path = tmp_path / "prediction.txt"
    window_path = tmp_path / "windows.txt"
    write_prediction_results(prediction_rows, prediction_path)
    write_prediction_windows(windows, window_path)
    restored_prediction = read_prediction_results(prediction_path)
    restored_windows = read_prediction_windows(window_path)
    assert restored_prediction[0]["station"] == "S 1"
    assert restored_prediction[0]["reflector"] == "R/1"
    assert restored_prediction[0]["observable"] is True
    assert restored_prediction[0]["reflector_itrf_x_m"] == 4.0
    assert restored_prediction[0]["range_up_geometric_m"] == 7.0
    assert restored_prediction[0]["local_t1"] == "2025-01-01T08:00:00.000000000+08:00"
    assert len(restored_prediction) == 2
    assert restored_windows == windows
