"""LlrObservationPrediction end to end on the static synthetic system."""

import pytest

from lunarops.classes.observation import ReflectorRecord, StationRecord
from lunarops.fileio.catalogs import write_reflector_catalog, write_station_catalog
from lunarops.fileio.prediction_results import read_prediction_results, read_prediction_windows
from lunarops.programs.llr_observation_prediction import llr_observation_prediction
from lunarops.programs.registry import ensure_builtin_programs, resolve_program_config
from tests.support.observation import EARTH_MOON_M, StaticKernelContext, static_frames, station_facing_the_moon

CONFIG = {
    "inputFileStationCatalog": "stations.txt",
    "inputFileReflectorCatalog": "reflectors.txt",
    "outputFilePrediction": "prediction.txt",
    "outputFileWindows": "windows.txt",
    "meteorologyPolicy": "nominal",
    "startTime": "2020-01-01T08:00:00",
    "endTime": "2020-01-01T08:02:00",
    "utcOffsetHours": 8.0,
    "stationName": "APOLLO",
    "reflectorName": "APOLLO15",
    "minElevationDeg": 0.0,
    "minReflectorElevationDeg": -90.0,
    "maxSunElevationDeg": 90.0,
    "showProgress": False,
    "ephemerides": {"type": "chebyshev", "directory": "kernels", "lunarRelativisticScaleConvention": "alreadyScaled"},
    "earthRotation": {"type": "file", "file": "eop.txt"},
    "troposphere": {"type": "none"},
    "relativity": {"type": "none"},
    "stationDisplacement": [{"type": "iers2010SolidEarthTide"}, {"type": "none"}],
    "reflectorDisplacement": {"type": "none"},
}


@pytest.fixture
def prediction(tmp_path):
    ensure_builtin_programs()
    write_station_catalog(
        {"APOLLO": StationRecord(name="Apache Point", itrf_xyz_m=tuple(station_facing_the_moon(static_frames())))},
        tmp_path / "stations.txt",
    )
    write_reflector_catalog(
        {"APOLLO15": ReflectorRecord(name="Apollo 15", moon_fixed_xyz_m=(1_737_400.0, 0.0, 0.0))},
        tmp_path / "reflectors.txt",
    )
    config = resolve_program_config("LlrObservationPrediction", CONFIG)
    context = StaticKernelContext(working_dir=tmp_path)
    context.resolve_defaults()  # as the CLI does before running programs
    return tmp_path, llr_observation_prediction(config, context)


def test_serial_prediction_writes_a_utc_grid_with_local_times(prediction):
    root, result = prediction
    rows = read_prediction_results(root / "prediction.txt")

    assert [row["utc_t1"] for row in rows] == [f"2020-01-01T00:0{minute}:00.000" for minute in range(3)]
    assert [row["local_t1"] for row in rows] == [f"2020-01-01T08:0{minute}:00.000+08:00" for minute in range(3)]
    # The file carries the published columns of each row; floats are rounded to at least millimetres.
    for written, computed in zip(rows, result["rows"], strict=True):
        for key, value in written.items():
            if isinstance(value, float):
                assert abs(value - computed[key]) <= 5e-4, key
            else:
                assert value == computed[key], key
    # The station looks straight at the Moon, about one Earth radius plus the lunar radius closer than its centre.
    assert all(abs(row["range_up_geometric_m"] - (EARTH_MOON_M - 6_378_137.0 + 1_737_400.0)) < 1e3 for row in rows)


def test_serial_prediction_writes_one_window_for_an_always_visible_grid(prediction):
    root, result = prediction
    windows = read_prediction_windows(root / "windows.txt")

    assert windows == result["windows"]
    assert [(w["start_utc"], w["end_utc"], w["sample_count"]) for w in windows] == [
        ("2020-01-01T00:00:00.000", "2020-01-01T00:02:00.000", 3)
    ]
