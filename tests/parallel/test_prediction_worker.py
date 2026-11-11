"""An MPI worker builds the predictor from the broadcast spec and evaluates its grid slice."""

import numpy as np

from lunarops.classes.observation import LlrObservationPredictor, PredictionCriteria, PredictionMeteorology
from lunarops.classes.observation.assembly import build_observation_runtime
from lunarops.classes.observation.catalogs import ReflectorRecord, StationRecord
from lunarops.parallel import mpi
from tests.support.observation import TRANSMIT_UTC, StaticKernelContext

MODELS = {
    "ephemerides": {"type": "chebyshev", "directory": "kernels", "lunarRelativisticScaleConvention": "alreadyScaled"},
    "earthRotation": {"type": "file", "file": "eop.txt"},
    "troposphere": {"type": "none"},
    "relativity": {"type": "none"},
    "stationDisplacement": [{"type": "none"}],
    "reflectorDisplacement": {"type": "none"},
}
STATIONS = {"S": StationRecord(name="S", itrf_xyz_m=(6_378_137.0, 0.0, 0.0))}
REFLECTORS = {"R": ReflectorRecord(name="R", moon_fixed_xyz_m=(1_737_400.0, 0.0, 0.0))}
CRITERIA = {
    "minimum_elevation_deg": 0.0,
    "minimum_reflector_elevation_deg": -90.0,
    "maximum_sun_elevation_deg": 90.0,
    "allowed_elongation_ranges_deg": ((0.0, 360.0),),
}
METEOROLOGY = {"pressure_hpa": 900.0, "temperature_k": 285.0, "relative_humidity_percent": 25.0, "wavelength_nm": 532.0}


def _runtime(tmp_path):
    context = StaticKernelContext(working_dir=tmp_path)
    context.resolve_defaults()
    return context, build_observation_runtime(
        context, dict(MODELS), station_catalog=STATIONS, reflector_catalog=REFLECTORS
    )


def test_worker_prediction_rows_equal_the_serial_predictor(tmp_path, monkeypatch):
    context, runtime = _runtime(tmp_path)
    # The worker's runtime builder is its injection point; the predictor and task handler are real.
    monkeypatch.setattr(mpi, "build_worker_observation_runtime", lambda spec, cache: (context, runtime))
    state = mpi.WorkerState()
    spec = {
        "prediction": {
            "station": "S",
            "reflector": "R",
            "criteria": CRITERIA,
            "meteorology": METEOROLOGY,
            "utcOffsetHours": 8.0,
        }
    }
    state.initialize("prediction", 0, spec)
    payload = {
        "specId": 0,
        "startEpoch": TRANSMIT_UTC,
        "startIndex": 2,
        "recordCount": 3,
        "stepSeconds": 30.0,
        "taskId": 0,
    }

    result = mpi.TASK_HANDLERS["prediction"](payload, state)

    serial = LlrObservationPredictor(
        runtime.frames,
        runtime.light_time_solver,
        STATIONS["S"],
        REFLECTORS["R"],
        station_key="S",
        reflector_key="R",
        criteria=PredictionCriteria(**CRITERIA),
        meteorology=PredictionMeteorology(**METEOROLOGY),
        utc_offset_hours=8.0,
    )
    assert [item["index"] for item in result["items"]] == [2, 3, 4]
    for item in result["items"]:
        expected = serial.evaluate(TRANSMIT_UTC.shifted(item["index"] * 30.0))
        assert item["row"].keys() == expected.keys()
        for key, value in expected.items():
            assert item["row"][key] == value or (isinstance(value, float) and np.isclose(item["row"][key], value))
