from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import lunarops.parallel.mpi as mpi_module
from lunarops.classes.displacement.terrestrial_geometry import enu2itrf, itrf2enu
from lunarops.classes.observation.prediction import PredictionCriteria, build_visibility_windows
from lunarops.classes.time import Epoch, TimeScale, format_time_with_utc_offset
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


class _PredictionMpiRuntime:
    def __init__(self) -> None:
        self.spec: dict | None = None
        self.payloads: list[dict] = []

    def prepare_observation_spec(self, spec: dict) -> bool:
        self.spec = spec
        return True

    def map_tasks(self, kind: str, payloads: list[dict], **_kwargs):
        assert kind == "prediction"
        assert self.spec is not None
        self.payloads = payloads
        cache = {("observationSpec", self.spec["specId"]): self.spec}
        results = [mpi_module._handle_prediction(dict(payload, taskId=index), cache) for index, payload in enumerate(payloads)]
        return list(reversed(results))


def test_mpi_prediction_rows_restore_serial_time_order(monkeypatch):
    class Predictor:
        def __init__(self, _frames, _solver, _station, _reflector, **options) -> None:
            self.utc_offset_hours = options["utc_offset_hours"]

        def evaluate(self, epoch: Epoch) -> dict[str, object]:
            return {
                "utc_t1": epoch.isot(precision=3),
                "local_t1": format_time_with_utc_offset(
                    epoch,
                    utc_offset_hours=self.utc_offset_hours,
                    precision=3,
                ),
                "station": "S",
                "reflector": "R",
                "observable": True,
            }

    def build_runtime(spec, _shared_cache, *, context=None):
        return context or object(), SimpleNamespace(
            frames=object(),
            light_time_solver=object(),
            station_catalog=spec["stationCatalog"],
            reflector_catalog=spec["reflectorCatalog"],
        )

    monkeypatch.setattr("lunarops.classes.observation.LlrObservationPredictor", Predictor)
    monkeypatch.setattr(mpi_module, "build_worker_observation_runtime", build_runtime)
    epochs = [Epoch.from_isot("2025-01-01T00:00:00", scale=TimeScale.UTC).shifted(second) for second in (0, 1, 2)]
    runtime = _PredictionMpiRuntime()
    rows = mpi_module.mpi_prediction_rows(
        runtime,
        {
            "specId": "prediction-test",
            "stationCatalog": {"S": object()},
            "reflectorCatalog": {"R": object()},
        },
        epochs,
        station="S",
        reflector="R",
        criteria={
            "minimum_elevation_deg": 0.0,
            "minimum_reflector_elevation_deg": 0.0,
            "maximum_sun_elevation_deg": 0.0,
            "allowed_elongation_ranges_deg": ((0.0, 360.0),),
        },
        meteorology={
            "pressure_hpa": 900.0,
            "temperature_k": 285.0,
            "relative_humidity_percent": 25.0,
            "wavelength_nm": 532.0,
        },
        utc_offset_hours=8.0,
        chunksize=1,
        quiet=True,
    )

    assert len(runtime.payloads) == 3
    assert [row["utc_t1"] for row in rows] == [epoch.isot(precision=3) for epoch in epochs]


def test_prediction_program_does_not_build_a_rank_zero_runtime_in_mpi_mode(monkeypatch, tmp_path):
    from lunarops.programs import llr_observation_prediction as prediction_program

    class Runtime:
        has_workers = True

    class Context:
        runtime = Runtime()

        def resolve_path(self, value):
            return tmp_path / str(value)

    row = {
        "utc_t1": "2025-01-01T00:00:00.000",
        "local_t1": "2025-01-01T08:00:00.000+08:00",
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
        "observable": True,
    }

    def unexpected_rank_zero_runtime(*_args, **_kwargs):
        raise AssertionError("rank 0 must not initialize the prediction runtime in MPI mode")

    monkeypatch.setattr(prediction_program, "build_observation_runtime", unexpected_rank_zero_runtime)
    monkeypatch.setattr(
        mpi_module,
        "make_observation_spec",
        lambda *_args, **_kwargs: {
            "stationCatalog": {"S": object()},
            "reflectorCatalog": {"R": object()},
        },
    )
    monkeypatch.setattr(mpi_module, "mpi_prediction_rows", lambda *_args, **_kwargs: [row])

    result = prediction_program.llr_observation_prediction(
        {
            "startTime": "2025-01-01T08:00:00",
            "endTime": "2025-01-01T08:00:00",
            "utcOffsetHours": 8.0,
            "stepSeconds": 1.0,
            "stationName": "S",
            "reflectorName": "R",
            "minElevationDeg": 0.0,
            "minReflectorElevationDeg": 0.0,
            "maxSunElevationDeg": 0.0,
            "allowedElongationRangesDeg": [{"startDeg": 0.0, "endDeg": 360.0}],
            "pressureHpa": 900.0,
            "temperatureK": 285.0,
            "relativeHumidityPercent": 25.0,
            "wavelengthNm": 532.0,
            "showProgress": False,
            "outputFilePrediction": "prediction.txt",
            "outputFileWindows": "windows.txt",
        },
        Context(),
    )

    assert result["rows"] == [row]
