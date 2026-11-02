import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from lunarops.classes.frames import EarthOrientationSample, TabulatedEarthOrientation
from lunarops.classes.observation.assembly import resolve_observation_assembly
from lunarops.classes.observation.catalogs import ReflectorRecord, StationRecord
from lunarops.config.context import RunContext
from lunarops.fileio.catalogs import (
    write_reflector_catalog,
    write_station_catalog,
)
from lunarops.parallel.observation_spec import _prepare_shared_resources, make_observation_spec


def test_native_file_eop_is_serialized_into_the_mpi_spec():
    earth_orientation = TabulatedEarthOrientation(
        (EarthOrientationSample(60000.0, 0.1, 0.2, -0.3),),
        source_file_path="eop.txt",
    )

    class Context:
        def create_class(self, category, config, *, cache):
            assert category == "earthRotation"
            assert config == {"type": "file", "file": "eop.txt"}
            assert cache is True
            return earth_orientation

    resources = _prepare_shared_resources({"earthRotation": {"type": "file", "file": "eop.txt"}}, Context())
    assert resources["earthRotation"]["mjdUtc"].tolist() == [60000.0]

def test_observation_specs_are_unique_and_use_explicit_catalogs():
    context = RunContext(
        working_dir=".",
        observation_model_configs={
            key: {"type": "test"}
            for key in (
                "ephemerides",
                "earthRotation",
                "troposphere",
                "relativity",
                "stationDisplacement",
                "reflectorDisplacement",
            )
        },
    )
    first_stations = {"station": object()}
    first_reflectors = {"reflector": object()}
    second_stations = {"other-station": object()}
    second_reflectors = {"other-reflector": object()}

    first = make_observation_spec(
        {},
        context,
        station_catalog=first_stations,
        reflector_catalog=first_reflectors,
    )
    second = make_observation_spec(
        {},
        context,
        station_catalog=second_stations,
        reflector_catalog=second_reflectors,
    )

    assert first["specId"] != second["specId"]
    assert first["stationCatalog"] is first_stations
    assert second["stationCatalog"] is second_stations
    assert first["reflectorCatalog"] is first_reflectors
    assert second["reflectorCatalog"] is second_reflectors

def test_serial_and_mpi_catalog_resolution_share_working_directory(tmp_path):
    write_station_catalog(
        {"TEST": StationRecord("Test station", [1.0, 2.0, 3.0])},
        tmp_path / "stations.txt",
    )
    write_reflector_catalog(
        {"REF": ReflectorRecord("Test reflector", [4.0, 5.0, 6.0])},
        tmp_path / "reflectors.txt",
    )
    config = {
        "inputFileStationCatalog": "stations.txt",
        "inputFileReflectorCatalog": "reflectors.txt",
    }
    context = RunContext(
        working_dir=tmp_path,
        observation_model_configs={
            key: {"type": "test"}
            for key in (
                "ephemerides",
                "earthRotation",
                "troposphere",
                "relativity",
                "stationDisplacement",
                "reflectorDisplacement",
            )
        },
    )

    serial = resolve_observation_assembly(context, config)
    mpi = make_observation_spec(config, context)

    assert set(serial.station_catalog) == {"TEST"}
    assert set(serial.reflector_catalog) == {"REF"}
    assert set(mpi["stationCatalog"]) == set(serial.station_catalog)
    assert set(mpi["reflectorCatalog"]) == set(serial.reflector_catalog)


_PUBLIC_API_RUNNER = """
import json
from types import SimpleNamespace
from lunarops.parallel import mpi

class Context:
    def __init__(self, cache):
        self.cache = cache
        self.closed = False
    def close(self):
        self.closed = True

contexts = []
def build(spec, cache):
    if spec["fail"]:
        raise ValueError("requested initialization failure")
    context = Context(cache)
    contexts.append(context)
    return context, SimpleNamespace(name=spec["name"])
mpi.build_worker_processor = build

def probe(payload, state):
    if payload["fail"]:
        raise ValueError("requested task failure")
    return state.processors[payload["specId"]].name, payload["value"]
mpi.TASK_HANDLERS["probe"] = probe
runtime = mpi.MpiRuntime()
if runtime.is_master:
    try:
        runtime.broadcast_spec("observationSpec", {"specId": "bad", "fail": True})
    except RuntimeError as exc:
        assert "requested initialization failure" in str(exc)
    else:
        raise AssertionError("initialization failure did not propagate")
    first = runtime.broadcast_spec("observationSpec", {"specId": "first", "fail": False, "name": "first"})
    assert first == runtime.broadcast_spec("observationSpec", {"specId": "first"})
    second = runtime.broadcast_spec("observationSpec", {"specId": "second", "fail": False, "name": "second"})
    assert first != second
    payloads = [{"specId": first, "value": i, "fail": False} for i in range(15)]
    expected = [("first", i) for i in range(15)]
    assert runtime.map_tasks("probe", payloads) == expected
    assert list(runtime.stream_tasks("probe", payloads, progress="probe")) == expected
    assert runtime.map_tasks("probe", []) == []
    stream = runtime.stream_tasks("probe", payloads)
    assert next(stream) == expected[0]
    stream.close()
    assert runtime.map_tasks("probe", payloads) == expected
    broken = [dict(p, fail=i == 2) for i, p in enumerate(payloads)]
    try:
        runtime.map_tasks("probe", broken)
    except RuntimeError as exc:
        assert "requested task failure" in str(exc)
    else:
        raise AssertionError("task failure did not propagate")
    assert runtime.map_tasks("probe", [{"specId": second, "value": 99, "fail": False}]) == [("second", 99)]
    runtime.shutdown()
else:
    runtime.worker_loop()
closed = runtime.comm.gather((len(contexts), all(c.closed for c in contexts),
                             len({id(c.cache) for c in contexts})), root=0)
if runtime.is_master:
    assert closed[1:] == [(2, True, 1)] * (runtime.size - 1)
    print(json.dumps({"public_api_pass": True, "workers_closed": len(closed) - 1}), flush=True)
"""


def test_public_mpi_api_initialization_ordering_errors_drain_and_shutdown():
    launcher = shutil.which("mpirun")
    if launcher is None or importlib.util.find_spec("mpi4py") is None:
        pytest.skip("MPI runtime and mpi4py are required")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [launcher, "-n", "3", sys.executable, "-c", _PUBLIC_API_RUNNER],
        cwd=root, env=dict(os.environ, OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1"),
        capture_output=True, text=True, timeout=45, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(result.stdout.splitlines()[-1])
    assert summary == {"public_api_pass": True, "workers_closed": 2}
